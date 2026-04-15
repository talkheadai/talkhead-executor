from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional

from loguru import logger
from executor.scoring.efficiency import EfficiencyConfig, aggregate_efficiency_scores
from executor.scoring.audio import compute_audio_score
from executor.scoring.identity import compute_identity_score
from executor.scoring.lipsync import compute_lipsync_score
from executor.scoring.penalties import compute_penalties
from executor.scoring.temporal import compute_temporal_score
from executor.scoring.utils import (
    clamp01,
    extract_audio_from_video,
    load_wav_mono_16k,
    read_video_frames,
    safe_run,
)
from executor.scoring.video import compute_video_quality_score

SYNC_C_THRESHOLD = 0.45


def _round_scores(x: dict) -> dict:
    out = {}
    for k, v in x.items():
        if isinstance(v, dict):
            out[k] = _round_scores(v)
        elif isinstance(v, float):
            out[k] = round(v, 6)
        else:
            out[k] = v
    return out


def score_video(
    video_path: str,
    reference_image_path: str,
    audio_path: str,
    expected_transcript: Optional[str],
    *,
    peak_vram_gb: float | None = None,
    inference_time_sec: float | None = None,
    efficiency_config: EfficiencyConfig | None = None,
) -> dict:
    """
    Score a talking-head generation output with multi-metric evaluation.

    Returns a structured dictionary:
    {
      "identity": float,
      "lipsync": float,
      "audio": float,
      "video": float,
      "temporal": float,
      "penalty": float,
      "quality_score": float,
      "final": float,
      "efficiency": {...},
      "debug": {...}
    }
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"video_path does not exist: {video_path}")
    if not Path(reference_image_path).exists():
        raise FileNotFoundError(
            f"reference_image_path does not exist: {reference_image_path}"
        )
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"audio_path does not exist: {audio_path}")

    frames, _meta = read_video_frames(video_path, max_frames=80)

    video_audio_tmp = extract_audio_from_video(video_path)
    source_audio_signal, source_audio_sr = load_wav_mono_16k(audio_path)

    if video_audio_tmp:
        video_audio_signal, video_audio_sr = load_wav_mono_16k(video_audio_tmp)
    else:
        video_audio_signal, video_audio_sr = source_audio_signal * 0.0, source_audio_sr

    identity = safe_run(
        "identity",
        lambda: compute_identity_score(reference_image_path, frames),
        {
            "identity": 0.0,
            "face_detect_ratio": 0.0,
            "per_frame_identity": [],
            "face_bboxes": [],
            "identity_mode": "unavailable",
        },
    )

    lipsync = safe_run(
        "lipsync",
        lambda: compute_lipsync_score(
            frames=frames,
            face_bboxes=identity.get("face_bboxes", []),
            video_audio_signal=video_audio_signal,
            video_audio_sr=video_audio_sr,
        ),
        {
            "lipsync": 0.0,
            "sync_c": 0.0,
            "sync_d": 1.0,
            "audio_envelope": [],
            "mouth_signal": [],
            "sync_curve": [],
        },
    )

    audio = safe_run(
        "audio",
        lambda: compute_audio_score(
            source_audio_signal=source_audio_signal,
            source_audio_sr=source_audio_sr,
            video_audio_signal=video_audio_signal,
            video_audio_sr=video_audio_sr,
            expected_transcript=expected_transcript,
            video_audio_path_for_asr=video_audio_tmp,
            source_audio_path_for_asr=audio_path,
        ),
        {
            "audio": 0.0,
            "wer": 1.0,
            "tsim": 0.0,
            "audio_quality": 0.0,
            "video_audio_energy": 0.0,
            "source_audio_energy": 0.0,
        },
    )

    video = safe_run(
        "video",
        lambda: compute_video_quality_score(frames, identity.get("face_bboxes", [])),
        {"video": 0.0, "aesthetic_score": 0.0, "motion_naturalness": 0.0},
    )

    temporal = safe_run(
        "temporal",
        lambda: compute_temporal_score(
            per_frame_identity=identity.get("per_frame_identity", []),
            face_bboxes=identity.get("face_bboxes", []),
            mouth_motion_signal=lipsync.get("mouth_signal", []),
        ),
        {
            "temporal": 0.0,
            "identity_temporal_consistency": 0.0,
            "motion_smoothness": 0.0,
        },
    )

    penalties = safe_run(
        "penalties",
        lambda: compute_penalties(
            frames=frames,
            face_bboxes=identity.get("face_bboxes", []),
            face_detect_ratio=float(identity.get("face_detect_ratio", 0.0)),
            audio_env=lipsync.get("audio_envelope", []),
            source_audio_energy=float(audio.get("source_audio_energy", 0.0)),
            video_audio_energy=float(audio.get("video_audio_energy", 0.0)),
            sync_per_frame=lipsync.get("sync_curve", []),
        ),
        {"penalty": 1.0},
    )

    gate = 1.0
    face_detect_ratio = float(identity.get("face_detect_ratio", 0.0))
    identity_mode = str(identity.get("identity_mode", "unavailable"))
    identity_score = float(identity.get("identity", 0.0))
    sync_c = float(lipsync.get("sync_c", 0.0))
    wer = float(audio.get("wer", 1.0))

    face_detect_gate_applied = identity_mode == "insightface"
    if face_detect_gate_applied and face_detect_ratio < 0.80:
        gate *= 0.2
    if identity_score < 0.35:
        gate *= 0.1
    if sync_c < SYNC_C_THRESHOLD:
        gate *= 0.2

    if wer > 0.60:
        quality_score = 0.0
    else:
        blended = (
            0.35 * identity_score
            + 0.35 * float(lipsync.get("lipsync", 0.0))
            + 0.15 * float(video.get("video", 0.0))
            + 0.10 * float(audio.get("audio", 0.0))
            + 0.05 * float(temporal.get("temporal", 0.0))
            - float(penalties.get("penalty", 1.0))
        )
        quality_score = clamp01(gate * blended)

    cfg = efficiency_config or EfficiencyConfig.from_env()
    efficiency = aggregate_efficiency_scores(
        per_challenge_peak_vram_gb=[peak_vram_gb],
        per_challenge_inference_sec=[inference_time_sec],
        mean_quality=quality_score,
        cfg=cfg,
    )

    out = {
        "identity": clamp01(identity_score),
        "lipsync": clamp01(float(lipsync.get("lipsync", 0.0))),
        "audio": clamp01(float(audio.get("audio", 0.0))),
        "video": clamp01(float(video.get("video", 0.0))),
        "temporal": clamp01(float(temporal.get("temporal", 0.0))),
        "penalty": clamp01(float(penalties.get("penalty", 1.0))),
        "quality_score": clamp01(quality_score),
        "final": clamp01(float(efficiency.get("final_score", quality_score))),
        "efficiency": {
            "peak_vram_gb": efficiency.get("peak_vram_gb"),
            "inference_time_sec": efficiency.get("inference_time_sec"),
            "time_norm": max(0.0, min(3.0, float(efficiency.get("time_norm", 0.0)))),
            "vram_norm": max(0.0, min(3.0, float(efficiency.get("vram_norm", 0.0)))),
            "efficiency_factor": clamp01(
                float(efficiency.get("efficiency_factor", 0.0))
            ),
            "cap_violation": bool(efficiency.get("cap_violation", False)),
        },
        "debug": {
            "sync_c": clamp01(sync_c),
            "sync_d": clamp01(float(lipsync.get("sync_d", 1.0))),
            "wer": clamp01(wer),
            "face_detect_ratio": clamp01(face_detect_ratio),
            "identity_mode": identity_mode,
            "face_detect_gate_applied": face_detect_gate_applied,
        },
    }
    logger.info(f"out: {out}")

    with contextlib.suppress(Exception):
        if video_audio_tmp:
            Path(video_audio_tmp).unlink(missing_ok=True)

    return _round_scores(out)
