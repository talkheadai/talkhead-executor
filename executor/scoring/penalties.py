from __future__ import annotations

from typing import Any

from executor.scoring.utils import clamp01

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


DEFAULT_PENALTY_WEIGHTS = {
    "freeze": 0.18,
    "loop": 0.12,
    "face_missing": 0.22,
    "offscreen": 0.12,
    "mouth_occlusion": 0.10,
    "silent_mismatch": 0.16,
    "desync_spikes": 0.10,
}


def _freeze_ratio(frames: list[Any]) -> float:
    if np is None or len(frames) < 2:
        return 0.0
    diffs = []
    for i in range(1, len(frames)):
        d = float(np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))))
        diffs.append(d)
    if not diffs:
        return 0.0
    freeze = float(np.mean(np.asarray(diffs) < 0.8))
    return clamp01(freeze)


def _loop_score(frames: list[Any]) -> float:
    if np is None or len(frames) < 12:
        return 0.0
    n = len(frames)
    k = max(3, n // 5)
    a = np.asarray([f.mean() for f in frames[:k]], dtype=np.float32)
    b = np.asarray([f.mean() for f in frames[-k:]], dtype=np.float32)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return clamp01((c - 0.85) / 0.15) if c > 0.85 else 0.0


def _offscreen_ratio(face_bboxes: list[Any], frames: list[Any]) -> float:
    if np is None or not frames:
        return 0.0
    bad = 0
    seen = 0
    for i, bbox in enumerate(face_bboxes[: len(frames)]):
        if bbox is None:
            continue
        seen += 1
        h, w = frames[i].shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        margin = 0.02
        if x1 < margin * w or y1 < margin * h or x2 > (1.0 - margin) * w or y2 > (1.0 - margin) * h:
            bad += 1
    if seen == 0:
        return 1.0
    return bad / seen


def _mouth_occlusion(face_bboxes: list[Any], frames: list[Any], audio_env: list[float]) -> float:
    if np is None or not frames:
        return 0.0
    vals = []
    for i, bbox in enumerate(face_bboxes[: len(frames)]):
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h = max(1, y2 - y1)
        w = max(1, x2 - x1)
        mouth = frames[i][
            max(0, y1 + int(0.62 * h)) : max(0, y1 + int(0.96 * h)),
            max(0, x1 + int(0.20 * w)) : max(0, x1 + int(0.80 * w)),
        ]
        if mouth.size == 0:
            continue
        mouth_var = float(np.var(mouth.astype(np.float32) / 255.0))
        ae = audio_env[i] if i < len(audio_env) else 0.0
        if ae > 0.04 and mouth_var < 0.005:
            vals.append(1.0)
        else:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def _desync_spikes(sync_per_frame: list[float]) -> float:
    if np is None or len(sync_per_frame) < 8:
        return 0.0
    x = np.asarray(sync_per_frame, dtype=np.float32)
    if x.std() < 1e-8:
        return 0.0
    z = (x - x.mean()) / (x.std() + 1e-6)
    spikes = float(np.mean(z < -1.5))
    return clamp01(spikes / 0.4)


def compute_penalties(
    frames: list[Any],
    face_bboxes: list[Any],
    face_detect_ratio: float,
    audio_env: list[float],
    source_audio_energy: float,
    video_audio_energy: float,
    sync_per_frame: list[float],
) -> dict[str, float]:
    freeze = _freeze_ratio(frames)
    loop = _loop_score(frames)
    face_missing = clamp01((0.80 - face_detect_ratio) / 0.80) if face_detect_ratio < 0.80 else 0.0
    offscreen = _offscreen_ratio(face_bboxes, frames)
    mouth_occ = _mouth_occlusion(face_bboxes, frames, audio_env)
    silent_mismatch = 0.0
    if source_audio_energy > 1e-5:
        ratio = video_audio_energy / source_audio_energy
        if ratio < 0.15:
            silent_mismatch = clamp01((0.15 - ratio) / 0.15)
    desync = _desync_spikes(sync_per_frame)

    weighted = (
        DEFAULT_PENALTY_WEIGHTS["freeze"] * freeze
        + DEFAULT_PENALTY_WEIGHTS["loop"] * loop
        + DEFAULT_PENALTY_WEIGHTS["face_missing"] * face_missing
        + DEFAULT_PENALTY_WEIGHTS["offscreen"] * offscreen
        + DEFAULT_PENALTY_WEIGHTS["mouth_occlusion"] * mouth_occ
        + DEFAULT_PENALTY_WEIGHTS["silent_mismatch"] * silent_mismatch
        + DEFAULT_PENALTY_WEIGHTS["desync_spikes"] * desync
    )
    return {
        "penalty": clamp01(weighted),
        "freeze_penalty": freeze,
        "loop_penalty": loop,
        "face_missing_penalty": face_missing,
        "offscreen_penalty": offscreen,
        "mouth_occlusion_penalty": mouth_occ,
        "silent_mismatch_penalty": silent_mismatch,
        "desync_spike_penalty": desync,
    }
