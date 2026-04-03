from __future__ import annotations

import contextlib
from typing import Any

from executor.scoring.utils import clamp01, corrcoef, normalize_down, signal_energy

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


def _tokenize_text(s: str) -> list[str]:
    return [t for t in s.lower().strip().split() if t]


def _levenshtein_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def _wer(ref_text: str, hyp_text: str) -> float:
    ref = _tokenize_text(ref_text)
    hyp = _tokenize_text(hyp_text)
    if not ref:
        return 1.0 if hyp else 0.0
    return min(1.0, _levenshtein_distance(ref, hyp) / len(ref))


def _asr_whisper(path: str) -> str:
    import whisper  # type: ignore
    from executor.scoring.utils import cached_model

    model = cached_model("whisper_base", lambda: whisper.load_model("base"))
    result = model.transcribe(path, language="en", fp16=False)
    return str(result.get("text", "")).strip()


def _transcribe(path: str) -> str:
    with contextlib.suppress(Exception):
        return _asr_whisper(path)
    return ""


def _speaker_similarity(signal_a: Any, signal_b: Any) -> float:
    if np is None or signal_a is None or signal_b is None:
        return 0.0
    n = min(len(signal_a), len(signal_b))
    if n < 1600:
        return 0.0
    a = np.asarray(signal_a[:n], dtype=np.float32)
    b = np.asarray(signal_b[:n], dtype=np.float32)
    # Lightweight deterministic proxy when a dedicated speaker model is unavailable.
    chunk = max(400, n // 50)
    a_env = np.asarray([float(np.mean(np.abs(a[i : i + chunk]))) for i in range(0, n, chunk)], dtype=np.float32)
    b_env = np.asarray([float(np.mean(np.abs(b[i : i + chunk]))) for i in range(0, n, chunk)], dtype=np.float32)
    return clamp01((corrcoef(a_env, b_env) + 1.0) / 2.0)


def _audio_quality(signal: Any) -> float:
    if np is None or signal is None or len(signal) == 0:
        return 0.0
    x = np.asarray(signal, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x * x) + 1e-9))
    clipping = float(np.mean(np.abs(x) > 0.99))
    loudness_score = clamp01((rms - 0.01) / (0.15 - 0.01))
    clipping_penalty = clamp01(clipping / 0.05)
    return clamp01(0.8 * loudness_score + 0.2 * (1.0 - clipping_penalty))


def compute_audio_score(
    source_audio_signal: Any,
    source_audio_sr: int,
    video_audio_signal: Any,
    video_audio_sr: int,
    expected_transcript: str | None,
    video_audio_path_for_asr: str | None,
    source_audio_path_for_asr: str | None,
) -> dict[str, float]:
    del source_audio_sr, video_audio_sr
    if np is None:
        return {"audio": 0.0, "wer": 1.0, "tsim": 0.0, "audio_quality": 0.0}

    ref_text = (expected_transcript or "").strip()
    if not ref_text and source_audio_path_for_asr:
        ref_text = _transcribe(source_audio_path_for_asr)
    hyp_text = _transcribe(video_audio_path_for_asr) if video_audio_path_for_asr else ""

    wer = _wer(ref_text, hyp_text) if ref_text else 0.5
    tsim = _speaker_similarity(source_audio_signal, video_audio_signal)
    quality = _audio_quality(video_audio_signal)

    score = 0.40 * normalize_down(wer, 0.0, 1.0) + 0.35 * tsim + 0.25 * quality
    return {
        "audio": clamp01(score),
        "wer": clamp01(wer),
        "tsim": clamp01(tsim),
        "audio_quality": clamp01(quality),
        "video_audio_energy": signal_energy(video_audio_signal),
        "source_audio_energy": signal_energy(source_audio_signal),
    }
