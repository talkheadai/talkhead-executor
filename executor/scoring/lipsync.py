from __future__ import annotations

from typing import Any

from executor.scoring.utils import clamp01, corrcoef, normalize_down, normalize_up

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


def _mouth_motion_signal(frames: list[Any], face_bboxes: list[Any]) -> list[float]:
    if np is None:
        return []
    values: list[float] = []
    prev_mouth: Any | None = None
    for idx, frame in enumerate(frames):
        bbox = face_bboxes[idx] if idx < len(face_bboxes) else None
        if bbox is None:
            values.append(0.0)
            prev_mouth = None
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h = max(1, y2 - y1)
        w = max(1, x2 - x1)
        my1 = y1 + int(0.62 * h)
        my2 = min(y2, y1 + int(0.96 * h))
        mx1 = x1 + int(0.20 * w)
        mx2 = x1 + int(0.80 * w)
        mouth = frame[max(0, my1) : max(0, my2), max(0, mx1) : max(0, mx2)]
        if mouth.size == 0:
            values.append(0.0)
            prev_mouth = None
            continue
        mouth_gray = mouth.mean(axis=2) if mouth.ndim == 3 else mouth
        if prev_mouth is None or prev_mouth.shape != mouth_gray.shape:
            values.append(0.0)
        else:
            diff = float(np.mean(np.abs(mouth_gray.astype(np.float32) - prev_mouth.astype(np.float32))))
            values.append(diff)
        prev_mouth = mouth_gray
    return values


def _audio_envelope(signal: Any, sr: int, target_len: int) -> list[float]:
    if np is None or signal is None or target_len <= 0:
        return [0.0] * max(target_len, 0)
    x = np.asarray(signal, dtype=np.float32)
    if len(x) == 0:
        return [0.0] * target_len
    win = max(64, int(sr * 0.04))
    env = []
    for i in range(0, len(x), win):
        seg = x[i : i + win]
        if len(seg) == 0:
            break
        env.append(float(np.sqrt(np.mean(seg * seg) + 1e-9)))
    if not env:
        return [0.0] * target_len
    old_x = np.linspace(0.0, 1.0, num=len(env), endpoint=False, dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float32)
    out = np.interp(new_x, old_x, np.asarray(env, dtype=np.float32))
    return out.tolist()


def compute_lipsync_score(
    frames: list[Any],
    face_bboxes: list[Any],
    video_audio_signal: Any,
    video_audio_sr: int,
) -> dict[str, float]:
    if np is None or not frames:
        return {"lipsync": 0.0, "sync_c": 0.0, "sync_d": 1.0, "audio_envelope": [], "mouth_signal": [], "sync_curve": []}

    mouth = _mouth_motion_signal(frames, face_bboxes)
    env = _audio_envelope(video_audio_signal, video_audio_sr, len(mouth))
    if len(mouth) < 4:
        return {"lipsync": 0.0, "sync_c": 0.0, "sync_d": 1.0, "audio_envelope": env, "mouth_signal": mouth, "sync_curve": []}

    mouth_arr = np.asarray(mouth, dtype=np.float32)
    env_arr = np.asarray(env, dtype=np.float32)

    best_corr = -1.0
    best_lag = 0
    sync_curve: list[float] = []
    max_lag = min(8, max(1, len(mouth_arr) // 8))
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = mouth_arr[-lag:]
            b = env_arr[: len(a)]
        elif lag > 0:
            a = mouth_arr[:-lag]
            b = env_arr[lag : lag + len(a)]
        else:
            a = mouth_arr
            b = env_arr
        c = corrcoef(a, b)
        sync_curve.append(c)
        if c > best_corr:
            best_corr = c
            best_lag = abs(lag)

    sync_c = clamp01((best_corr + 1.0) / 2.0)
    sync_d = float(best_lag) / max(max_lag, 1)

    score = 0.65 * normalize_up(sync_c, 0.2, 0.85) + 0.35 * normalize_down(sync_d, 0.0, 1.0)
    return {
        "lipsync": clamp01(score),
        "sync_c": sync_c,
        "sync_d": clamp01(sync_d),
        "audio_envelope": env,
        "mouth_signal": mouth,
        "sync_curve": sync_curve,
    }
