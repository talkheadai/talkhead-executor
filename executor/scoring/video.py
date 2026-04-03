from __future__ import annotations

from typing import Any

from executor.scoring.utils import clamp01, temporal_smoothness

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def _blur_score(frame: Any) -> float:
    if cv2 is None or np is None:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    var = float(lap.var())
    return clamp01(var / 500.0)


def _brightness_contrast_score(frame: Any) -> float:
    if np is None:
        return 0.0
    gray = frame.mean(axis=2) if frame.ndim == 3 else frame
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    bright = 1.0 - min(1.0, abs(mean - 127.0) / 127.0)
    contrast = clamp01(std / 64.0)
    return clamp01(0.6 * bright + 0.4 * contrast)


def _face_size_score(face_bboxes: list[Any], frames: list[Any]) -> float:
    if np is None or not frames:
        return 0.0
    scores = []
    for i, bbox in enumerate(face_bboxes[: len(frames)]):
        if bbox is None:
            continue
        h, w = frames[i].shape[:2]
        area = max(1.0, float(w * h))
        bw = max(1.0, float(bbox[2] - bbox[0]))
        bh = max(1.0, float(bbox[3] - bbox[1]))
        ratio = (bw * bh) / area
        # healthy talking-head range around 10%-45%.
        if ratio < 0.10:
            score = clamp01(ratio / 0.10)
        elif ratio > 0.45:
            score = clamp01((0.70 - ratio) / 0.25)
        else:
            score = 1.0
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def _artifact_score(frames: list[Any]) -> float:
    if np is None or not frames:
        return 0.0
    vals = []
    for frame in frames:
        x = frame.astype(np.float32) / 255.0
        over = float(np.mean(x < 0.01) + np.mean(x > 0.99))
        vals.append(1.0 - clamp01(over / 0.35))
    return float(np.mean(vals))


def _motion_naturalness(frames: list[Any]) -> float:
    if np is None or len(frames) < 3:
        return 0.0
    diffs = []
    for i in range(1, len(frames)):
        d = float(np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))))
        diffs.append(d)
    if not diffs:
        return 0.0
    mean_motion = float(np.mean(diffs))
    motion_score = clamp01((mean_motion - 1.0) / 18.0)
    smooth = temporal_smoothness(diffs)
    return clamp01(0.6 * motion_score + 0.4 * smooth)


def compute_video_quality_score(frames: list[Any], face_bboxes: list[Any]) -> dict[str, float]:
    if np is None or not frames:
        return {"video": 0.0, "aesthetic_score": 0.0, "motion_naturalness": 0.0}
    blur = float(np.mean([_blur_score(f) for f in frames]))
    bc = float(np.mean([_brightness_contrast_score(f) for f in frames]))
    face = _face_size_score(face_bboxes, frames)
    art = _artifact_score(frames)
    aesthetic = clamp01(0.35 * blur + 0.25 * bc + 0.25 * face + 0.15 * art)
    motion = _motion_naturalness(frames)
    score = clamp01(0.6 * aesthetic + 0.4 * motion)
    return {
        "video": score,
        "aesthetic_score": aesthetic,
        "motion_naturalness": motion,
    }
