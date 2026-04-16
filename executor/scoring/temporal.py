from __future__ import annotations

from typing import Any

from executor.scoring.utils import clamp01, cosine_similarity, temporal_smoothness

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


def _identity_temporal_consistency(identity_values: list[float]) -> float:
    if np is None or len(identity_values) < 3:
        return 0.0
    arr = np.asarray(identity_values, dtype=np.float32)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return clamp01(0.7 * mean + 0.3 * (1.0 - clamp01(std / 0.25)))


def _bbox_center_jitter(face_bboxes: list[Any]) -> float:
    if np is None:
        return 0.0
    centers = []
    for bbox in face_bboxes:
        if bbox is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        centers.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
    if len(centers) < 3:
        return 0.0
    c = np.asarray(centers, dtype=np.float32)
    dx = np.diff(c[:, 0])
    dy = np.diff(c[:, 1])
    speed = np.sqrt(dx * dx + dy * dy)
    return temporal_smoothness(speed)


def _mouth_continuity(mouth_motion_signal: list[float]) -> float:
    return temporal_smoothness(mouth_motion_signal)


def compute_temporal_score(
    per_frame_identity: list[float],
    face_bboxes: list[Any],
    mouth_motion_signal: list[float],
) -> dict[str, float]:
    id_consistency = _identity_temporal_consistency(per_frame_identity)
    motion_smooth = clamp01(0.6 * _bbox_center_jitter(face_bboxes) + 0.4 * _mouth_continuity(mouth_motion_signal))
    temporal = clamp01(0.5 * id_consistency + 0.5 * motion_smooth)
    return {
        "temporal": temporal,
        "identity_temporal_consistency": id_consistency,
        "motion_smoothness": motion_smooth,
        "embedding_similarity_proxy": cosine_similarity(
            np.asarray(per_frame_identity[:32], dtype=np.float32) if np is not None else [],
            np.asarray(per_frame_identity[-32:], dtype=np.float32) if np is not None else [],
        ),
    }
