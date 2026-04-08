from __future__ import annotations

import contextlib
from typing import Any

import cv2
from loguru import logger
from executor.scoring.utils import (
    cached_model,
    clamp01,
    cosine_similarity,
    percentile,
    read_image,
)

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

_INSIGHTFACE_INIT_LOGGED = False


def _build_insightface() -> Any:
    import insightface  # type: ignore

    app = insightface.app.FaceAnalysis(name="buffalo_l")
    # Prefer GPU, but gracefully fall back to CPU so identity mode remains robust.
    try:
        app.prepare(ctx_id=0)
    except Exception:
        app.prepare(ctx_id=-1)
    return app


def get_insightface_app() -> Any | None:
    global _INSIGHTFACE_INIT_LOGGED
    with contextlib.suppress(Exception):
        return cached_model("insightface_buffalo_l", _build_insightface)
    if not _INSIGHTFACE_INIT_LOGGED:
        try:
            cached_model("insightface_buffalo_l", _build_insightface)
            logger.info("✅ insightface initialized")
        except Exception as exc:
            logger.warning(
                f"❌ insightface unavailable; using fallback_histogram mode; reason={exc}",
            )
        _INSIGHTFACE_INIT_LOGGED = True
    return None


def _face_embed_from_frame(face_app: Any, img: Any) -> tuple[Any | None, Any | None]:
    faces = face_app.get(img)
    if not faces:
        # Some close-cropped faces fail detection; pad to give detector context.
        h, w = img.shape[:2]
        pad = max(32, int(0.1 * max(h, w)))
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        faces = face_app.get(padded)
    if not faces:
        logger.info("❌ faces is None or empty")
        return None, None
    faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
    face = faces[0]
    emb = getattr(face, "normed_embedding", None)
    bbox = getattr(face, "bbox", None)
    return emb, bbox


def compute_identity_score(reference_image_path: str, sampled_frames: list[Any]) -> dict[str, Any]:
    if np is None or not sampled_frames:
        logger.info("❌ sampled_frames is None or empty")
        return {
            "identity": 0.0,
            "face_detect_ratio": 0.0,
            "per_frame_identity": [],
            "face_bboxes": [],
            "identity_mode": "unavailable",
        }
    face_app = get_insightface_app()
    if face_app is None:
        # Deterministic fallback: compare color histograms against reference.
        ref = read_image(reference_image_path)
        ref_hist = np.histogram(ref.flatten(), bins=64, range=(0, 255), density=True)[0]
        sims: list[float] = []
        for frame in sampled_frames:
            fh = np.histogram(frame.flatten(), bins=64, range=(0, 255), density=True)[0]
            sims.append(clamp01(cosine_similarity(ref_hist, fh)))
        mean_sim = float(np.mean(sims)) if sims else 0.0
        p10 = percentile(sims, 10.0)
        return {
            "identity": clamp01(0.75 * mean_sim + 0.25 * p10),
            "face_detect_ratio": 0.0,
            "per_frame_identity": sims,
            "face_bboxes": [None] * len(sims),
            "identity_mode": "fallback_histogram",
        }

    ref_img = read_image(reference_image_path)
    ref_emb, _ = _face_embed_from_frame(face_app, ref_img)
    if ref_emb is None:
        logger.info("❌ reference image embedding is None")
        return {
            "identity": 0.0,
            "face_detect_ratio": 0.0,
            "per_frame_identity": [],
            "face_bboxes": [],
            "identity_mode": "insightface_no_ref_face",
        }

    sims: list[float] = []
    bboxes: list[Any] = []
    detect_count = 0
    for frame in sampled_frames:
        emb, bbox = _face_embed_from_frame(face_app, frame)
        bboxes.append(bbox)
        if emb is None:
            continue
        detect_count += 1
        sims.append(clamp01(cosine_similarity(ref_emb, emb)))
    mean_sim = float(np.mean(sims)) if sims else 0.0
    p10 = percentile(sims, 10.0)
    face_detect_ratio = detect_count / max(len(sampled_frames), 1)
    score = clamp01(0.75 * mean_sim + 0.25 * p10)
    return {
        "identity": score,
        "mean_similarity": mean_sim,
        "p10_similarity": p10,
        "face_detect_ratio": face_detect_ratio,
        "per_frame_identity": sims,
        "face_bboxes": bboxes,
        "identity_mode": "insightface",
    }
