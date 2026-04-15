"""
Efficiency scoring: soft modifier on quality from inference time and peak VRAM.

VRAM and wall-clock inference time are measured **inside the miner container**
during the steady-state inference phase only. The executor reads them from the
miner `result.json` under `efficiency.peak_vram_gb` and
`efficiency.inference_time_sec` (see `executor.evaluation.docker_runner`).

**Limitation:** The executor process cannot call `torch.cuda.max_memory_allocated()`
for an isolated GPU workload in another container. If the miner omits these
fields, metrics are treated as unavailable: norms are 0 and `efficiency_factor`
is 1.0 so quality-only scoring remains backward compatible.
"""

from __future__ import annotations

MAX_VRAM_GB = 10
MAX_INFERENCE_TIME_SEC = 30
CAP_PENALTY_SCORE = 0.01
TIME_REF_SEC = 30
VRAM_REF_GB = 8
TIE_QUALITY_EPSILON = 1e-5

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EfficiencyConfig:
    """Configurable caps and normalization scales for efficiency scoring."""

    max_vram_gb: float | None
    max_inference_time_sec: float | None
    cap_mode: str | None
    # When cap_mode is penalty, multiply the combined score by this (severe soft penalty).
    cap_penalty_score: float
    time_ref_sec: float
    vram_ref_gb: float
    # For tie-break: treat scores as equal if within this absolute difference.
    tie_quality_epsilon: float

    @staticmethod
    def from_env() -> EfficiencyConfig:
        penalty = CAP_PENALTY_SCORE
        penalty = max(0.0, min(1.0, penalty))

        return EfficiencyConfig(
            max_vram_gb=MAX_VRAM_GB,
            max_inference_time_sec=MAX_INFERENCE_TIME_SEC,
            cap_mode="penalty",
            cap_penalty_score=penalty,
            time_ref_sec=TIME_REF_SEC,
            vram_ref_gb=VRAM_REF_GB,
            tie_quality_epsilon=TIE_QUALITY_EPSILON,
        )


def norm_higher_worse_linear(value: float, ref: float) -> float:
    """
    Normalize so larger values map to larger norms (worse), clamped safely.
    Uses value/ref so ref acts as the "typical" scale; result clamped to [0, 3].
    """
    if ref <= 0 or not math.isfinite(value) or not math.isfinite(ref):
        return 0.0
    v = max(0.0, float(value)) / float(ref)
    return max(0.0, min(3.0, v))


def efficiency_factor_from_norms(time_norm: float, vram_norm: float) -> float:
    """Soft efficiency modifier: neither time nor VRAM dominates alone."""
    t = max(0.0, min(3.0, float(time_norm)))
    v = max(0.0, min(3.0, float(vram_norm)))
    raw = math.exp(-0.15 * t - 0.10 * v)
    return max(0.0, min(1.0, raw))


def check_hard_caps(
    *,
    worst_vram_gb: float | None,
    worst_time_sec: float | None,
    cfg: EfficiencyConfig,
) -> tuple[bool, str | None]:
    """
    Returns (violates_reject, reason). Penalty mode does not set violates_reject.
    """
    if cfg.cap_mode != "reject":
        return False, None
    if cfg.max_vram_gb is not None and worst_vram_gb is not None:
        if worst_vram_gb > cfg.max_vram_gb:
            return True, f"peak_vram_gb>{cfg.max_vram_gb}"
    if cfg.max_inference_time_sec is not None and worst_time_sec is not None:
        if worst_time_sec > cfg.max_inference_time_sec:
            return True, f"inference_time_sec>{cfg.max_inference_time_sec}"
    return False, None


def cap_penalty_multiplier(
    *,
    violates_vram: bool,
    violates_time: bool,
    cfg: EfficiencyConfig,
) -> float:
    """If cap_mode is penalty and any cap violated, multiply final score by this (once)."""
    if cfg.cap_mode != "penalty":
        return 1.0
    if not violates_vram and not violates_time:
        return 1.0
    return max(0.0, min(1.0, cfg.cap_penalty_score))


def aggregate_efficiency_scores(
    *,
    per_challenge_peak_vram_gb: list[float | None],
    per_challenge_inference_sec: list[float | None],
    mean_quality: float,
    cfg: EfficiencyConfig,
) -> dict:
    """
    Build the structured efficiency block using mean VRAM/time for the modifier
    and worst-case (max) for caps and debug.

    Missing per-challenge values are omitted from means; if all missing,
    """
    vrams = [
        v
        for v in per_challenge_peak_vram_gb
        if v is not None and math.isfinite(v) and v >= 0
    ]
    times = [
        t
        for t in per_challenge_inference_sec
        if t is not None and math.isfinite(t) and t >= 0
    ]

    if not vrams and not times:
        return {
            "peak_vram_gb": None,
            "inference_time_sec": None,
            "peak_vram_gb_worst": None,
            "inference_time_sec_worst": None,
            "time_norm": 0.0,
            "vram_norm": 0.0,
            "efficiency_factor": 1.0,
            "cap_violation": False,
            "quality_score": mean_quality,
            "final_score": mean_quality,
        }

    mean_vram = sum(vrams) / len(vrams) if vrams else None
    mean_time = sum(times) / len(times) if times else None
    worst_vram = max(vrams) if vrams else None
    worst_time = max(times) if times else None

    time_norm = (
        norm_higher_worse_linear(mean_time or 0.0, cfg.time_ref_sec)
        if mean_time is not None
        else 0.0
    )
    vram_norm = (
        norm_higher_worse_linear(mean_vram or 0.0, cfg.vram_ref_gb)
        if mean_vram is not None
        else 0.0
    )

    eff_factor = efficiency_factor_from_norms(time_norm, vram_norm)

    violates_vram = bool(
        cfg.max_vram_gb is not None
        and worst_vram is not None
        and worst_vram > cfg.max_vram_gb
    )
    violates_time = bool(
        cfg.max_inference_time_sec is not None
        and worst_time is not None
        and worst_time > cfg.max_inference_time_sec
    )

    reject, reject_reason = check_hard_caps(
        worst_vram_gb=worst_vram, worst_time_sec=worst_time, cfg=cfg
    )
    if reject:
        return {
            "peak_vram_gb": mean_vram,
            "inference_time_sec": mean_time,
            "peak_vram_gb_worst": worst_vram,
            "inference_time_sec_worst": worst_time,
            "time_norm": time_norm,
            "vram_norm": vram_norm,
            "efficiency_factor": 0.0,
            "cap_violation": True,
            "reject_reason": reject_reason,
            "quality_score": mean_quality,
            "final_score": 0.0,
        }

    pen_mult = cap_penalty_multiplier(
        violates_vram=violates_vram,
        violates_time=violates_time,
        cfg=cfg,
    )
    combined_eff = eff_factor * pen_mult
    final_score = max(0.0, min(1.0, mean_quality * combined_eff))

    out = {
        "peak_vram_gb": mean_vram,
        "inference_time_sec": mean_time,
        "peak_vram_gb_worst": worst_vram,
        "inference_time_sec_worst": worst_time,
        "time_norm": time_norm,
        "vram_norm": vram_norm,
        "efficiency_factor": combined_eff,
        "cap_violation": violates_vram or violates_time,
        "quality_score": mean_quality,
        "final_score": final_score,
    }
    return out


def tie_break_tuple(
    mean_inference_sec: float | None, mean_peak_vram_gb: float | None
) -> tuple[float, float]:
    """
    Sort key fragment: lower inference time wins, then lower VRAM.
    Use with: (-score, tie_break_tuple(...), submit_time) for stable ordering.
    Missing metrics sort after finite ones (worse tie-break).
    """
    t = mean_inference_sec
    v = mean_peak_vram_gb
    t_key = t if t is not None and math.isfinite(t) and t >= 0 else float("inf")
    v_key = v if v is not None and math.isfinite(v) and v >= 0 else float("inf")
    return (t_key, v_key)


def should_prefer_candidate_a(
    *,
    score_a: float,
    score_b: float,
    infer_sec_a: float | None,
    infer_sec_b: float | None,
    peak_vram_a: float | None,
    peak_vram_b: float | None,
    cfg: EfficiencyConfig,
) -> bool:
    """
    Optional ranking helper:
      1) higher score wins
      2) if score difference <= tie_quality_epsilon, lower inference time wins
      3) if still tied, lower peak VRAM wins
    """
    if score_a > score_b + cfg.tie_quality_epsilon:
        return True
    if score_b > score_a + cfg.tie_quality_epsilon:
        return False
    return tie_break_tuple(infer_sec_a, peak_vram_a) < tie_break_tuple(
        infer_sec_b, peak_vram_b
    )
