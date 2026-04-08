from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class Challenge:
    challenge_id: str
    text: str
    audio_bytes: bytes
    face_bytes: bytes


@dataclass
class ChallengeResult:
    challenge_id: str
    success: bool
    host_time_sec: float
    error: str | None = None
    peak_vram_gb: float | None = None
    inference_time_sec: float | None = None
    efficiency_source: str | None = None


@dataclass
class MinerRecord:
    hotkey: str
    image_ref: str
    submit_time: float
    metrics: dict[str, Any] | None = field(default=None)
    coming_metrics: dict[str, Any] | None = field(default=None)


class MinerSubmission(BaseModel):
    hotkey: str
    image_ref: str
    submit_time: float


class MinerMetricsResponse(BaseModel):
    hotkey: str
    image_ref: str
    submit_time: float
    metrics: dict[str, Any] | None = None
