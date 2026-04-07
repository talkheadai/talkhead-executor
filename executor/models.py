from __future__ import annotations

from dataclasses import dataclass

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
    score: float = -1.0
    coming_score: float = -1.0


class MinerSubmission(BaseModel):
    hotkey: str
    image_ref: str
    submit_time: float


class MinerScoreResponse(BaseModel):
    hotkey: str
    image_ref: str
    submit_time: float
    score: float
    coming_score: float
