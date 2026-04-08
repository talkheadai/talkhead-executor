from __future__ import annotations

import contextlib
import math
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


MODEL_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class VideoMeta:
    fps: float
    total_frames: int
    duration_sec: float


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def normalize_up(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp01((value - low) / (high - low))


def normalize_down(value: float, low: float, high: float) -> float:
    return 1.0 - normalize_up(value, low, high)


def get_torch_device() -> str:
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def cached_model(name: str, builder: Any) -> Any:
    if name in MODEL_CACHE:
        return MODEL_CACHE[name]
    model = builder()
    MODEL_CACHE[name] = model
    return model


def cosine_similarity(a: Any, b: Any) -> float:
    if np is None:
        return 0.0
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(av, bv) / denom)


def read_video_frames(video_path: str, *, max_frames: int = 64) -> tuple[list[Any], VideoMeta]:
    if cv2 is None:
        raise RuntimeError("opencv-python is not available")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 25.0)
    total = int(safe_float(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0))
    if total <= 0:
        total = max_frames
    duration = total / fps if fps > 0 else 0.0
    indices = sample_indices(total, max_frames)
    frames: list[Any] = []
    cursor = 0
    target = indices[cursor] if indices else None
    i = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if target is not None and i == target:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            cursor += 1
            if cursor >= len(indices):
                break
            target = indices[cursor]
        i += 1
    cap.release()
    return frames, VideoMeta(fps=fps if fps > 0 else 25.0, total_frames=total, duration_sec=duration)


def sample_indices(total: int, max_frames: int) -> list[int]:
    if total <= 0 or max_frames <= 0:
        return []
    if total <= max_frames:
        return list(range(total))
    step = total / float(max_frames)
    return [min(total - 1, int(i * step)) for i in range(max_frames)]


def read_image(path: str) -> Any:
    if cv2 is None:
        raise RuntimeError("opencv-python is not available")
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image at {path}")
    # InsightFace expects 3-channel BGR input; some PNGs are BGRA or grayscale.
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def extract_audio_from_video(video_path: str) -> str | None:
    out_fd, out_path = tempfile.mkstemp(prefix="score_audio_", suffix=".wav")
    os.close(out_fd)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        out_path,
    ]
    with contextlib.suppress(Exception):
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and Path(out_path).exists():
            return out_path
    with contextlib.suppress(Exception):
        Path(out_path).unlink(missing_ok=True)
    return None


def load_wav_mono_16k(path: str) -> tuple[Any, int]:
    def _silence() -> tuple[Any, int]:
        if np is None:
            return [], 16000
        return np.zeros(16000, dtype=np.float32), 16000

    if np is None:
        return _silence()

    def _resample_to_16k(audio: Any, sr: int) -> Any:
        if sr == 16000:
            return audio
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False, dtype=np.float32)
        tgt_len = max(1, int(len(audio) * 16000 / max(sr, 1)))
        x_new = np.linspace(0.0, 1.0, num=tgt_len, endpoint=False, dtype=np.float32)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    def _decode_riff_wave(raw_bytes: bytes) -> tuple[Any, int]:
        if len(raw_bytes) < 12:
            raise ValueError("wav too small")
        riff = raw_bytes[0:4]
        wave_tag = raw_bytes[8:12]
        if riff != b"RIFF" or wave_tag != b"WAVE":
            raise ValueError("not RIFF/WAVE")

        pos = 12
        fmt_chunk: bytes | None = None
        data_chunk: bytes | None = None
        while pos + 8 <= len(raw_bytes):
            chunk_id = raw_bytes[pos : pos + 4]
            chunk_sz = struct.unpack("<I", raw_bytes[pos + 4 : pos + 8])[0]
            pos += 8
            if pos + chunk_sz > len(raw_bytes):
                break
            payload = raw_bytes[pos : pos + chunk_sz]
            pos += chunk_sz + (chunk_sz % 2)
            if chunk_id == b"fmt ":
                fmt_chunk = payload
            elif chunk_id == b"data":
                data_chunk = payload
            if fmt_chunk is not None and data_chunk is not None:
                break

        if fmt_chunk is None or data_chunk is None or len(fmt_chunk) < 16:
            raise ValueError("missing fmt/data chunk")

        audio_fmt, channels, sr, _byte_rate, _block_align, bits = struct.unpack("<HHIIHH", fmt_chunk[:16])
        if channels <= 0:
            raise ValueError("invalid channel count")

        # WAVE_FORMAT_EXTENSIBLE: subtype stored in extra bytes (GUID starts with subtype code LE).
        if audio_fmt == 0xFFFE and len(fmt_chunk) >= 40:
            cb_size = struct.unpack("<H", fmt_chunk[16:18])[0]
            if cb_size >= 22:
                subformat_code = struct.unpack("<H", fmt_chunk[24:26])[0]
                audio_fmt = subformat_code

        if audio_fmt == 1:  # PCM int
            if bits == 8:
                audio = (np.frombuffer(data_chunk, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif bits == 16:
                audio = np.frombuffer(data_chunk, dtype="<i2").astype(np.float32) / 32768.0
            elif bits == 24:
                b = np.frombuffer(data_chunk, dtype=np.uint8)
                usable = (len(b) // 3) * 3
                b = b[:usable].reshape(-1, 3)
                i32 = (
                    b[:, 0].astype(np.int32)
                    | (b[:, 1].astype(np.int32) << 8)
                    | (b[:, 2].astype(np.int32) << 16)
                )
                sign = i32 & 0x800000
                i32 = i32 - (sign << 1)
                audio = i32.astype(np.float32) / 8388608.0
            elif bits == 32:
                audio = np.frombuffer(data_chunk, dtype="<i4").astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"unsupported PCM bit depth {bits}")
        elif audio_fmt == 3:  # IEEE float
            if bits == 32:
                audio = np.frombuffer(data_chunk, dtype="<f4").astype(np.float32)
            elif bits == 64:
                audio = np.frombuffer(data_chunk, dtype="<f8").astype(np.float32)
            else:
                raise ValueError(f"unsupported float bit depth {bits}")
        else:
            raise ValueError(f"unsupported wav format code {audio_fmt}")

        if channels > 1:
            usable = (len(audio) // channels) * channels
            audio = audio[:usable].reshape(-1, channels).mean(axis=1).astype(np.float32)
        return audio, int(sr)

    decoder_errors: list[str] = []
    file_size = -1
    header_hex = ""
    with contextlib.suppress(Exception):
        p = Path(path)
        file_size = p.stat().st_size
        header_hex = p.read_bytes()[:16].hex()

    # 1) Most robust path across wav containers/codecs.
    try:
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            path,
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode == 0 and proc.stdout:
            raw = np.frombuffer(proc.stdout, dtype=np.float32)
            if raw.size > 0:
                return raw, 16000
        stderr = (proc.stderr.decode("utf-8", errors="ignore") if proc.stderr else "").strip()
        decoder_errors.append(f"ffmpeg rc={proc.returncode} stderr={stderr[:160] or 'empty'}")
    except FileNotFoundError:
        decoder_errors.append("ffmpeg not found")
    except Exception as exc:
        decoder_errors.append(f"ffmpeg exception={exc}")

    # 2) Torchaudio fallback.
    if torch is not None:
        try:
            import torchaudio  # type: ignore

            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav_np = wav.squeeze(0).cpu().numpy().astype(np.float32)
            return _resample_to_16k(wav_np, int(sr)), 16000
        except Exception as exc:
            decoder_errors.append(f"torchaudio exception={exc}")
    else:
        decoder_errors.append("torch unavailable")

    # 3) Pure stdlib wave fallback for common PCM wav flavors.
    try:
        import wave

        with wave.open(path, "rb") as wf:
            channels = wf.getnchannels()
            sr = int(wf.getframerate())
            frames = wf.getnframes()
            width = wf.getsampwidth()
            data = wf.readframes(frames)

        if width == 1:
            audio = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 2:
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 3:
            b = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
            i32 = (
                b[:, 0].astype(np.int32)
                | (b[:, 1].astype(np.int32) << 8)
                | (b[:, 2].astype(np.int32) << 16)
            )
            sign = i32 & 0x800000
            i32 = i32 - (sign << 1)
            audio = i32.astype(np.float32) / 8388608.0
        elif width == 4:
            audio = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported wav sample width: {width}")

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return _resample_to_16k(audio.astype(np.float32), sr), 16000
    except Exception as exc:
        decoder_errors.append(f"wave exception={exc}")

    # 4) Manual RIFF/WAVE parser for WAV variants unsupported by `wave` in py3.10
    #    (e.g. WAVE_FORMAT_EXTENSIBLE or float PCM depending on headers).
    try:
        raw_bytes = Path(path).read_bytes()
        audio, sr = _decode_riff_wave(raw_bytes)
        return _resample_to_16k(audio.astype(np.float32), sr), 16000
    except Exception as exc:
        decoder_errors.append(f"riff-parser exception={exc}")

    format_hint = "unknown"
    if header_hex.startswith("494433"):  # ID3
        format_hint = "looks_like_mp3_with_id3"
    elif header_hex.startswith("52494646"):  # RIFF
        format_hint = "looks_like_riff"
    elif header_hex.startswith("4f676753"):  # OggS
        format_hint = "looks_like_ogg"
    elif header_hex.startswith("664c6143"):  # fLaC
        format_hint = "looks_like_flac"

    logger.warning(
        f"failed to decode audio path={path} size={file_size} header={header_hex or 'n/a'} "
        f"format_hint={format_hint}; using silence fallback; decode_errors={' | '.join(decoder_errors)}"
    )
    return _silence()


def signal_energy(signal: Any) -> float:
    if np is None:
        return 0.0
    if signal is None or len(signal) == 0:
        return 0.0
    x = np.asarray(signal, dtype=np.float32)
    return float(np.mean(np.square(x)))


def safe_run(metric_name: str, fn: Any, default: dict[str, float]) -> dict[str, float]:
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"metric {metric_name} failed: {exc}")
        return default


def percentile(values: list[float], q: float) -> float:
    if np is None or not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def temporal_smoothness(signal: Any) -> float:
    if np is None or signal is None or len(signal) < 3:
        return 0.0
    x = np.asarray(signal, dtype=np.float32)
    diff = np.diff(x)
    jerk = np.diff(diff)
    score = 1.0 / (1.0 + float(np.mean(np.abs(jerk))))
    return clamp01(score)


def corrcoef(a: Any, b: Any) -> float:
    if np is None:
        return 0.0
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    n = min(len(av), len(bv))
    if n < 3:
        return 0.0
    av = av[:n]
    bv = bv[:n]
    av = av - av.mean()
    bv = bv - bv.mean()
    den = float(math.sqrt(float((av * av).sum()) * float((bv * bv).sum())))
    if den <= 1e-12:
        return 0.0
    return float((av * bv).sum() / den)
