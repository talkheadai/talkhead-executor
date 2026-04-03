from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from loguru import logger

from executor.models import Challenge, ChallengeResult
from executor.scoring import score_video

READY_TIMEOUT_SEC = 600
CHALLENGE_TIMEOUT_SEC = 120
WARMUP_COUNT = 0
SCORING_COUNT = 1
PULL_IMAGE_MAX_RETRIES = 5
PULL_IMAGE_RETRY_SLEEP_SEC = 2.0


def _safe_remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _infer_audio_extension(audio_bytes: bytes) -> str:
    if audio_bytes.startswith(b"RIFF") and len(audio_bytes) >= 12 and audio_bytes[8:12] == b"WAVE":
        return ".wav"
    if audio_bytes.startswith(b"ID3") or (len(audio_bytes) >= 2 and audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
        return ".mp3"
    if audio_bytes.startswith(b"OggS"):
        return ".ogg"
    if audio_bytes.startswith(b"fLaC"):
        return ".flac"
    return ".bin"


def _load_challenges(challenges_dir: str) -> list[Challenge]:
    root = Path(challenges_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Challenges directory not found: {challenges_dir}")

    challenges: list[Challenge] = []
    for challenge_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        face_path = challenge_dir / "face.png"
        audio_path = challenge_dir / "audio.wav"
        if not face_path.exists() or not audio_path.exists():
            continue
        challenges.append(
            Challenge(
                challenge_id=challenge_dir.name,
                text="",
                face_bytes=face_path.read_bytes(),
                audio_bytes=audio_path.read_bytes(),
            )
        )
    return challenges


def pull_image(image_ref: str) -> bool:
    for attempt in range(1, PULL_IMAGE_MAX_RETRIES + 1):
        try:
            subprocess.run(["docker", "pull", image_ref], check=True)
            if attempt > 1:
                logger.info(f"docker pull succeeded on retry {attempt}/{PULL_IMAGE_MAX_RETRIES}: {image_ref}")
            return True
        except Exception as exc:
            logger.warning(
                f"docker pull failed attempt {attempt}/{PULL_IMAGE_MAX_RETRIES} "
                f"for image={image_ref}: {exc}"
            )
            if attempt < PULL_IMAGE_MAX_RETRIES:
                time.sleep(PULL_IMAGE_RETRY_SLEEP_SEC)
    logger.error(f"docker pull failed after {PULL_IMAGE_MAX_RETRIES} attempts: {image_ref}")
    return False

def start_container(image_ref: str, job_dir: str) -> str:
    job_path = Path(job_dir)
    input_dir = job_path / "input"
    output_dir = job_path / "output"

    cmd = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--gpus",
        "all",
        "--network=none",
        "--cpus=8",
        "--memory=16g",
        "--pids-limit=256",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=1g",
        "--tmpfs",
        "/var/tmp:rw,nosuid,nodev,noexec,size=256m",
        "-e",
        "TMPDIR=/tmp",
        "-e",
        "HF_HOME=/tmp/hf",
        "-e",
        "TRANSFORMERS_CACHE=/tmp/hf/transformers",
        "-e",
        "XDG_CACHE_HOME=/tmp/.cache",
        "-v",
        f"{input_dir}:/input:rw",
        "-v",
        f"{output_dir}:/output:rw",
        image_ref,
    ]

    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    container_id = res.stdout.strip()
    if not container_id:
        raise RuntimeError("docker run did not return a container id")
    return container_id


def wait_for_ready(job_dir: str, timeout_sec: int) -> None:
    ready_path = Path(job_dir) / "output" / "ready.txt"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if ready_path.exists():
            return
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for ready file: {ready_path}")


def stop_container(container_id: str) -> None:
    subprocess.run(["docker", "kill", container_id], check=False)


def _send_challenge(job_dir: Path, challenge: Challenge) -> ChallengeResult:
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"

    task_path = input_dir / "task.json"
    face_dst = input_dir / "face.png"
    audio_dst = input_dir / "audio.wav"
    result_json_path = output_dir / f"{challenge.challenge_id}.json"
    result_mp4_path = output_dir / f"{challenge.challenge_id}.mp4"

    _safe_remove(result_json_path)
    _safe_remove(result_mp4_path)
    _safe_remove(task_path)
    _safe_remove(face_dst)
    _safe_remove(audio_dst)

    face_dst.write_bytes(challenge.face_bytes)
    audio_dst.write_bytes(challenge.audio_bytes)

    task_payload = {
        "challenge_id": challenge.challenge_id,
        "text": challenge.text,
        "seed": 123,
        "fps": 25,
        "resolution": 512,
        "max_seconds": 5,
    }
    task_path.write_text(json.dumps(task_payload), encoding="utf-8")

    start = time.perf_counter()
    deadline = time.time() + CHALLENGE_TIMEOUT_SEC

    while time.time() < deadline:
        if result_json_path.exists():
            elapsed = time.perf_counter() - start
            try:
                result_data = json.loads(result_json_path.read_text(encoding="utf-8"))
            except Exception:
                result_data = {}
            success = bool(result_data.get("success", False))
            error = result_data.get("error")
            _safe_remove(task_path)
            _safe_remove(face_dst)
            _safe_remove(audio_dst)
            return ChallengeResult(
                challenge_id=challenge.challenge_id,
                success=success,
                host_time_sec=elapsed,
                error=error,
            )
        time.sleep(0.1)

    elapsed = time.perf_counter() - start
    _safe_remove(task_path)
    _safe_remove(face_dst)
    _safe_remove(audio_dst)
    return ChallengeResult(
        challenge_id=challenge.challenge_id,
        success=False,
        host_time_sec=elapsed,
        error=f"timeout_{CHALLENGE_TIMEOUT_SEC}s",
    )


def build_image_ref(image_ref: str) -> str:
    if "@" in image_ref:
        return image_ref
    return f"aerast/musetalk@{image_ref}"


def evaluate(image_ref: str, challenges: list[Challenge]) -> float:
    needed = WARMUP_COUNT + SCORING_COUNT
    if len(challenges) < needed:
        raise ValueError(f"Need at least {needed} challenges, found {len(challenges)}")

    warmup_challenges = challenges[:WARMUP_COUNT]
    scoring_challenges = challenges[WARMUP_COUNT:needed]

    job_dir = Path(tempfile.mkdtemp(prefix="sn108_job_"))
    (job_dir / "input").mkdir(parents=True, exist_ok=True)
    (job_dir / "output").mkdir(parents=True, exist_ok=True)

    container_id: str | None = None
    warmup_results: list[ChallengeResult] = []
    scoring_results: list[ChallengeResult] = []
    scoring_quality: list[float] = []
    try:
        logger.info(f"pulling image: {image_ref}")
        if not pull_image(image_ref):
            return 0.0
        container_id = start_container(image_ref, str(job_dir))
        wait_for_ready(str(job_dir), READY_TIMEOUT_SEC)
        for challenge in warmup_challenges:
            warmup_results.append(_send_challenge(job_dir, challenge))
        for challenge in scoring_challenges:
            result = _send_challenge(job_dir, challenge)
            scoring_results.append(result)
            if result.success:
                output_video = job_dir / "output" / f"{challenge.challenge_id}.mp4"
                if output_video.exists():
                    face_tmp = job_dir / "input" / f"face_{challenge.challenge_id}.png"
                    audio_ext = _infer_audio_extension(challenge.audio_bytes)
                    audio_tmp = job_dir / "input" / f"audio_{challenge.challenge_id}{audio_ext}"
                    face_tmp.write_bytes(challenge.face_bytes)
                    audio_tmp.write_bytes(challenge.audio_bytes)
                    try:
                        score_obj = score_video(
                            video_path=str(output_video),
                            reference_image_path=str(face_tmp),
                            audio_path=str(audio_tmp),
                            expected_transcript=challenge.text or None,
                        )
                        scoring_quality.append(float(score_obj.get("final", 0.0)))
                    except Exception as exc:
                        logger.warning(
                            f"quality scoring failed challenge={challenge.challenge_id} "
                            f"image_ref={image_ref} err={exc}"
                        )
                        scoring_quality.append(0.0)
                    finally:
                        pass
                        _safe_remove(face_tmp)
                        _safe_remove(audio_tmp)
    finally:
        if container_id:
            stop_container(container_id)
        shutil.rmtree(job_dir, ignore_errors=True)

    failures = [r for r in warmup_results + scoring_results if not r.success]
    if failures:
        errors = ", ".join(f"{r.challenge_id}:{r.error}" for r in failures)
        raise RuntimeError(f"challenge failures: {errors}")

    if not scoring_quality:
        return 0.0
    return _avg(scoring_quality)


def load_challenges_from_dir(challenges_dir: str) -> list[Challenge]:
    return _load_challenges(challenges_dir)
