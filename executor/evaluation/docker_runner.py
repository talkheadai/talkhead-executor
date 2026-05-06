from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from loguru import logger

from executor.models import Challenge, ChallengeResult
from executor.r2 import upload_video_base64_to_r2
from executor.scoring.efficiency import EfficiencyConfig, aggregate_efficiency_scores
from executor.scoring import score_video

READY_TIMEOUT_SEC = 600
CHALLENGE_TIMEOUT_SEC = 120
WARMUP_COUNT = 2
SCORING_COUNT = 5
PULL_IMAGE_MAX_RETRIES = 5
PULL_IMAGE_RETRY_SLEEP_SEC = 2.0
PULL_IMAGE_TIMEOUT_SEC = 900
EVALUATION_CONTAINER_LABEL = "talkhead.executor.evaluation=true"

_ACTIVE_CONTAINERS: set[str] = set()
_ACTIVE_CONTAINERS_LOCK = threading.Lock()


def _safe_remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _infer_audio_extension(audio_bytes: bytes) -> str:
    if (
        audio_bytes.startswith(b"RIFF")
        and len(audio_bytes) >= 12
        and audio_bytes[8:12] == b"WAVE"
    ):
        return ".wav"
    if audio_bytes.startswith(b"ID3") or (
        len(audio_bytes) >= 2
        and audio_bytes[0] == 0xFF
        and (audio_bytes[1] & 0xE0) == 0xE0
    ):
        return ".mp3"
    if audio_bytes.startswith(b"OggS"):
        return ".ogg"
    if audio_bytes.startswith(b"fLaC"):
        return ".flac"
    return ".bin"


def _safe_float_or_none(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out < 0:
        return None
    return out


def _upload_video_to_r2(video_path: Path, challenge: Challenge) -> str | None:
    try:
        video_b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
        return upload_video_base64_to_r2(
            video_b64,
            prompt=challenge.text or challenge.challenge_id,
        )
    except Exception as exc:
        logger.warning(
            f"r2 upload failed challenge={challenge.challenge_id} "
            f"video_path={video_path} err={exc}"
        )
        return None


def _container_host_pids(container_id: str) -> set[int]:
    """
    Return host PIDs for processes inside this container.
    Best effort: empty set if lookup fails.
    """
    try:
        res = subprocess.run(
            ["docker", "top", container_id, "-eo", "pid"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            return set()
        pids: set[int] = set()
        for line in res.stdout.splitlines():
            s = line.strip()
            if not s or s.lower() == "pid":
                continue
            try:
                pids.add(int(s))
            except ValueError:
                continue
        return pids
    except Exception:
        return set()


def _gpu_memory_by_pid_mib() -> dict[int, float]:
    """
    Query active CUDA process memory via nvidia-smi.
    Returns {host_pid: used_gpu_memory_mib}.
    """
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            return {}
        out: dict[int, float] = {}
        for line in res.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                mib = float(parts[1])
            except ValueError:
                continue
            if pid >= 0 and mib >= 0:
                out[pid] = mib
        return out
    except Exception:
        return {}


class _ContainerVramPeakMonitor:
    """
    Best-effort host-side VRAM monitor for one container.
    Samples active CUDA memory and tracks max total MiB across container PIDs.
    """

    def __init__(self, container_id: str, interval_sec: float) -> None:
        self.container_id = container_id
        self.interval_sec = max(0.05, interval_sec)
        self.peak_mib = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="vram-peak-monitor"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            container_pids = _container_host_pids(self.container_id)
            if container_pids:
                gpu_by_pid = _gpu_memory_by_pid_mib()
                total_mib = sum(gpu_by_pid.get(pid, 0.0) for pid in container_pids)
                if total_mib > self.peak_mib:
                    self.peak_mib = total_mib
            self._stop.wait(self.interval_sec)


def _extract_efficiency_metrics(
    *,
    elapsed_host_time_sec: float,
    executor_peak_vram_gb: float | None,
) -> tuple[float | None, float | None]:
    """
    Extract efficiency metrics from the executor.
    """
    if executor_peak_vram_gb is not None:
        return max(0.0, executor_peak_vram_gb), max(0.0, elapsed_host_time_sec)
    return None, max(0.0, elapsed_host_time_sec)


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


def pull_image(image_ref: str, timeout_sec: int = PULL_IMAGE_TIMEOUT_SEC) -> bool:
    for attempt in range(1, PULL_IMAGE_MAX_RETRIES + 1):
        try:
            subprocess.run(
                ["docker", "pull", image_ref], check=True, timeout=timeout_sec
            )
            if attempt > 1:
                logger.info(
                    f"docker pull succeeded on retry {attempt}/{PULL_IMAGE_MAX_RETRIES}: {image_ref}"
                )
            return True
        except Exception as exc:
            logger.warning(
                f"docker pull failed attempt {attempt}/{PULL_IMAGE_MAX_RETRIES} "
                f"for image={image_ref}: {exc}"
            )
            if attempt < PULL_IMAGE_MAX_RETRIES:
                time.sleep(PULL_IMAGE_RETRY_SLEEP_SEC)
    logger.error(
        f"docker pull failed after {PULL_IMAGE_MAX_RETRIES} attempts: {image_ref}"
    )
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
        "--label",
        EVALUATION_CONTAINER_LABEL,
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
    with _ACTIVE_CONTAINERS_LOCK:
        _ACTIVE_CONTAINERS.discard(container_id)


def stop_all_running_containers() -> None:
    container_ids: list[str] = []
    try:
        res = subprocess.run(
            [
                "docker",
                "ps",
                "-q",
                "--filter",
                f"label={EVALUATION_CONTAINER_LABEL}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if res.stdout.strip():
            container_ids.extend(
                [line.strip() for line in res.stdout.splitlines() if line.strip()]
            )
    except Exception as exc:
        logger.warning(f"failed to list running evaluation containers: {exc}")

    with _ACTIVE_CONTAINERS_LOCK:
        container_ids.extend(_ACTIVE_CONTAINERS)

    deduped_ids = list(dict.fromkeys(container_ids))
    for cid in deduped_ids:
        stop_container(cid)


def remove_all_images() -> None:
    try:
        subprocess.run(["docker", "image", "prune", "-a", "-f"], check=False)
    except Exception as exc:
        logger.warning(f"failed to prune docker images: {exc}")


def _send_challenge(
    job_dir: Path,
    challenge: Challenge,
    *,
    container_id: str | None = None,
    measure_executor_vram_peak: bool = False,
) -> ChallengeResult:
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
    vram_monitor: _ContainerVramPeakMonitor | None = None
    if measure_executor_vram_peak and container_id:
        vram_monitor = _ContainerVramPeakMonitor(
            container_id=container_id,
            interval_sec=0.1,
        )
        vram_monitor.start()

    try:
        while time.time() < deadline:
            if result_json_path.exists():
                elapsed = time.perf_counter() - start
                try:
                    result_data = json.loads(
                        result_json_path.read_text(encoding="utf-8")
                    )
                except Exception:
                    result_data = {}
                success = bool(result_data.get("success", False))
                error = result_data.get("error")
                peak_vram_gb, inference_time_sec = _extract_efficiency_metrics(
                    elapsed_host_time_sec=elapsed,
                    executor_peak_vram_gb=(
                        (vram_monitor.peak_mib / 1024.0)
                        if vram_monitor and vram_monitor.peak_mib > 0
                        else None
                    ),
                )
                _safe_remove(task_path)
                _safe_remove(face_dst)
                _safe_remove(audio_dst)
                return ChallengeResult(
                    challenge_id=challenge.challenge_id,
                    success=success,
                    host_time_sec=elapsed,
                    error=error,
                    peak_vram_gb=peak_vram_gb,
                    inference_time_sec=inference_time_sec,
                )
            time.sleep(0.1)
    finally:
        if vram_monitor:
            vram_monitor.stop()

    elapsed = time.perf_counter() - start
    _safe_remove(task_path)
    _safe_remove(face_dst)
    _safe_remove(audio_dst)
    return ChallengeResult(
        challenge_id=challenge.challenge_id,
        success=False,
        host_time_sec=elapsed,
        error=f"timeout_{CHALLENGE_TIMEOUT_SEC}s",
        peak_vram_gb=(
            (vram_monitor.peak_mib / 1024.0)
            if vram_monitor and vram_monitor.peak_mib > 0
            else None
        ),
        inference_time_sec=max(0.0, elapsed),
    )


def build_image_ref(image_ref: str) -> str:
    if "@" in image_ref:
        return image_ref
    return f"aerast/musetalk@{image_ref}"


def evaluate(image_ref: str, challenges: list[Challenge]) -> tuple[float, dict]:
    if image_ref == "blacklist":
        return 0.0, {"error": "Blacklisted"}

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
    scoring_peak_vram_gb: list[float | None] = []
    scoring_inference_sec: list[float | None] = []
    challenge_metrics: list[dict] = []
    eff_cfg = EfficiencyConfig.from_env()
    try:
        logger.info(f"pulling image: {image_ref}")
        if not pull_image(image_ref):
            return 0.0, {"error": "Docker pull failed"}
        container_id = start_container(image_ref, str(job_dir))
        with _ACTIVE_CONTAINERS_LOCK:
            _ACTIVE_CONTAINERS.add(container_id)
        wait_for_ready(str(job_dir), READY_TIMEOUT_SEC)
        for challenge in warmup_challenges:
            warmup_results.append(
                _send_challenge(
                    job_dir,
                    challenge,
                    container_id=container_id,
                    measure_executor_vram_peak=False,
                )
            )
        for challenge in scoring_challenges:
            result = _send_challenge(
                job_dir,
                challenge,
                container_id=container_id,
                measure_executor_vram_peak=True,
            )
            scoring_results.append(result)
            scoring_peak_vram_gb.append(result.peak_vram_gb)
            scoring_inference_sec.append(result.inference_time_sec)
            if result.success:
                output_video = job_dir / "output" / f"{challenge.challenge_id}.mp4"
                if output_video.exists():
                    video_url = _upload_video_to_r2(output_video, challenge)
                    face_tmp = job_dir / "input" / f"face_{challenge.challenge_id}.png"
                    audio_ext = _infer_audio_extension(challenge.audio_bytes)
                    audio_tmp = (
                        job_dir / "input" / f"audio_{challenge.challenge_id}{audio_ext}"
                    )
                    face_tmp.write_bytes(challenge.face_bytes)
                    audio_tmp.write_bytes(challenge.audio_bytes)
                    try:
                        score_obj = score_video(
                            video_path=str(output_video),
                            reference_image_path=str(face_tmp),
                            audio_path=str(audio_tmp),
                            expected_transcript=challenge.text or None,
                            peak_vram_gb=result.peak_vram_gb,
                            inference_time_sec=result.inference_time_sec,
                            efficiency_config=eff_cfg,
                        )
                        scoring_quality.append(
                            float(
                                score_obj.get(
                                    "quality_score", score_obj.get("final", 0.0)
                                )
                            )
                        )
                        challenge_metrics.append(
                            {
                                "challenge_id": challenge.challenge_id,
                                "success": True,
                                # "host_time_sec": result.host_time_sec,
                                # "peak_vram_gb": result.peak_vram_gb,
                                # "inference_time_sec": result.inference_time_sec,
                                "video_url": video_url,
                                "identity": float(score_obj.get("identity", 0.0)),
                                "lipsync": float(score_obj.get("lipsync", 0.0)),
                                "audio": float(score_obj.get("audio", 0.0)),
                                "video": float(score_obj.get("video", 0.0)),
                                "temporal": float(score_obj.get("temporal", 0.0)),
                                "penalty": float(score_obj.get("penalty", 0.0)),
                                "wer": float(
                                    (score_obj.get("debug") or {}).get("wer", 1.0)
                                    if isinstance(score_obj.get("debug"), dict)
                                    else 1.0
                                ),
                                "quality_score": float(
                                    score_obj.get("quality_score", 0.0)
                                ),
                                "final_score": float(score_obj.get("final", 0.0)),
                                "efficiency": score_obj.get("efficiency"),
                            }
                        )
                    except Exception as exc:
                        logger.warning(
                            f"quality scoring failed challenge={challenge.challenge_id} "
                            f"image_ref={image_ref} err={exc}"
                        )
                        scoring_quality.append(0.0)
                        challenge_metrics.append(
                            {
                                "challenge_id": challenge.challenge_id,
                                "success": False,
                                "host_time_sec": result.host_time_sec,
                                "peak_vram_gb": result.peak_vram_gb,
                                "inference_time_sec": result.inference_time_sec,
                                "video_url": video_url,
                                "error": f"quality_scoring_failed: {exc}",
                            }
                        )
                    finally:
                        pass
                        _safe_remove(face_tmp)
                        _safe_remove(audio_tmp)
                else:
                    scoring_quality.append(0.0)
                    challenge_metrics.append(
                        {
                            "challenge_id": challenge.challenge_id,
                            "success": False,
                            "host_time_sec": result.host_time_sec,
                            "peak_vram_gb": result.peak_vram_gb,
                            "inference_time_sec": result.inference_time_sec,
                            "video_url": None,
                            "error": "output_video_missing",
                        }
                    )
            else:
                challenge_metrics.append(
                    {
                        "challenge_id": challenge.challenge_id,
                        "success": False,
                        "host_time_sec": result.host_time_sec,
                        "peak_vram_gb": result.peak_vram_gb,
                        "inference_time_sec": result.inference_time_sec,
                        "video_url": None,
                        "error": result.error or "challenge_failed",
                    }
                )
    finally:
        if container_id:
            stop_container(container_id)
        shutil.rmtree(job_dir, ignore_errors=True)
        remove_all_images()

    failures = [r for r in warmup_results + scoring_results if not r.success]
    if failures:
        errors = ", ".join(f"{r.challenge_id}:{r.error}" for r in failures)
        raise RuntimeError(f"challenge failures: {errors}")

    if not scoring_quality:
        return 0.0, {
            "scoring_count": len(scoring_challenges),
            "final_score": 0.0,
            "quality_score": 0.0,
            "challenge_metrics": challenge_metrics,
            "efficiency": {
                "peak_vram_gb": None,
                "inference_time_sec": None,
                "time_norm": 0.0,
                "vram_norm": 0.0,
                "efficiency_factor": 0.0,
            },
        }
    mean_quality = _avg(scoring_quality)
    eff = aggregate_efficiency_scores(
        per_challenge_peak_vram_gb=scoring_peak_vram_gb,
        per_challenge_inference_sec=scoring_inference_sec,
        mean_quality=mean_quality,
        cfg=eff_cfg,
    )
    logger.info(
        "efficiency aggregate: "
        f"mean_quality={mean_quality:.6f} "
        f"peak_vram_gb={eff.get('peak_vram_gb')} "
        f"inference_time_sec={eff.get('inference_time_sec')} "
        f"time_norm={eff.get('time_norm')} vram_norm={eff.get('vram_norm')} "
        f"efficiency_factor={eff.get('efficiency_factor')} "
        f"cap_violation={eff.get('cap_violation')}"
    )
    final_score = float(eff.get("final_score", mean_quality))
    metrics_payload = {
        "scoring_count": len(scoring_challenges),
        "final_score": final_score,
        "quality_score": mean_quality,
        "challenge_metrics": challenge_metrics,
        "efficiency": {
            "peak_vram_gb": eff.get("peak_vram_gb"),
            "inference_time_sec": eff.get("inference_time_sec"),
            "time_norm": eff.get("time_norm"),
            "vram_norm": eff.get("vram_norm"),
            "efficiency_factor": eff.get("efficiency_factor"),
            "cap_violation": eff.get("cap_violation"),
        },
        "updated_at": time.time(),
    }
    return final_score, metrics_payload


def load_challenges_from_dir(challenges_dir: str) -> list[Challenge]:
    return _load_challenges(challenges_dir)
