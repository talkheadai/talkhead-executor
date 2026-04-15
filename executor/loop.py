from __future__ import annotations

import base64
import binascii
import os
import threading
import time
from pathlib import Path

import bittensor as bt
from loguru import logger

from executor.evaluation.docker_runner import evaluate, load_challenges_from_dir, stop_all_running_containers
from executor.models import Challenge
from executor.state import MinerState
from executor.verify import _http_json, signed_subnet_headers

SUBNET_API_URL = os.getenv("SUBNET_API_URL", "https://subnet.talkhead.ai")

PENALTY_SCORE = 0

class EvaluationLoop:
    def __init__(
        self,
        state: MinerState,
        *,
        idle_sleep_sec: float = 1.0,
    ) -> None:
        wallet_name = os.getenv("WALLET_NAME", "default")
        wallet_hotkey = os.getenv("HOTKEY_NAME", "default")
        self.wallet = bt.Wallet(name=wallet_name, hotkey=wallet_hotkey)
        self._state = state
        self._idle_sleep_sec = idle_sleep_sec
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="evaluation-loop", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        stop_all_running_containers()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if not self._state.has_miners():
                time.sleep(self._idle_sleep_sec)
                continue

            try:
                challenges = self._fetch_challenge()
            except Exception as exc:
                logger.exception(f"failed to fetch challenge: {exc}")
                time.sleep(self._idle_sleep_sec)
                continue

            while not self._stop_event.is_set():
                pending = self._state.get_pending_miners()
                if not pending:
                    if self._state.commit_round_if_complete():
                        logger.info("round completed")
                    break

                for miner in pending:
                    print(f"Evaluating miner {miner.hotkey} {miner.image_ref}")
                    logger.info(
                        f"evaluation start: hotkey={miner.hotkey} image_ref={miner.image_ref}"
                    )
                    metrics_payload: dict | None = None
                    score = PENALTY_SCORE
                    try:
                        score, metrics_payload = evaluate(miner.image_ref, challenges)
                    except Exception as exc:
                        logger.exception(
                            f"evaluation failed: hotkey={miner.hotkey} "
                            f"image_ref={miner.image_ref} error={exc}"
                        )
                        metrics_payload = {
                            "image_ref": miner.image_ref,
                            "quality_score": 0.0,
                            "final_score": float(PENALTY_SCORE),
                            "challenge_metrics": [],
                            "error": str(exc),
                            "updated_at": time.time(),
                        }

                    updated = self._state.set_evaluation_result(
                        hotkey=miner.hotkey,
                        image_ref=miner.image_ref,
                        metrics=metrics_payload,
                    )
                    if updated:
                        logger.info(
                            f"evaluation end: hotkey={miner.hotkey} "
                            f"image_ref={miner.image_ref} staged_final_score={score}"
                        )
                    else:
                        logger.info(
                            f"evaluation discarded due to updated submission: "
                            f"hotkey={miner.hotkey} image_ref={miner.image_ref}"
                        )

    def _fetch_challenge(self) -> list[Challenge]:
        status, payload = _http_json(
            f"{SUBNET_API_URL.rstrip('/')}/challenge",
            "GET",
            headers=signed_subnet_headers(self.wallet, "/challenge"),
        )
        challenges = _parse_challenge_payload(payload)
        logger.info(f"new challenge fetched: count={len(challenges)}")
        return challenges


def _parse_challenge_payload(payload: object) -> list[Challenge]:
    if isinstance(payload, dict):
        if isinstance(payload.get("challenges_dir"), str):
            return load_challenges_from_dir(payload["challenges_dir"])
        if isinstance(payload.get("challenge_dir"), str):
            return load_challenges_from_dir(payload["challenge_dir"])
        if isinstance(payload.get("challenges"), list):
            return _parse_challenge_list(payload["challenges"])
    if isinstance(payload, list):
        return _parse_challenge_list(payload)
    raise ValueError("Unsupported /challenge payload format")


def _decode_base64(raw: object) -> bytes | None:
    """Subnet sends `face_bytes` / `audio_bytes` as base64-encoded strings."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip()
    if s.startswith("data:") and "," in s:
        s = s.split(",", 1)[1].strip()
    # Normalize padding to avoid accepting malformed payload silently.
    if len(s) % 4:
        s = s + ("=" * (4 - (len(s) % 4)))
    for altchars in (None, b"-_"):
        try:
            return base64.b64decode(s, altchars=altchars, validate=True)
        except (binascii.Error, ValueError):
            continue
    return None


def _parse_challenge_list(items: list[object]) -> list[Challenge]:
    challenges: list[Challenge] = []
    for idx, item in enumerate(items):
        challenge_id: str
        text: str
        face_bytes: bytes | None
        audio_bytes: bytes | None

        if isinstance(item, list) and len(item) >= 3:
            t0, t1, t2 = item[0], item[1], item[2]
            text = t0 if isinstance(t0, str) else str(t0)
            audio_bytes = _decode_base64(t1)
            face_bytes = _decode_base64(t2)
            challenge_id = str(idx)
        elif isinstance(item, dict):
            cid = item.get("challenge_id")
            challenge_id = cid if isinstance(cid, str) else str(idx)
            tx = item.get("text", "")
            text = tx if isinstance(tx, str) else str(tx)
            face_bytes = _decode_base64(item.get("face_base64"))
            audio_bytes = _decode_base64(item.get("audio_base64"))
            if face_bytes is None and isinstance(item.get("face_path"), str):
                p = Path(item["face_path"])
                if p.is_file():
                    face_bytes = p.read_bytes()
            if audio_bytes is None and isinstance(item.get("audio_path"), str):
                p = Path(item["audio_path"])
                if p.is_file():
                    audio_bytes = p.read_bytes()
        else:
            continue

        if face_bytes is None or audio_bytes is None:
            continue
        challenges.append(
            Challenge(
                challenge_id=challenge_id,
                text=text,
                audio_bytes=audio_bytes,
                face_bytes=face_bytes,
            )
        )
    if not challenges:
        raise ValueError("No valid challenges found in /challenge payload")
    return challenges
