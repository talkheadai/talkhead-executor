from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path

from loguru import logger

from executor.models import MinerRecord, MinerScoreResponse, MinerSubmission


def _resolve_db_and_legacy_paths(state_file: Path) -> tuple[Path, Path | None]:
    if state_file.suffix.lower() == ".json":
        return state_file.with_suffix(".db"), state_file
    return state_file, state_file.with_suffix(".json")


class MinerState:
    def __init__(self, state_file: str | None = None) -> None:
        self._lock = threading.Lock()
        self._miners: dict[str, MinerRecord] = {}
        if state_file:
            raw = Path(state_file)
            self._db_path, self._legacy_json_path = _resolve_db_and_legacy_paths(raw)
        else:
            self._db_path = None
            self._legacy_json_path = None
        self._load_from_disk()

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS miners (
                hotkey TEXT PRIMARY KEY NOT NULL,
                image_ref TEXT NOT NULL,
                submit_time REAL NOT NULL,
                score REAL NOT NULL DEFAULT -1,
                coming_score REAL NOT NULL DEFAULT -1
            )
            """
        )

    @staticmethod
    def _miners_from_json_payload(payload: object) -> dict[str, MinerRecord]:
        if not isinstance(payload, dict):
            return {}
        miners = payload.get("miners")
        if not isinstance(miners, list):
            return {}

        loaded: dict[str, MinerRecord] = {}
        for item in miners:
            if not isinstance(item, dict):
                continue
            hotkey = item.get("hotkey")
            image_ref = item.get("image_ref")
            submit_time = item.get("submit_time")
            score = item.get("score", -1.0)
            coming_score = item.get("coming_score", -1.0)
            if not isinstance(hotkey, str) or not isinstance(image_ref, str):
                continue
            if not isinstance(submit_time, (int, float)):
                continue
            if not isinstance(score, (int, float)):
                score = -1.0
            if not isinstance(coming_score, (int, float)):
                coming_score = -1.0
            loaded[hotkey] = MinerRecord(
                hotkey=hotkey,
                image_ref=image_ref,
                submit_time=float(submit_time),
                score=float(score),
                coming_score=float(coming_score),
            )
        return loaded

    def _load_miners_from_json_file(self, path: Path) -> dict[str, MinerRecord]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"failed to load legacy state file {path}: {exc}")
            return {}
        return self._miners_from_json_payload(payload)

    def _load_from_disk(self) -> None:
        if self._db_path is None:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        db_existed = self._db_path.exists()

        conn = sqlite3.connect(str(self._db_path))
        try:
            self._ensure_schema(conn)
            if db_existed:
                cur = conn.execute(
                    "SELECT hotkey, image_ref, submit_time, score, coming_score "
                    "FROM miners ORDER BY submit_time"
                )
                for hotkey, image_ref, submit_time, score, coming_score in cur.fetchall():
                    self._miners[hotkey] = MinerRecord(
                        hotkey=hotkey,
                        image_ref=image_ref,
                        submit_time=float(submit_time),
                        score=float(score),
                        coming_score=float(coming_score),
                    )
                logger.info(f"loaded {len(self._miners)} miners from sqlite")
        finally:
            conn.close()

        if not db_existed and self._legacy_json_path and self._legacy_json_path.exists():
            loaded = self._load_miners_from_json_file(self._legacy_json_path)
            if loaded:
                with self._lock:
                    self._miners = loaded
                    self._save_locked()
                logger.info(
                    f"migrated {len(loaded)} miners from {self._legacy_json_path} to sqlite"
                )

    def _save_locked(self) -> None:
        if self._db_path is None:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            (r.hotkey, r.image_ref, r.submit_time, r.score, r.coming_score)
            for r in sorted(self._miners.values(), key=lambda r: r.submit_time)
        ]
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._ensure_schema(conn)
            conn.execute("DELETE FROM miners")
            conn.executemany(
                "INSERT INTO miners (hotkey, image_ref, submit_time, score, coming_score) "
                "VALUES (?,?,?,?,?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_submissions(self, submissions: list[MinerSubmission]) -> None:
        changed = False
        with self._lock:
            for submission in submissions:
                existing = self._miners.get(submission.hotkey)
                if existing is None:
                    self._miners[submission.hotkey] = MinerRecord(
                        hotkey=submission.hotkey,
                        image_ref=submission.image_ref,
                        submit_time=submission.submit_time,
                    )
                    changed = True
                    logger.info(f"new miner added: hotkey={submission.hotkey}")
                    continue

                if existing.image_ref != submission.image_ref:
                    self._miners[submission.hotkey] = MinerRecord(
                        hotkey=submission.hotkey,
                        image_ref=submission.image_ref,
                        submit_time=submission.submit_time,
                        score=-1.0,
                        coming_score=-1.0,
                    )
                    changed = True
                    logger.info(
                        f"image_ref updated: hotkey={submission.hotkey} "
                        f"image_ref={submission.image_ref}"
                    )
                    continue

                if existing.submit_time != submission.submit_time:
                    self._miners[submission.hotkey] = MinerRecord(
                        hotkey=existing.hotkey,
                        image_ref=existing.image_ref,
                        submit_time=submission.submit_time,
                        score=existing.score,
                        coming_score=existing.coming_score,
                    )
                    changed = True
                    logger.info(
                        f"submit_time updated: hotkey={submission.hotkey} "
                        f"submit_time={submission.submit_time}"
                    )
            if changed:
                self._save_locked()

    def list_scores(self) -> list[MinerScoreResponse]:
        with self._lock:
            records = sorted(self._miners.values(), key=lambda r: r.submit_time)
            return [
                MinerScoreResponse(
                    hotkey=record.hotkey,
                    image_ref=record.image_ref,
                    submit_time=record.submit_time,
                    score=record.score,
                    coming_score=record.coming_score,
                )
                for record in records
                if record.score != -1.0
            ]

    def has_miners(self) -> bool:
        with self._lock:
            return bool(self._miners)

    def get_pending_miners(self) -> list[MinerRecord]:
        with self._lock:
            pending = [m for m in self._miners.values() if m.coming_score == -1.0]
            pending.sort(key=lambda r: r.submit_time)
            # Return copies to keep internal state isolated.
            return [replace(record) for record in pending]

    def set_coming_score(self, hotkey: str, image_ref: str, coming_score: float) -> bool:
        with self._lock:
            record = self._miners.get(hotkey)
            if record is None:
                return False
            if record.image_ref != image_ref:
                # Submission changed while evaluation was running.
                return False
            record.coming_score = coming_score
            self._save_locked()
            return True

    def commit_round_if_complete(self) -> bool:
        with self._lock:
            if not self._miners:
                return False
            if any(record.coming_score == -1.0 for record in self._miners.values()):
                return False
            for record in self._miners.values():
                record.score = record.coming_score
                record.coming_score = -1.0
            self._save_locked()
            return True
