from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from loguru import logger

from executor.models import MinerMetricsResponse, MinerRecord, MinerSubmission


def _resolve_db_and_legacy_paths(state_file: Path) -> tuple[Path, Path | None]:
    if state_file.suffix.lower() == ".json":
        return state_file.with_suffix(".db"), state_file
    return state_file, state_file.with_suffix(".json")


class MinerState:
    def __init__(self, state_file: str | None = None) -> None:
        self._lock = threading.Lock()
        self._miners: dict[str, MinerRecord] = {}
        self._metrics_version: int = 0
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
                metrics_json TEXT,
                coming_metrics_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS state_meta (
                key TEXT PRIMARY KEY NOT NULL,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO state_meta (key, value) VALUES ('metrics_version', '0')"
        )

    @staticmethod
    def _read_metrics_version(conn: sqlite3.Connection) -> int:
        try:
            row = conn.execute(
                "SELECT value FROM state_meta WHERE key='metrics_version'"
            ).fetchone()
            if row and row[0] is not None:
                return int(str(row[0]))
        except Exception:
            return 0
        return 0

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
            metrics = item.get("metrics")
            coming_metrics = item.get("coming_metrics")
            if not isinstance(hotkey, str) or not isinstance(image_ref, str):
                continue
            if not isinstance(submit_time, (int, float)):
                continue
            loaded[hotkey] = MinerRecord(
                hotkey=hotkey,
                image_ref=image_ref,
                submit_time=float(submit_time),
                metrics=metrics if isinstance(metrics, dict) else None,
                coming_metrics=coming_metrics if isinstance(coming_metrics, dict) else None,
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
            self._metrics_version = self._read_metrics_version(conn)
            if db_existed:
                cur = conn.execute(
                    "SELECT hotkey, image_ref, submit_time, metrics_json, coming_metrics_json "
                    "FROM miners ORDER BY submit_time"
                )
                for hotkey, image_ref, submit_time, metrics_json, coming_metrics_json in cur.fetchall():
                    metrics: dict[str, Any] | None = None
                    coming_metrics: dict[str, Any] | None = None
                    if isinstance(metrics_json, str) and metrics_json.strip():
                        try:
                            parsed = json.loads(metrics_json)
                            if isinstance(parsed, dict):
                                metrics = parsed
                        except Exception:
                            metrics = None
                    if isinstance(coming_metrics_json, str) and coming_metrics_json.strip():
                        try:
                            parsed = json.loads(coming_metrics_json)
                            if isinstance(parsed, dict):
                                coming_metrics = parsed
                        except Exception:
                            coming_metrics = None
                    self._miners[hotkey] = MinerRecord(
                        hotkey=hotkey,
                        image_ref=image_ref,
                        submit_time=float(submit_time),
                        metrics=metrics,
                        coming_metrics=coming_metrics,
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

    def _save_locked(self, *, bump_metrics_version: bool = False) -> None:
        if self._db_path is None:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            (
                r.hotkey,
                r.image_ref,
                r.submit_time,
                json.dumps(r.metrics) if isinstance(r.metrics, dict) else None,
                json.dumps(r.coming_metrics) if isinstance(r.coming_metrics, dict) else None,
            )
            for r in sorted(self._miners.values(), key=lambda r: r.submit_time)
        ]
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._ensure_schema(conn)
            conn.execute("DELETE FROM miners")
            conn.executemany(
                "INSERT INTO miners (hotkey, image_ref, submit_time, metrics_json, coming_metrics_json) "
                "VALUES (?,?,?,?,?)",
                rows,
            )
            if bump_metrics_version:
                self._metrics_version += 1
                conn.execute(
                    "UPDATE state_meta SET value = ? WHERE key='metrics_version'",
                    (str(self._metrics_version),),
                )
            conn.commit()
        finally:
            conn.close()

    def upsert_submissions(self, submissions: list[MinerSubmission]) -> None:
        changed = False
        affects_committed_metrics = False
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
                    if existing.metrics is not None:
                        affects_committed_metrics = True
                    self._miners[submission.hotkey] = MinerRecord(
                        hotkey=submission.hotkey,
                        image_ref=submission.image_ref,
                        submit_time=submission.submit_time,
                        metrics=None,
                        coming_metrics=None,
                    )
                    changed = True
                    logger.info(
                        f"image_ref updated: hotkey={submission.hotkey} "
                        f"image_ref={submission.image_ref}"
                    )
                    continue

                if existing.submit_time != submission.submit_time:
                    if existing.metrics is not None:
                        affects_committed_metrics = True
                    self._miners[submission.hotkey] = MinerRecord(
                        hotkey=existing.hotkey,
                        image_ref=existing.image_ref,
                        submit_time=submission.submit_time,
                        metrics=existing.metrics,
                        coming_metrics=existing.coming_metrics,
                    )
                    changed = True
                    logger.info(
                        f"submit_time updated: hotkey={submission.hotkey} "
                        f"submit_time={submission.submit_time}"
                    )
            if changed:
                self._save_locked(bump_metrics_version=affects_committed_metrics)

    def list_metrics(self) -> list[MinerMetricsResponse]:
        with self._lock:
            records = sorted(self._miners.values(), key=lambda r: r.submit_time)
            return [
                MinerMetricsResponse(
                    hotkey=record.hotkey,
                    image_ref=record.image_ref,
                    submit_time=record.submit_time,
                    metrics=record.metrics,
                )
                for record in records
                if record.metrics is not None
            ]

    def list_metrics_with_etag(self) -> tuple[list[MinerMetricsResponse], str]:
        with self._lock:
            records = sorted(self._miners.values(), key=lambda r: r.submit_time)
            metrics = [
                MinerMetricsResponse(
                    hotkey=record.hotkey,
                    image_ref=record.image_ref,
                    submit_time=record.submit_time,
                    metrics=record.metrics,
                )
                for record in records
                if record.metrics is not None
            ]
            etag = f'W/"metrics-{self._metrics_version}"'
            return metrics, etag

    def has_miners(self) -> bool:
        with self._lock:
            return bool(self._miners)

    def get_pending_miners(self) -> list[MinerRecord]:
        with self._lock:
            pending = [m for m in self._miners.values() if m.coming_metrics is None]
            pending.sort(key=lambda r: r.submit_time)
            # Return copies to keep internal state isolated.
            return [replace(record) for record in pending]

    def set_evaluation_result(
        self,
        *,
        hotkey: str,
        image_ref: str,
        metrics: dict[str, Any] | None,
    ) -> bool:
        """
        Atomically stage next-round metrics snapshot for one miner.
        """
        with self._lock:
            record = self._miners.get(hotkey)
            if record is None:
                return False
            if record.image_ref != image_ref:
                return False
            record.coming_metrics = metrics
            self._save_locked()
            return True

    def commit_round_if_complete(self) -> bool:
        with self._lock:
            if not self._miners:
                return False
            if any(record.coming_metrics is None for record in self._miners.values()):
                return False
            for record in self._miners.values():
                record.metrics = record.coming_metrics
                record.coming_metrics = None
            self._save_locked(bump_metrics_version=True)
            return True
