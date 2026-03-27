# Minimal Stateful Executor Service

This service maintains an in-memory table of miner submissions, evaluates miners continuously, and exposes HTTP APIs to update submissions and fetch scores.

## Layout

- `executor/` — installable package (`app`, `loop`, `state`, `models`, `verify`)
- `executor/evaluation/` — Docker miner run and challenge orchestration (`docker_runner`)
- `executor/scoring/` — reserved for output quality metrics (lipsync, identity, etc.)

## Endpoints

- `POST /update` to upsert miner submissions
- `GET /scores` to fetch the full score table

## Environment Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies from `pyproject.toml`:

```bash
pip install -e .
```

## Run

Start the API service:

```bash
executor-service
```

Alternatives:

```bash
python app.py
```

```bash
uvicorn executor.app:app --host 0.0.0.0 --port 8000
```

## Environment Variables

- `SUBNET_API_URL` (default in loop: `https://subnet.talkhead.ai`; challenges are fetched from `{SUBNET_API_URL}/challenge`)
- `PORT` (default: `8000`)
- `STATE_FILE` (default: `./state.db`) — SQLite database path. If you previously used `./state.json`, set `STATE_FILE` to that path once: on first run the service creates `state.db` next to it and imports rows from the JSON file.

## Notes

- State is held in memory and persisted to a local SQLite database (WAL mode)
- Single process / single evaluation loop
