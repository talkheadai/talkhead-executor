# Minimal Stateful Executor Service

This service maintains an in-memory table of miner submissions, evaluates miners continuously, and exposes HTTP APIs to update submissions and fetch scores.

## Endpoints

- `POST /update` to upsert miner submissions
- `GET /score` to fetch the full score table

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

Alternative:

```bash
python app.py
```

## Environment Variables

- `CHALLENGE_API_URL` (default: `http://127.0.0.1:9000/challenge`)
- `PENALTY_SCORE` (default: `9999`)
- `PORT` (default: `8000`)
- `STATE_FILE` (default: `./state.db`) — SQLite database path. If you previously used `./state.json`, set `STATE_FILE` to that path once: on first run the service creates `state.db` next to it and imports rows from the JSON file.

## Notes

- State is held in memory and persisted to a local SQLite database (WAL mode)
- Single process / single evaluation loop
