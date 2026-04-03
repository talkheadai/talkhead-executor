# Minimal Stateful Executor Service

This service maintains an in-memory table of miner submissions, evaluates miners continuously, and exposes HTTP APIs to update submissions and fetch scores.

## Layout

- `executor/` — installable package (`app`, `loop`, `state`, `models`, `verify`)
- `executor/evaluation/` — Docker miner run and challenge orchestration (`docker_runner`)
- `executor/scoring/` — multi-metric quality scoring (identity, lipsync, audio, video, temporal, penalties)

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

## System Prerequisites (Scoring)

For robust audio decoding in the scoring pipeline, install `ffmpeg` on the host:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Then verify:

```bash
ffmpeg -version
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

## GPU / Docker Troubleshooting

If `nvidia-smi` shows `Failed to initialize NVML: Driver/library version mismatch`, your loaded NVIDIA kernel module does not match the installed NVIDIA user-space libraries.

1. Reboot the host to load the current NVIDIA module:

```bash
sudo reboot
```

2. After reboot, verify GPU access:

```bash
nvidia-smi
```

3. If Docker shows `could not select device driver "" with capabilities: [[gpu]]`, install and configure NVIDIA Container Toolkit:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

4. Validate GPU access from containers:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
