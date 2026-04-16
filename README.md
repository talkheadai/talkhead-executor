# Talkhead Executor Service

This service maintains a table of miner submissions, evaluates miners continuously, and exposes APIs to update submissions and fetch scores.

> [REFERENCE]
> This repository is part of the Talkhead subnet. The full subnet description is available in [talkheadai/talkhead-subnet](https://github.com/talkheadai/talkhead-subnet).

## Endpoints

- `POST /update` to upsert miner submissions
- `GET /metrics` to fetch latest committed scoring metrics per miner (includes per-challenge entries)

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
python app.py
```

```bash
uvicorn executor.app:app --host 0.0.0.0 --port 8000
```

## Environment Variables

- `SUBNET_API_URL` (default in loop: `https://subnet.talkhead.ai`; challenges are fetched from `{SUBNET_API_URL}/challenge`)
- `PORT` (default: `8000`)
- `STATE_FILE` (default: `./state.db`) — SQLite database path.

## Quality Scoring

- Per-video quality is computed before efficiency is applied.
- Hard fail on transcript mismatch: if `wer > 0.60`, then `quality_score = 0.0`.
- Otherwise:
  - `blended = 0.35*identity + 0.35*lipsync + 0.15*video + 0.10*audio + 0.05*temporal - penalty`
  - `quality_score = clamp01(gate * blended)`
- Gate multipliers (applied multiplicatively):
  - If `identity_mode == "insightface"` and `face_detect_ratio < 0.80`, multiply gate by `0.2`.
  - If `identity < 0.35`, multiply gate by `0.1`.
  - If `sync_c < 0.45`, multiply gate by `0.2`.
- All component scores and `quality_score` are clamped to `[0, 1]`.

## Efficiency Scoring

- Final scoring is quality-first with a soft efficiency modifier: `final_score = quality_score * efficiency_factor`.
> [IMPORTANT]
> Winner selection is by maximum `final_score`: among all miner submissions, the highest `final_score` is preferred.
- `efficiency_factor = exp(-0.15 * time_norm - 0.10 * vram_norm)`.
- Normalization is robust and clamped (`time_norm`, `vram_norm` in `[0, 3]`).
- Preferred metric source is miner-reported `result.json` fields:
  - `efficiency.peak_vram_gb`
  - `efficiency.inference_time_sec`
- Limitation: executor cannot read container-internal PyTorch CUDA peak memory directly, so VRAM must be reported by the miner for accurate measurement.
- Fallback behavior:
  - Inference time can fall back to executor-observed wall-clock challenge time.
  - Peak VRAM can fall back to executor-observed `nvidia-smi` process memory for container PIDs.
  - This fallback excludes image pull and container startup (evaluation starts after ready), but may include lightweight file IPC/polling overhead.
- Optional tie-break helper is available in `executor.scoring.efficiency.should_prefer_candidate_a`: if two scores are within `EFFICIENCY_TIE_QUALITY_EPSILON`, lower inference time wins first, then lower peak VRAM.

Per-video score output now includes:

```json
{
  "efficiency": {
    "peak_vram_gb": 0.0,
    "inference_time_sec": 0.0,
    "time_norm": 0.0,
    "vram_norm": 0.0,
    "efficiency_factor": 0.0
  }
}
```

## Notes

- State is held in memory and persisted to a local SQLite database (WAL mode)
- Single process / single evaluation loop
- Metrics retention is bounded to the latest evaluation per miner, so storing all per-challenge metrics for `SCORING_COUNT=5` is practical even with many miners.

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
