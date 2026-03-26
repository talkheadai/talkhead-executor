import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI

from loop import EvaluationLoop
from models import MinerScoreResponse, MinerSubmission
from state import MinerState
from verify import verify_validator_signature


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

load_dotenv()

SUBNET_API_URL = os.getenv("SUBNET_API_URL", "http://127.0.0.1:9000")

PENALTY_SCORE = -1
PORT = int(os.getenv("PORT", "8000"))
STATE_FILE = os.getenv("STATE_FILE", "./state.db")

state = MinerState(state_file=STATE_FILE)
eval_loop = EvaluationLoop(
    state=state,
)


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    eval_loop.start()
    try:
        yield
    finally:
        eval_loop.stop()


app = FastAPI(
    title="Minimal Stateful Executor",
    lifespan=_lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.post("/update", dependencies=[Depends(verify_validator_signature)])
def update(submissions: list[MinerSubmission]) -> dict[str, int]:
    state.upsert_submissions(submissions)
    return {"count": len(submissions)}


@app.get("/scores", response_model=list[MinerScoreResponse], dependencies=[Depends(verify_validator_signature)])
def score() -> list[MinerScoreResponse]:
    return state.list_scores()


def run() -> None:
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)


if __name__ == "__main__":
    run()
