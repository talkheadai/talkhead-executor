import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, Response, status

from executor.loop import EvaluationLoop
from executor.models import MinerMetricsResponse, MinerSubmission
from executor.state import MinerState
from executor.verify import verify_validator_signature


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


@app.get("/metrics", response_model=list[MinerMetricsResponse], dependencies=[Depends(verify_validator_signature)])
def metrics(response: Response, if_none_match: str | None = Header(default=None, alias="If-None-Match")) -> list[MinerMetricsResponse] | Response:
    rows, etag = state.list_metrics_with_etag()
    response.headers["ETag"] = etag
    if if_none_match:
        candidates = {part.strip() for part in if_none_match.split(",") if part.strip()}
        if etag in candidates or "*" in candidates:
            return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers={"ETag": etag})
    return rows


def run() -> None:
    uvicorn.run("executor.app:app", host="0.0.0.0", port=PORT, reload=False)


if __name__ == "__main__":
    run()
