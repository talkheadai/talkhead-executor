"""Backward-compatible entrypoint: ``python app.py`` or ``uvicorn app:app``."""
from executor.app import app, run

__all__ = ["app", "run"]

if __name__ == "__main__":
    run()
