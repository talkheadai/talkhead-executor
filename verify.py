from __future__ import annotations

import logging
import os
import json
import time
from binascii import unhexlify
from urllib import request, error

from dotenv import load_dotenv
from fastapi import Header, HTTPException, Request

try:
    from substrateinterface import Keypair
except Exception:  # pragma: no cover - optional dependency
    Keypair = None

try:
    import bittensor as bt
except Exception:  # pragma: no cover - optional dependency
    bt = None


logger = logging.getLogger("talkhead.verify")

load_dotenv()

MAX_SKEW_SECONDS = int(os.getenv("VALIDATOR_MAX_SKEW_SECONDS", "120"))
YOUR_HOTKEY = bt.Wallet(name=os.getenv("WALLET_NAME", "default"), hotkey=os.getenv("HOTKEY_NAME", "default")).hotkey.ss58_address


def keypair_from_hotkey(
    hotkey: str,
    timestamp: float,
) -> Keypair:
    if Keypair is None:
        raise HTTPException(
            status_code=503,
            detail="Signature verification unavailable: substrateinterface not installed",
        )

    try:
        ts = float(timestamp)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Invalid validator timestamp"
        ) from exc

    if abs(time.time() - ts) > MAX_SKEW_SECONDS:
        raise HTTPException(
            status_code=401, detail=f"Timestamp out of range for {hotkey}"
        )

    try:
        keypair = Keypair(ss58_address=hotkey, ss58_format=42)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid hotkey {hotkey}") from exc

    return keypair


def verify_hotkey_signature(
    message: str,
    hotkey: str,
    signature: str,
    timestamp: int,
    keypair: Keypair,
) -> bool:

    message = f"<Bytes>{message}:{timestamp}:{hotkey}</Bytes>"
    try:
        real_signature = unhexlify(signature.encode())
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid signature for {hotkey}"
        ) from exc

    if not keypair.verify(data=message, signature=real_signature):
        raise HTTPException(status_code=401, detail=f"Invalid signature for {hotkey}")
    return hotkey


def verify_validator_signature(
    request: Request,
    x_hotkey: str = Header(..., alias="X-Hotkey"),
    x_timestamp: str = Header(..., alias="X-Timestamp"),
    x_signature: str = Header(..., alias="X-Signature"),
) -> str:

    try:
        keypair = keypair_from_hotkey(x_hotkey, x_timestamp)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid hotkey {x_hotkey}"
        ) from exc

    if x_hotkey != YOUR_HOTKEY:
        raise HTTPException(
            status_code=403,
            detail=f"Hotkey not allowed: {x_hotkey}",
        )

    try:
        verify_hotkey_signature(
            request.url.path, x_hotkey, x_signature, int(x_timestamp), keypair
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return x_hotkey


def signature_hex(wallet: bt.Wallet, message: str) -> str:
    signed = wallet.hotkey.sign(message.encode("utf-8"))
    if isinstance(signed, bytes):
        return signed.hex()
    if hasattr(signed, "hex"):
        return signed.hex()
    return str(signed)

def signed_subnet_headers(wallet: bt.Wallet, message: str = "") -> dict[str, str]:
    hotkey = wallet.hotkey.ss58_address
    timestamp = str(int(time.time()))
    signature = signature_hex(wallet, f"<Bytes>{message}:{timestamp}:{hotkey}</Bytes>")
    return {
        "X-Hotkey": hotkey,
        "X-Timestamp": timestamp,
        "X-Signature": signature,
    }


def _http_json(
    url: str,
    method: str,
    body: object | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> tuple[int, object]:
    payload = None
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
    req = request.Request(url, data=payload, method=method)
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)

    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else None
    except error.HTTPError as http_err:
        raw = http_err.read().decode("utf-8")
        try:
            return http_err.code, json.loads(raw) if raw else None
        except json.JSONDecodeError:
            return http_err.code, raw
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc)
