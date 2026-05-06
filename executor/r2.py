import base64
import binascii
import logging
import os
import dotenv
dotenv.load_dotenv()
import re
import uuid
from datetime import datetime
from typing import Optional
import requests
from loguru import logger

try:
    import boto3  # type: ignore[import]
    from botocore.config import Config  # type: ignore[import]
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore[import]
except ImportError:  # pragma: no cover - boto3 is optional at runtime
    boto3 = None
    Config = None

    class BotoCoreError(Exception):  # type: ignore
        """Fallback placeholder when botocore is missing."""

    class ClientError(Exception):  # type: ignore
        """Fallback placeholder when botocore is missing."""


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _slugify(text: Optional[str], max_length: int = 40) -> str:
    if not text:
        return "clip"
    slug = _NON_ALNUM.sub("-", text.lower()).strip("-")
    if not slug:
        return "clip"
    return slug[:max_length].strip("-") or "clip"


def _build_object_key(prompt: Optional[str]) -> str:
    timestamp = datetime.utcnow()
    slug = _slugify(prompt)
    unique = uuid.uuid4().hex[:8]
    return f"videos/{timestamp:%Y/%m/%d}/{timestamp:%H%M%S}_{slug}_{unique}.mp4"


def _get_endpoint(account_id: Optional[str]) -> Optional[str]:
    custom_endpoint = os.getenv("CLOUDFLARE_R2_ENDPOINT")
    if custom_endpoint:
        return custom_endpoint.rstrip("/")
    if account_id:
        return f"https://{account_id}.r2.cloudflarestorage.com"
    return None


def upload_video_base64_to_r2(
    video_b64: str, *, prompt: Optional[str] = None, credentials: Optional[dict] = None
) -> Optional[str]:
    """
    Upload a base64-encoded MP4 to Cloudflare R2 and return its public URL.

    Returns:
        Optional[str]: Fully-qualified, publicly-accessible URL or None on failure.
    """

    if not video_b64:
        logger.warning("No video data provided for R2 upload.")
        return None

    if boto3 is None:
        logger.warning("boto3 is not installed; skipping R2 upload.")
        return None

    credentials = credentials or {}
    bucket = credentials.get("bucket", os.getenv("CLOUDFLARE_R2_BUCKET"))
    account_id = credentials.get("account_id", os.getenv("CLOUDFLARE_R2_ACCOUNT_ID"))
    access_key = credentials.get("access_key_id", os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID"))
    secret_key = credentials.get("secret_access_key", os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY"))
    public_base_url = credentials.get("public_base_url", os.getenv("CLOUDFLARE_R2_PUBLIC_BASE_URL"))
    endpoint_url = _get_endpoint(account_id)

    if not all([bucket, access_key, secret_key, public_base_url, endpoint_url]):
        logger.warning(
            "Missing Cloudflare R2 configuration; bucket=%s endpoint=%s public_base_url=%s",
            bool(bucket),
            bool(endpoint_url),
            bool(public_base_url),
        )
        return None

    try:
        video_bytes = base64.b64decode(video_b64)
    except (ValueError, binascii.Error) as err:
        logger.error("Invalid video_base64 payload; unable to upload to R2: %s", err)
        return None

    object_key = _build_object_key(prompt)

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    try:
        client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=video_bytes,
            ContentType="video/mp4",
        )
    except (BotoCoreError, ClientError) as err:
        logger.error(f"Failed to upload video to R2: {err}")
        return None

    return f"{public_base_url.rstrip('/')}/{object_key}"

def download_video_from_url(url: str) -> Optional[str]:
    """
    Download a video from a URL and return its base64-encoded string.
    """
    response = requests.get(url)
    return base64.b64encode(response.content).decode("ascii")