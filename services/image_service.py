"""
services/image_service.py
이미지를 로드하여 Anthropic API에 전달할 content 블록으로 변환합니다.
로컬 파일과 HTTP URL 모두 지원합니다.
"""
import base64
import mimetypes
from pathlib import Path
from typing import Union

import httpx

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Anthropic이 지원하는 이미지 MIME 타입
SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _ext_to_mime(ext: str) -> str:
    ext = ext.lower().lstrip(".")
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/jpeg")


def load_image_block(source: Union[str, Path]) -> dict:
    """
    이미지 소스(경로 또는 URL)를 받아 Anthropic messages content 블록을 반환합니다.

    반환 형식:
        - URL  → {"type": "image", "source": {"type": "url", "url": "..."}}
        - 파일 → {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}

    Raises:
        FileNotFoundError: 로컬 파일이 존재하지 않을 때
        ValueError: 지원하지 않는 이미지 형식일 때
        httpx.HTTPError: URL 다운로드 실패 시
    """
    source_str = str(source)

    # ── URL ────────────────────────────────────────────────────────────────────
    if source_str.startswith(("http://", "https://")):
        logger.info("URL 이미지 사용: %s", source_str)
        # Anthropic API는 공개 URL을 직접 받을 수 있음
        # 단, 접근이 불가한 경우 base64로 다운로드
        try:
            return {"type": "image", "source": {"type": "url", "url": source_str}}
        except Exception:
            logger.warning("URL 직접 전달 실패, base64로 다운로드 시도...")
            response = httpx.get(source_str, timeout=15)
            response.raise_for_status()
            raw = response.content
            mime = response.headers.get("content-type", "image/jpeg").split(";")[0]
            return _make_base64_block(raw, mime)

    # ── 로컬 파일 ───────────────────────────────────────────────────────────────
    path = Path(source_str)
    if not path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")

    mime = _ext_to_mime(path.suffix)
    if mime not in SUPPORTED_MIME_TYPES:
        raise ValueError(f"지원하지 않는 이미지 형식: {path.suffix}. 지원 형식: jpg/png/gif/webp")

    logger.info("로컬 이미지 로드: %s (%s)", path.name, mime)
    raw = path.read_bytes()
    return _make_base64_block(raw, mime)


def _make_base64_block(raw: bytes, mime: str) -> dict:
    """bytes → Anthropic base64 이미지 블록"""
    data = base64.standard_b64encode(raw).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": mime, "data": data},
    }