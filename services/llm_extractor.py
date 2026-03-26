"""
services/llm_extractor.py
[Step 1 — LLM]
간판 이미지를 Gemini Vision에 보내 상호명/전화번호/업종/주소를 JSON으로 추출합니다.
google-genai 패키지 사용 (구 google-generativeai 대체)
"""
from pathlib import Path

import httpx
from google import genai
from google.genai import types

from config import settings
from schemas.extraction_schema import SignboardExtraction
from utils.json_utils import safe_parse_json
from utils.phone_utils import normalize_phone
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load_prompt() -> str:
    prompt_path = settings.PROMPTS_DIR / "extraction_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def _load_image_part(image_source: str) -> tuple[bytes, str]:
    """이미지를 bytes와 mime_type으로 반환합니다."""
    src = str(image_source)
    if src.startswith(("http://", "https://")):
        logger.info("[LLM] URL 이미지 다운로드: %s", src)
        resp = httpx.get(src, timeout=15)
        resp.raise_for_status()
        mime = resp.headers.get("content-type", "image/jpeg").split(";")[0]
        return resp.content, mime

    path = Path(src)
    if not path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")

    ext_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}
    mime = ext_map.get(path.suffix.lower(), "image/jpeg")
    logger.info("[LLM] 로컬 이미지 로드: %s (%s)", path.name, mime)
    return path.read_bytes(), mime


def extract_from_signboard(image_source: str) -> SignboardExtraction:
    """
    간판 이미지에서 사업자 관련 정보를 추출합니다.

    Args:
        image_source: 로컬 이미지 경로 또는 공개 URL

    Returns:
        SignboardExtraction
    """
    logger.info("[LLM] 간판 이미지 분석 시작: %s", image_source)

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    system_prompt = _load_prompt()
    image_bytes, mime_type = _load_image_part(image_source)

    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            "이 간판 이미지에서 사업자 정보를 추출해주세요.",
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=2048,   # 1024 → 2048 (잘림 방지)
            temperature=0.1,
        ),
    )

    raw_text = response.text or ""

    # 응답이 토큰 한도로 잘렸는지 감지
    finish_reason = None
    try:
        finish_reason = response.candidates[0].finish_reason
    except (AttributeError, IndexError):
        pass
    if str(finish_reason) in ("MAX_TOKENS", "2"):  # 2 = MAX_TOKENS enum 값
        logger.warning("[LLM] 응답이 토큰 한도로 잘렸습니다 (finish_reason=%s). 복구 시도...", finish_reason)

    logger.debug("[LLM] 원시 응답: %s", raw_text[:300])

    parsed = safe_parse_json(raw_text)
    if parsed is None:
        logger.warning("[LLM] JSON 파싱 실패 — 빈 추출 결과 반환")
        return SignboardExtraction(raw_text=raw_text)

    if parsed.get("phone"):
        parsed["phone"] = normalize_phone(parsed["phone"])

    extraction = SignboardExtraction(**parsed)
    logger.info(
        "[LLM] 추출 완료 — 상호명: %s / 전화: %s / 업종: %s",
        extraction.business_name,
        extraction.phone,
        extraction.industry,
    )
    return extraction