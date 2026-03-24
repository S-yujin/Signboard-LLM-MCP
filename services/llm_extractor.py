"""
services/llm_extractor.py
[Step 1 — LLM]
간판 이미지를 GPT-4o Vision에 보내 상호명/전화번호/업종/주소를 JSON으로 추출합니다.
"""
from openai import OpenAI

from config import settings
from schemas.extraction_schema import SignboardExtraction
from services.image_service import load_image_block
from utils.json_utils import safe_parse_json
from utils.phone_utils import normalize_phone
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load_prompt() -> str:
    """prompts/extraction_prompt.txt를 읽어 반환합니다."""
    prompt_path = settings.PROMPTS_DIR / "extraction_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def _to_openai_image_block(image_block: dict) -> dict:
    """
    image_service.load_image_block() 반환값을 OpenAI content 형식으로 변환합니다.

    Anthropic 형식:
        {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
        {"type": "image", "source": {"type": "url", "url": "..."}}

    OpenAI 형식:
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        {"type": "image_url", "image_url": {"url": "https://..."}}
    """
    source = image_block["source"]
    if source["type"] == "url":
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    else:
        data_url = f"data:{source['media_type']};base64,{source['data']}"
        return {"type": "image_url", "image_url": {"url": data_url}}


def extract_from_signboard(image_source: str) -> SignboardExtraction:
    """
    간판 이미지에서 사업자 관련 정보를 추출합니다.

    Args:
        image_source: 로컬 이미지 경로 또는 공개 URL

    Returns:
        SignboardExtraction: 추출된 필드와 신뢰도 점수
    """
    logger.info("[LLM] 간판 이미지 분석 시작: %s", image_source)

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    system_prompt = _load_prompt()

    raw_block = load_image_block(image_source)
    openai_image = _to_openai_image_block(raw_block)

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        max_tokens=settings.OPENAI_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    openai_image,
                    {"type": "text", "text": "이 간판 이미지에서 사업자 정보를 추출해주세요."},
                ],
            },
        ],
    )

    raw_text = response.choices[0].message.content or ""
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