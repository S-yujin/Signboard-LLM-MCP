"""
utils/json_utils.py
JSON 파싱/직렬화 헬퍼
"""
import json
import re
from typing import Any, Optional

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def safe_parse_json(text: str) -> Optional[dict]:
    """
    LLM 응답 텍스트에서 JSON을 안전하게 파싱합니다.
    - 마크다운 코드블록(```json ... ```) 제거
    - 앞뒤 공백 제거
    - 파싱 실패 시 None 반환 (예외를 밖으로 전파하지 않음)
    """
    if not text:
        return None

    # 코드블록 제거
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    # 중괄호 시작 위치 찾기 (LLM이 앞에 설명을 붙이는 경우 대비)
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        logger.warning("응답에서 JSON 객체를 찾지 못했습니다: %s", cleaned[:100])
        return None

    json_str = cleaned[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error("JSON 파싱 오류: %s — 원본(앞 200자): %s", e, json_str[:200])
        return None


def pretty_json(obj: Any, ensure_ascii: bool = False) -> str:
    """Python 객체를 들여쓰기 2칸의 JSON 문자열로 변환합니다."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, ensure_ascii=ensure_ascii, indent=2, default=str)


def save_json(obj: Any, path: str) -> None:
    """객체를 JSON 파일로 저장합니다."""
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(pretty_json(obj))
    logger.info("JSON 저장 완료: %s", path)