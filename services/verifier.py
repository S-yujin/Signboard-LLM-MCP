"""
services/verifier.py
[Step 2 & 3 — MCP 에이전트]
GPT-4o에게 MCP 도구를 제공하고, 사업자 후보 조회 + 상태 검증을 수행하는 에이전트 루프.
"""
import json
from openai import OpenAI

from config import settings
from schemas.extraction_schema import SignboardExtraction
from services.mcp_client import MCP_TOOLS, dispatch_tool
from utils.json_utils import safe_parse_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_AGENT_SYSTEM_PROMPT = """당신은 사업자 정보 검증 전문 에이전트입니다.
간판 분석 결과를 바탕으로 다음 단계를 순서대로 수행하세요:

1. lookup_business_candidates 도구를 호출하여 상호명/전화/업종으로 후보를 조회합니다.
2. 후보가 있으면 verify_business_status 도구로 각 사업자등록번호의 상태를 확인합니다.
3. 모든 정보를 종합하여 아래 JSON 형식으로만 최종 응답합니다.
   마크다운 코드블록이나 추가 설명 없이 순수 JSON만 출력하세요.

최종 응답 스키마:
{
  "status": "verified | partial | not_found",
  "candidates": [
    {
      "registration_number": "10자리 문자열",
      "business_name": "상호명",
      "representative": "대표자명 또는 null",
      "address": "주소 또는 null",
      "industry": "업종 또는 null",
      "phone": "전화번호 또는 null",
      "business_status": "계속사업자 | 휴업자 | 폐업자 | unknown",
      "tax_type": "과세유형 또는 null",
      "status_verified": true/false,
      "confidence_score": 0.0~1.0,
      "source": "조회 출처"
    }
  ],
  "best_match": { ...가장 신뢰도 높은 후보 또는 null },
  "warnings": ["처리 중 발생한 경고 메시지 배열"]
}

status 선택 기준:
- verified  : 사업자등록번호 + 국세청 상태 모두 확인
- partial   : 후보는 찾았으나 상태 검증 불완전
- not_found : 후보를 찾지 못함
"""

# OpenAI function calling 형식으로 변환
def _to_openai_tools(mcp_tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in mcp_tools
    ]


def run_verification_agent(extraction: SignboardExtraction) -> dict:
    """
    LLM이 추출한 간판 정보를 기반으로 MCP 에이전트 루프를 실행합니다.

    Args:
        extraction: LLM 간판 추출 결과

    Returns:
        dict: 후보 목록 및 검증 결과
    """
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    openai_tools = _to_openai_tools(MCP_TOOLS)

    messages = [
        {"role": "system", "content": _AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "다음 간판 분석 결과로 사업자를 검증해주세요:\n"
                + json.dumps(extraction.model_dump(), ensure_ascii=False, indent=2)
            ),
        },
    ]

    for iteration in range(1, settings.MCP_MAX_ITERATIONS + 1):
        logger.debug("[Agent] 반복 %d / %d", iteration, settings.MCP_MAX_ITERATIONS)

        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            max_tokens=2048,
            tools=openai_tools,
            tool_choice="auto",
            messages=messages,
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # ── 도구 호출 없음 → 최종 응답 ────────────────────────────────────────
        if finish_reason == "stop" or not message.tool_calls:
            raw_text = message.content or ""
            result = safe_parse_json(raw_text)
            if result:
                logger.info("[Agent] 검증 완료 (status=%s)", result.get("status"))
                return result
            logger.warning("[Agent] 최종 텍스트 응답 파싱 실패")
            return {"status": "partial", "candidates": [], "warnings": ["응답 파싱 실패"]}

        # ── tool_calls → 도구 실행 → tool 결과 메시지 추가 ───────────────────
        messages.append(message)  # assistant 메시지 (tool_calls 포함)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_input = json.loads(tool_call.function.arguments)

            logger.info("[Agent] 도구 호출: %s / 입력: %s", tool_name, tool_input)
            result_content = dispatch_tool(tool_name, tool_input)
            logger.debug("[Agent] 도구 결과: %s", result_content[:200])

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_content,
            })

    logger.error("[Agent] 최대 반복 횟수 초과")
    return {"status": "error", "candidates": [], "warnings": ["MCP 에이전트 최대 반복 횟수 초과"]}