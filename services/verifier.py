"""
services/verifier.py
[Step 2 & 3 — MCP 에이전트]
Gemini function calling으로 사업자 후보 조회 + 상태 검증을 수행하는 에이전트 루프.
google-genai 패키지 사용 (구 google-generativeai 대체)
"""
import json

from google import genai
from google.genai import types

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
  "warnings": []
}"""


def _build_tools() -> list[types.Tool]:
    """MCP_TOOLS → google-genai Tool 형식 변환"""
    declarations = [
        types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=t["input_schema"],
        )
        for t in MCP_TOOLS
    ]
    return [types.Tool(function_declarations=declarations)]


def run_verification_agent(extraction: SignboardExtraction) -> dict:
    """
    LLM이 추출한 간판 정보를 기반으로 Gemini function calling 에이전트 루프를 실행합니다.
    """
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    config = types.GenerateContentConfig(
        system_instruction=_AGENT_SYSTEM_PROMPT,
        tools=_build_tools(),
        temperature=0.1,
    )

    # 대화 히스토리 직접 관리
    contents: list = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(
                text="다음 간판 분석 결과로 사업자를 검증해주세요:\n"
                + json.dumps(extraction.model_dump(), ensure_ascii=False, indent=2)
            )],
        )
    ]

    for iteration in range(1, settings.MCP_MAX_ITERATIONS + 1):
        logger.debug("[Agent] 반복 %d / %d", iteration, settings.MCP_MAX_ITERATIONS)

        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=contents,
            config=config,
        )

        # assistant 응답을 히스토리에 추가
        contents.append(response.candidates[0].content)

        # function_call 파트 수집
        fn_calls = [
            part.function_call
            for part in response.candidates[0].content.parts
            if part.function_call is not None
        ]

        # ── 도구 호출 없음 → 최종 텍스트 응답 ────────────────────────────────
        if not fn_calls:
            text = response.text or ""
            result = safe_parse_json(text)
            if result:
                logger.info("[Agent] 검증 완료 (status=%s)", result.get("status"))
                return result
            logger.warning("[Agent] 최종 응답 파싱 실패: %s", text[:200])
            return {"status": "partial", "candidates": [], "warnings": ["응답 파싱 실패"]}

        # ── 도구 실행 후 function_response로 돌려줌 ──────────────────────────
        fn_response_parts = []
        for fn_call in fn_calls:
            tool_name = fn_call.name
            tool_input = dict(fn_call.args)

            logger.info("[Agent] 도구 호출: %s / 입력: %s", tool_name, tool_input)
            raw_result = dispatch_tool(tool_name, tool_input)
            result_dict = json.loads(raw_result)
            logger.debug("[Agent] 도구 결과: %s", raw_result[:200])

            fn_response_parts.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result_dict},
                )
            )

        contents.append(
            types.Content(role="user", parts=fn_response_parts)
        )

    logger.error("[Agent] 최대 반복 횟수 초과")
    return {"status": "error", "candidates": [], "warnings": ["MCP 에이전트 최대 반복 횟수 초과"]}