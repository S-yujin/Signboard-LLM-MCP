"""
services/verifier.py
[Step 2 & 3 — MCP 에이전트]

Gemini에게 두 개의 MCP 도구를 제공하고 에이전트 루프를 실행합니다.

실행 순서 (에이전트가 자율 결정):
  1. bizno_search_candidates    — 비즈노로 상호명 검색 (지역 필터 활용)
  2. nts_verify_business_status — 국세청으로 유력 후보 최종 검증
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
간판 분석 결과를 바탕으로 반드시 아래 두 단계를 순서대로 수행하세요.

[Step 1] bizno_search_candidates 호출
  - 추출된 상호명(business_name)으로 검색합니다.
  - 주소(address)에서 지역명을 파악할 수 있으면 area 파라미터에 함께 전달하세요.
    예) address가 "부산시 해운대구"이면 area="부산"
  - 결과가 없으면 상호명을 축약하거나 변형해서 재검색합니다.
    예) "청담부동산공인중개사무소" → "청담부동산"

[Step 2] nts_verify_business_status 호출
  - Step 1 결과 중 상호명이 유사한 후보의 bno(사업자등록번호)에서 하이픈을 제거하여 전달합니다.
    예) "123-45-67890" → "1234567890"
  - 국세청 API로 계속/휴업/폐업 상태를 최종 확인합니다.
  - NTS_SERVICE_KEY가 없으면 비즈노의 bstt 값을 그대로 사용하고 status_verified=false로 표시합니다.

두 단계를 모두 완료한 뒤, 아래 JSON 형식으로만 최종 응답하세요.
마크다운 코드블록이나 추가 설명 없이 순수 JSON만 출력하세요.

{
  "status": "verified | partial | not_found",
  "candidates": [
    {
      "registration_number": "하이픈 없는 10자리",
      "business_name": "상호명",
      "representative": null,
      "address": null,
      "industry": null,
      "phone": null,
      "business_status": "계속사업자 | 휴업자 | 폐업자 | unknown",
      "tax_type": "과세유형 또는 null",
      "status_verified": true,
      "confidence_score": 0.0,
      "source": "bizno_api | nts_api | mock"
    }
  ],
  "best_match": { ...최고 신뢰도 후보 또는 null },
  "warnings": []
}

confidence_score 산정 기준 (0.0 ~ 1.0):
  - 상호명 완전 일치: +0.5
  - 상호명 부분 일치: +0.3
  - 지역 일치: +0.2
  - 국세청 계속사업자 확인: +0.3

status 기준:
  verified  — 비즈노 후보 확보 + 국세청 상태 확인 완료
  partial   — 비즈노 후보는 있으나 국세청 검증 불완전
  not_found — 후보를 찾지 못함
"""


def _build_tools() -> list[types.Tool]:
    return [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=t["input_schema"],
            )
            for t in MCP_TOOLS
        ])
    ]


def run_verification_agent(extraction: SignboardExtraction) -> dict:
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    config = types.GenerateContentConfig(
        system_instruction=_AGENT_SYSTEM_PROMPT,
        tools=_build_tools(),
        temperature=0.1,
    )

    contents: list = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=
                "다음 간판 분석 결과로 사업자를 검증해주세요:\n"
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

        contents.append(response.candidates[0].content)

        fn_calls = [
            part.function_call
            for part in response.candidates[0].content.parts
            if part.function_call is not None
        ]

        if not fn_calls:
            text = response.text or ""
            result = safe_parse_json(text)
            if result:
                logger.info("[Agent] 완료 (status=%s)", result.get("status"))
                return result
            logger.warning("[Agent] 응답 파싱 실패: %s", text[:200])
            return {"status": "partial", "candidates": [], "warnings": ["응답 파싱 실패"]}

        fn_response_parts = []
        for fn_call in fn_calls:
            tool_name  = fn_call.name
            tool_input = dict(fn_call.args)

            logger.info("[Agent] 도구 호출: %s / 입력: %s", tool_name, tool_input)
            raw_result  = dispatch_tool(tool_name, tool_input)
            result_dict = json.loads(raw_result)
            logger.debug("[Agent] 결과: %s", raw_result[:300])

            fn_response_parts.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result_dict},
                )
            )

        contents.append(types.Content(role="user", parts=fn_response_parts))

    logger.error("[Agent] 최대 반복 횟수 초과")
    return {"status": "error", "candidates": [], "warnings": ["최대 반복 횟수 초과"]}