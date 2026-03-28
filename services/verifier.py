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

# ──────────────────────────────────────────────────────────────────────────────
# 영문→한글 쿼리 변형 (비즈노 등록명 불일치 대응)
# ──────────────────────────────────────────────────────────────────────────────

_EN_TO_KO: dict[str, str] = {
    "CAFE": "카페", "cafe": "카페",
    "BAKERY": "베이커리", "bakery": "베이커리",
    "HAIR": "헤어", "hair": "헤어",
    "BEAUTY": "뷰티", "beauty": "뷰티",
    "CHICKEN": "치킨", "chicken": "치킨",
    "PIZZA": "피자", "pizza": "피자",
    "BURGER": "버거", "burger": "버거",
    "MART": "마트", "mart": "마트",
    "PHARMACY": "약국", "pharmacy": "약국",
    "HOSPITAL": "병원", "hospital": "병원",
    "SALON": "살롱", "salon": "살롱",
    "STUDIO": "스튜디오", "studio": "스튜디오",
    "GYM": "짐", "gym": "짐",
}


# 행정구역 키워드 (앞에서부터 매칭 — 더 구체적인 것 우선)
_REGION_KEYWORDS: list[str] = [
    "해운대", "서면", "남포", "동래", "사직", "광안", "수영", "센텀",
    "강남", "강북", "홍대", "신촌", "이태원", "명동", "종로", "여의도",
    "판교", "분당", "일산", "수원", "인천", "대구", "광주", "대전",
    "울산", "부산", "서울", "경기", "충청", "전라", "경상", "제주",
]


def extract_area_from_address(address: str | None) -> str | None:
    """
    주소 또는 지점명에서 비즈노 area 파라미터로 쓸 지역명을 추출합니다.
    예) "서면롯데점" → "서면", "부산시 해운대구" → "해운대"
    """
    if not address:
        return None
    for kw in _REGION_KEYWORDS:
        if kw in address:
            return kw
    return None


def generate_query_variants(name: str) -> list[str]:
    """
    상호명 검색 실패에 대비한 쿼리 후보 리스트를 생성합니다.

    변형 단계 (순서대로 시도, 최대 7단계):
      1. 원본 그대로                          "CAFE 051 서면롯데점"
      2. 지점명 제거                          "CAFE 051"
      3. 영문 → 한글 음역                     "카페 051"
      4. 공백 제거 (한글)                     "카페051"
      5. 공백 제거 (원본)                     "CAFE051"
      6. 숫자 제거한 핵심 브랜드명 (2자 이상)  "카페"
      7. 첫 단어(토큰)만                      "CAFE"

    Bizno는 부분일치 검색을 지원하므로 짧을수록 더 많은 후보를 반환합니다.
    """
    import re

    variants: list[str] = [name]

    # ── Step 2: 지점명 제거 ─────────────────────────────────────────────────
    branch_removed = re.sub(
        r"\s*\S*(점|호점|지점|본점|직영점|가맹점|분점|센터|지사)$",
        "", name
    ).strip()
    if branch_removed and branch_removed != name:
        variants.append(branch_removed)
    else:
        branch_removed = name

    # ── Step 3: 영문 → 한글 음역 ────────────────────────────────────────────
    translated = branch_removed
    for en, ko in _EN_TO_KO.items():
        translated = translated.replace(en, ko)
    if translated != branch_removed:
        variants.append(translated)

    # ── Step 4: 공백 제거 (한글) ────────────────────────────────────────────
    translated_nospace = translated.replace(" ", "")
    if translated_nospace not in variants:
        variants.append(translated_nospace)

    # ── Step 5: 공백 제거 (원본) ────────────────────────────────────────────
    nospace = branch_removed.replace(" ", "")
    if nospace not in variants:
        variants.append(nospace)

    # ── Step 6: 숫자 제거한 브랜드명 (2자 이상이어야 유효) ─────────────────
    no_digits = re.sub(r"\d+", "", translated_nospace).strip()
    if len(no_digits) >= 2 and no_digits not in variants:
        variants.append(no_digits)

    # ── Step 7: 첫 토큰만 ───────────────────────────────────────────────────
    tokens = branch_removed.split()
    if tokens and tokens[0] not in variants and len(tokens[0]) >= 2:
        variants.append(tokens[0])

    # 중복 제거, 빈 문자열 제외, 순서 유지
    seen: set[str] = set()
    result: list[str] = []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            result.append(v)
    return result

_AGENT_SYSTEM_PROMPT = """당신은 사업자 정보 검증 전문 에이전트입니다.
간판 분석 결과를 바탕으로 반드시 아래 두 단계를 순서대로 수행하세요.

[Step 1] bizno_search_candidates 호출 — Fallback 검색 전략
  후보가 확보될 때까지 아래 순서로 쿼리를 변형하며 최대 5회 재시도합니다.
  결과가 0건이면 즉시 다음 단계로 넘어가고, 1건 이상이면 Step 2로 진행합니다.

  시도 순서 (variants_hint에 제공된 리스트를 순서대로 사용):
    1단계) 원본 상호명 그대로
    2단계) 지점명 제거  ("서면롯데점", "1호점", "본점" 등 제거)
    3단계) 영문 → 한글 음역  ("CAFE" → "카페", "CHICKEN" → "치킨")
    4단계) 공백 제거  ("카페 051" → "카페051")
    5단계) 핵심 브랜드명만  (숫자·지점명 제거 후 핵심어)

  ※ area 파라미터는 사용하지 마세요. 지역 필터링은 후처리에서 처리합니다.

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
  ※ 실제 점수는 Python integrator에서 재계산되므로 근사값을 넣어도 됩니다.

status 기준:
  verified  — 비즈노 후보 확보 + 국세청 상태 확인 완료
  partial   — 비즈노 후보는 있으나 국세청 검증 불완전
  not_found — 5단계 fallback을 모두 시도했음에도 후보를 찾지 못함
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

    # 쿼리 변형 후보 미리 계산 (에이전트가 재검색 시 참고)
    query_variants = generate_query_variants(extraction.business_name or "")
    variants_hint = ""
    if len(query_variants) > 1:
        variants_hint = (
            f"\n\n※ 검색 실패 시 아래 순서로 재시도하세요: {query_variants}"
        )

    # 주소에서 지역명 추출 (integrator의 confidence 재계산에서만 활용, area 필터는 사용 안 함)
    # 비즈노 area 파라미터는 결과를 과도하게 제한하므로 제거

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
                + variants_hint
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

        # candidates가 비어있으면 (안전 필터 등으로 차단) 조기 종료
        if not response.candidates:
            logger.warning("[Agent] 응답 candidates 없음")
            return {"status": "partial", "candidates": [], "warnings": ["Gemini 응답 없음"]}

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            logger.warning("[Agent] 응답 content/parts 없음")
            return {"status": "partial", "candidates": [], "warnings": ["Gemini 응답 파싱 불가"]}

        contents.append(candidate.content)

        fn_calls = [
            part.function_call
            for part in candidate.content.parts
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