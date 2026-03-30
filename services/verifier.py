"""
services/verifier.py
[Step 2 & 3 — MCP 에이전트]

Gemini에게 두 개의 MCP 도구를 제공하고 에이전트 루프를 실행합니다.

변경 이력 (논문 수정안 반영):
  - run_verification_agent(): poi_context 파라미터 추가
    → app.py에서 GPS 반경 POI 목록을 전달하면 에이전트 프롬프트에 주입
    → 에이전트가 Bizno 검색 전에 카카오맵 POI 명칭을 우선 참고
  - bizno_search_by_phone 도구 제거: 전화번호 추출은 유지하되 검색에는 사용하지 않음

실행 순서 (에이전트가 자율 결정):
  1. bizno_search_candidates    — 비즈노로 상호명 검색
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

_REGION_KEYWORDS: list[str] = [
    "해운대", "서면", "남포", "동래", "사직", "광안", "수영", "센텀",
    "강남", "강북", "홍대", "신촌", "이태원", "명동", "종로", "여의도",
    "판교", "분당", "일산", "수원", "인천", "대구", "광주", "대전",
    "울산", "부산", "서울", "경기", "충청", "전라", "경상", "제주",
]


def extract_area_from_address(address: str | None) -> str | None:
    """주소 또는 지점명에서 비즈노 area 파라미터로 쓸 지역명을 추출합니다."""
    if not address:
        return None
    for kw in _REGION_KEYWORDS:
        if kw in address:
            return kw
    return None


def generate_query_variants(name: str) -> list[str]:
    """
    상호명 검색 실패에 대비한 쿼리 후보 리스트를 생성합니다.
    (최대 7단계 변형)
    """
    import re

    variants: list[str] = [name]

    # Step 2: 지점명 제거
    branch_removed = re.sub(
        r"\s*\S*(점|호점|지점|본점|직영점|가맹점|분점|센터|지사)$",
        "", name,
    ).strip()
    if branch_removed and branch_removed != name:
        variants.append(branch_removed)
    else:
        branch_removed = name

    # Step 3: 영문→한글 음역
    translated = branch_removed
    for en, ko in _EN_TO_KO.items():
        translated = translated.replace(en, ko)
    if translated != branch_removed:
        variants.append(translated)

    # Step 4: 공백 제거 (한글)
    translated_nospace = translated.replace(" ", "")
    if translated_nospace not in variants:
        variants.append(translated_nospace)

    # Step 5: 공백 제거 (원본)
    nospace = branch_removed.replace(" ", "")
    if nospace not in variants:
        variants.append(nospace)

    # Step 6: 숫자 제거한 브랜드명 (2자 이상)
    no_digits = re.sub(r"\d+", "", translated_nospace).strip()
    if len(no_digits) >= 2 and no_digits not in variants:
        variants.append(no_digits)

    # Step 7: 첫 토큰만
    tokens = branch_removed.split()
    if tokens and tokens[0] not in variants and len(tokens[0]) >= 2:
        variants.append(tokens[0])

    seen: set[str] = set()
    result: list[str] = []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            result.append(v)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 에이전트 시스템 프롬프트
# ──────────────────────────────────────────────────────────────────────────────

_AGENT_SYSTEM_PROMPT = """당신은 사업자 정보 검증 전문 에이전트입니다.
간판 분석 결과를 바탕으로 반드시 아래 두 단계를 순서대로 수행하세요.

[Step 1] bizno_search_candidates 호출 — Fallback 검색 전략

  ※ GPS_POI_HINT가 제공된 경우:
    - POI 명칭이 추출 상호명보다 정확할 수 있습니다.
    - POI 명칭을 우선 검색 쿼리로 사용하고, 실패 시 아래 순서로 fallback하세요.

  후보가 확보될 때까지 아래 순서로 쿼리를 변형하며 최대 5회 재시도합니다.
  결과가 0건이면 not_found로 처리하고, 1건 이상이면 Step 2로 진행합니다.

  시도 순서 (variants_hint에 제공된 리스트를 순서대로 사용):
    1단계) 원본 상호명 그대로
    2단계) 지점명 제거  ("서면롯데점", "1호점", "본점" 등 제거)
    3단계) 영문 → 한글 음역  ("CAFE" → "카페", "CHICKEN" → "치킨")
    4단계) 공백 제거  ("카페 051" → "카페051")
    5단계) 핵심 브랜드명만  (숫자·지점명 제거 후 핵심어)

  ※ area 파라미터는 사용하지 마세요. 지역 필터링은 후처리에서 처리합니다.

[Step 2] nts_verify_business_status 호출
  - Step 1 결과 중 상호명이 유사한 후보의 bno에서 하이픈을 제거하여 전달합니다.
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


# ──────────────────────────────────────────────────────────────────────────────
# 에이전트 실행
# ──────────────────────────────────────────────────────────────────────────────

def run_verification_agent(
    extraction: SignboardExtraction,
    poi_context: list[str] | None = None,
) -> dict:
    """
    MCP 에이전트를 실행하여 사업자 검증 결과를 반환합니다.

    Args:
        extraction  : LLM 간판 추출 결과
        poi_context : GPS 반경 내 카카오 POI 힌트 문자열 리스트.
                      poi_service.build_poi_context_hints() 결과를 전달합니다.
                      None이면 POI 힌트 없이 진행합니다.
    """
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    # 쿼리 변형 후보 미리 계산
    query_variants = generate_query_variants(extraction.business_name or "")
    variants_hint = ""
    if len(query_variants) > 1:
        variants_hint = f"\n\n※ 검색 실패 시 아래 순서로 재시도하세요: {query_variants}"

    # GPS POI 힌트 (논문 수정안 Step 2-1 결과)
    poi_hint = ""
    if poi_context:
        poi_hint = (
            "\n\n[GPS_POI_HINT] GPS 반경 300m 내 카카오맵 POI (정확도 높음, 우선 참고):\n"
            + "\n".join(f"  - {p}" for p in poi_context)
            + "\n위 POI 명칭 중 간판과 일치하는 것이 있으면 해당 명칭으로 먼저 검색하세요."
        )
        logger.info("[Agent] POI 컨텍스트 %d건 주입", len(poi_context))

    config = types.GenerateContentConfig(
        system_instruction=_AGENT_SYSTEM_PROMPT,
        tools=_build_tools(),
        temperature=0.1,
    )

    contents: list = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=(
                "다음 간판 분석 결과로 사업자를 검증해주세요:\n"
                + json.dumps(extraction.model_dump(), ensure_ascii=False, indent=2)
                + variants_hint
                + poi_hint
            ))],
        )
    ]

    for iteration in range(1, settings.MCP_MAX_ITERATIONS + 1):
        logger.debug("[Agent] 반복 %d / %d", iteration, settings.MCP_MAX_ITERATIONS)

        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=contents,
            config=config,
        )

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