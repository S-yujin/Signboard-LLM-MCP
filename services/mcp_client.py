"""
services/mcp_client.py

MCP Tool 정의 + 실제 외부 API 호출 디스패처.

세 개의 MCP 도구:
  1. bizno_search_candidates    — 비즈노 API (bizno.net/api/fapi) 상호명 검색
  2. bizno_search_by_phone      — 비즈노 API 전화번호 직접 검색 (Layer 2)
  3. nts_verify_business_status — 국세청 API 사업자 상태 최종 검증
"""
import json
from typing import Any

import httpx

from config import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# MCP Tool 정의
# ──────────────────────────────────────────────────────────────────────────────

MCP_TOOLS: list[dict] = [
    {
        "name": "bizno_search_candidates",
        "description": (
            "비즈노(bizno.net) API로 상호명을 검색하여 사업자 후보 목록을 반환합니다. "
            "사업자등록번호(bno), 법인등록번호(cno), 상호명(company), "
            "사업자상태(bstt), 과세유형(taxtype)을 포함합니다. "
            "반드시 이 도구를 먼저 호출하여 후보를 확보하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색할 상호명 (예: '청담부동산', '홍길동순대국')",
                },
                "area": {
                    "type": "string",
                    "description": "검색 지역 필터 (선택, 예: '부산', '서울', '강남')",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "bizno_search_by_phone",
        "description": (
            "비즈노(bizno.net) API로 전화번호를 검색하여 사업자를 직접 조회합니다. "
            "상호명 변형 fallback으로 0건이 반환될 때, "
            "간판에서 추출한 전화번호(phone)가 있으면 이 도구를 호출하세요. "
            "전화번호가 정확히 등록된 경우 상호명과 무관하게 정확한 사업자를 반환합니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "phone": {
                    "type": "string",
                    "description": "검색할 전화번호 (하이픈 포함/제거 모두 가능, 예: '051-817-9233')",
                },
            },
            "required": ["phone"],
        },
    },
    {
        "name": "nts_verify_business_status",
        "description": (
            "국세청 공공데이터 API로 사업자등록번호 배열의 상태를 최종 검증합니다. "
            "bizno_search_candidates로 후보를 확보한 뒤, "
            "유력 후보의 사업자등록번호(bno, 하이픈 제거)를 전달하여 "
            "계속/휴업/폐업 상태와 과세유형을 공식 확인합니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "registration_numbers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "사업자등록번호 배열 (하이픈 없이 10자리, 예: ['1234567890'])",
                },
            },
            "required": ["registration_numbers"],
        },
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# 디스패처
# ──────────────────────────────────────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "bizno_search_candidates":
        result = _bizno_search_candidates(**tool_input)
    elif tool_name == "bizno_search_by_phone":
        result = _bizno_search_by_phone(**tool_input)
    elif tool_name == "nts_verify_business_status":
        result = _nts_verify_business_status(**tool_input)
    else:
        result = {"error": f"알 수 없는 도구: {tool_name}"}
    return json.dumps(result, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# Tool 1: 비즈노 상호명 검색
# API: GET https://bizno.net/api/fapi
# 파라미터: key, q(검색어), gb(1:사업자번호/2:법인번호/3:상호명), type=json, status=Y
# ──────────────────────────────────────────────────────────────────────────────

def _bizno_search_candidates(query: str, area: str | None = None) -> dict:
    """
    비즈노 API로 상호명 검색하여 사업자 후보 목록을 반환합니다.

    Fallback 전략 (Python 레벨 이중 방어):
        에이전트가 단일 쿼리만 시도하다 멈춰도 mcp_client 자체에서
        generate_query_variants()의 변형 쿼리를 순서대로 자동 재시도합니다.
        에이전트 레벨 fallback + Python 레벨 fallback의 이중 구조로 검색 성공률 향상.

    응답 필드:
        company   : 상호명
        bno       : 사업자등록번호 (하이픈 포함, 예: 123-45-67890)
        bstt      : 사업자상태 (계속사업자 / 휴업자 / 폐업자)
        taxtype   : 과세유형
    """
    logger.info("[MCP/비즈노] 검색어: %s / 지역: %s", query, area)

    if not settings.BIZNO_API_KEY:
        logger.debug("[MCP/비즈노] API 키 없음 — mock 반환")
        return _bizno_mock(query)

    # fallback 쿼리 목록 생성
    from services.verifier import generate_query_variants
    queries_to_try = generate_query_variants(query)

    last_error: str | None = None
    for attempt_query in queries_to_try:
        result = _bizno_single_request(attempt_query, area)

        if "error" in result:
            last_error = result["error"]
            continue

        candidates = result.get("candidates", [])
        if candidates:
            if attempt_query != query:
                logger.info("[MCP/비즈노] fallback 성공: '%s' → '%s' (%d건)",
                            query, attempt_query, len(candidates))
                result["fallback_query"] = attempt_query
            return result

        logger.info("[MCP/비즈노] 0건 — 다음 쿼리 시도: '%s'", attempt_query)

    logger.warning("[MCP/비즈노] 모든 fallback 쿼리 소진 — 후보 없음 (원본: '%s')", query)
    if last_error:
        return {"error": last_error, "candidates": [], "source": "bizno_api"}
    return {"candidates": [], "total": 0, "source": "bizno_api", "fallback_exhausted": True}


def _bizno_single_request(query: str, area: str | None = None) -> dict:
    """비즈노 API 단건 요청."""
    params = {
        "key":     settings.BIZNO_API_KEY,
        "q":       query,
        "gb":      "3",
        "type":    "json",
        "status":  "Y",
        "pagecnt": "5",
    }
    if area:
        params["area"] = area

    try:
        resp = httpx.get(settings.BIZNO_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, dict):
            logger.warning("[MCP/비즈노] 응답이 dict가 아님: %s", type(data))
            return {"error": "잘못된 응답 형식", "candidates": []}

        if data.get("resultCode") != 0:
            logger.warning("[MCP/비즈노] 결과코드 오류: %s", data.get("resultMsg"))
            return {"error": data.get("resultMsg", "알 수 없는 오류"), "candidates": []}

        raw_items = data.get("items") or []
        candidates = [_normalize_bizno_item(item) for item in raw_items if isinstance(item, dict)]

        logger.info("[MCP/비즈노] 후보 %d건 / 전체 %s건", len(candidates), data.get("totalCount"))
        return {"candidates": candidates, "total": data.get("totalCount", len(candidates)), "source": "bizno_api"}

    except httpx.HTTPError as e:
        logger.error("[MCP/비즈노] API 오류: %s", e)
        return {"error": str(e), "candidates": [], "source": "bizno_api"}


# ──────────────────────────────────────────────────────────────────────────────
# Tool 2: 비즈노 전화번호 검색 (Layer 2 — 상호명 fallback 소진 시 사용)
# gb=4: 전화번호 검색 모드
# ──────────────────────────────────────────────────────────────────────────────

def _bizno_search_by_phone(phone: str) -> dict:
    """
    비즈노 API로 전화번호를 직접 검색합니다.

    사용 시점:
        - bizno_search_candidates fallback 쿼리가 모두 소진되어 0건인 경우
        - 간판 이미지에서 전화번호가 추출된 경우

    전처리:
        - 하이픈, 공백 제거 후 검색 (API는 숫자만 받음)
        - 지역번호 0 제거 후 재시도 (051-xxx → 51-xxx 형태 일부 API 허용)

    응답:
        - 성공: candidates 포함 (정규화 포맷 동일)
        - 실패: candidates: [], phone_not_found: True
    """
    logger.info("[MCP/비즈노/전화] 검색: %s", phone)

    if not settings.BIZNO_API_KEY:
        logger.debug("[MCP/비즈노/전화] API 키 없음 — mock 반환")
        return _bizno_phone_mock(phone)

    # 전화번호 정규화: 숫자만 추출
    digits = "".join(c for c in phone if c.isdigit())
    # 하이픈 포함 포맷도 시도 (일부 API 구현이 하이픈 포함을 기대)
    formatted = _format_phone(digits)

    queries_to_try = [formatted, digits]
    # 지역번호 0 제거 변형 (051-xxx → 51-xxx)
    if digits.startswith("0"):
        alt = digits[1:]
        queries_to_try.append(_format_phone(alt))
        queries_to_try.append(alt)

    for q in queries_to_try:
        result = _bizno_phone_single_request(q)
        candidates = result.get("candidates", [])
        if candidates:
            logger.info("[MCP/비즈노/전화] 히트: '%s' → %d건", q, len(candidates))
            result["phone_query"] = q
            return result
        if "error" in result:
            logger.warning("[MCP/비즈노/전화] 오류: %s", result["error"])

    logger.warning("[MCP/비즈노/전화] 전화번호 검색 0건: %s", phone)
    return {"candidates": [], "total": 0, "source": "bizno_api", "phone_not_found": True}


def _bizno_phone_single_request(phone_query: str) -> dict:
    """비즈노 전화번호 단건 요청 (gb=4)."""
    params = {
        "key":     settings.BIZNO_API_KEY,
        "q":       phone_query,
        "gb":      "4",          # 전화번호 검색 모드
        "type":    "json",
        "status":  "Y",
        "pagecnt": "5",
    }
    try:
        resp = httpx.get(settings.BIZNO_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, dict):
            return {"error": "잘못된 응답 형식", "candidates": []}

        if data.get("resultCode") != 0:
            return {"error": data.get("resultMsg", "알 수 없는 오류"), "candidates": []}

        raw_items = data.get("items") or []
        candidates = [_normalize_bizno_item(item) for item in raw_items if isinstance(item, dict)]
        return {"candidates": candidates, "total": data.get("totalCount", len(candidates)), "source": "bizno_api"}

    except httpx.HTTPError as e:
        logger.error("[MCP/비즈노/전화] API 오류: %s", e)
        return {"error": str(e), "candidates": []}


def _format_phone(digits: str) -> str:
    """숫자열을 하이픈 포함 전화번호 포맷으로 변환. 예: '0518179233' → '051-817-9233'"""
    if len(digits) == 10 and digits.startswith("0"):
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    if len(digits) == 11 and digits.startswith("01"):
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
    return digits  # 변환 불가 → 원본 반환


def _bizno_phone_mock(phone: str) -> dict:
    return {
        "candidates": [
            {
                "registration_number": "9876543210",
                "business_name":       f"(전화검색) {phone}",
                "business_status":     "계속사업자",
                "status_code":         "01",
                "tax_type":            "일반과세자",
                "tax_type_cd":         "1",
                "corporation_number":  "",
                "end_date":            "",
                "source":              "mock_phone",
            }
        ],
        "total": 1,
        "source": "mock_phone",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 공통 정규화 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_bizno_item(item: dict) -> dict:
    """비즈노 응답 항목 → 내부 통일 포맷"""
    raw_bno = str(item.get("bno") or "").replace("-", "")
    return {
        "registration_number": raw_bno,
        "business_name":       item.get("company") or "",
        "business_status":     item.get("bstt") or "unknown",
        "status_code":         item.get("bsttcd") or "",
        "tax_type":            item.get("taxtype") or "",
        "tax_type_cd":         item.get("TaxTypeCd") or "",
        "corporation_number":  item.get("cno") or "",
        "end_date":            item.get("EndDt") or "",
        "source":              "bizno_api",
    }


def _bizno_mock(query: str) -> dict:
    return {
        "candidates": [
            {
                "registration_number": "1234567890",
                "business_name":       query,
                "business_status":     "계속사업자",
                "status_code":         "01",
                "tax_type":            "일반과세자",
                "tax_type_cd":         "1",
                "corporation_number":  "",
                "end_date":            "",
                "source":              "mock",
            }
        ],
        "total": 1,
        "source": "mock",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tool 3: 국세청 최종 상태 검증
# ──────────────────────────────────────────────────────────────────────────────

def _nts_verify_business_status(registration_numbers: list[str]) -> dict:
    logger.info("[MCP/국세청] 검증 번호: %s", registration_numbers)

    if not settings.NTS_SERVICE_KEY:
        logger.debug("[MCP/국세청] 서비스키 없음 — mock 반환")
        return _nts_mock(registration_numbers)

    try:
        resp = httpx.post(
            f"{settings.NTS_API_BASE}/status",
            params={"serviceKey": settings.NTS_SERVICE_KEY},
            json={"b_no": registration_numbers},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("data", [])
        logger.info("[MCP/국세청] 검증 완료 %d건", len(results))
        return {"verified": True, "source": "nts_api", "results": results}

    except httpx.HTTPError as e:
        logger.error("[MCP/국세청] API 오류: %s", e)
        return {"verified": False, "error": str(e), "results": []}


def _nts_mock(registration_numbers: list[str]) -> dict:
    return {
        "verified": False,
        "source":   "mock",
        "results": [
            {
                "b_no":        num,
                "b_stt":       "계속사업자" if i == 0 else "휴업자",
                "b_stt_cd":    "01"         if i == 0 else "02",
                "tax_type":    "일반과세자",
                "tax_type_cd": "1",
                "end_dt":      "",
                "utcc_yn":     "N",
            }
            for i, num in enumerate(registration_numbers)
        ],
    }