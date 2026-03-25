"""
services/mcp_client.py

MCP Tool 정의 + 실제 외부 API 호출 디스패처.

두 개의 MCP 도구:
  1. bizno_search_candidates    — 비즈노 API로 상호명/전화번호 기반 후보 조회
  2. nts_verify_business_status — 국세청 API로 사업자등록번호 상태 최종 검증
"""
import json
from typing import Any

import httpx

from config import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# MCP Tool 정의 (Gemini FunctionDeclaration 스펙)
# ──────────────────────────────────────────────────────────────────────────────

MCP_TOOLS: list[dict] = [
    {
        "name": "bizno_search_candidates",
        "description": (
            "비즈노(bizno.net) API를 통해 상호명 또는 전화번호로 사업자 후보를 검색합니다. "
            "사업자등록번호, 상호명, 사업자상태, 과세유형을 반환합니다. "
            "반드시 이 도구를 먼저 호출하여 후보 목록을 확보하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색어 — 상호명 또는 전화번호 (예: '홍길동순대국', '02-1234-5678')",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "nts_verify_business_status",
        "description": (
            "국세청 공공데이터 API를 통해 사업자등록번호 배열의 상태를 최종 검증합니다. "
            "bizno_search_candidates로 후보를 확보한 후, 유력 후보의 사업자등록번호를 "
            "이 도구에 전달하여 과세유형과 계속/휴업/폐업 상태를 확인하세요."
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
    """tool_name에 따라 실행 함수를 호출하고 JSON 문자열로 반환합니다."""
    if tool_name == "bizno_search_candidates":
        result = _bizno_search_candidates(**tool_input)
    elif tool_name == "nts_verify_business_status":
        result = _nts_verify_business_status(**tool_input)
    else:
        result = {"error": f"알 수 없는 도구: {tool_name}"}

    return json.dumps(result, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# Tool 1: 비즈노 후보 검색
# ──────────────────────────────────────────────────────────────────────────────

def _bizno_search_candidates(query: str) -> dict:
    """
    비즈노 API로 상호명/전화번호 검색하여 사업자 후보 목록을 반환합니다.

    무료 API 응답 필드: BIZNO | CMPNM_NM | BSN_STATE_NM | INDUTY_NM
    유료 API 추가 필드: RPRSV_NM | ADRES | TEL_NO | FXNO | LCTN_LOTNO_ADRES 등

    API 키가 없으면 mock 데이터를 반환합니다.
    """
    logger.info("[MCP/비즈노] 검색어: %s", query)

    if not settings.BIZNO_API_KEY:
        logger.debug("[MCP/비즈노] API 키 없음 — mock 반환")
        return _bizno_mock(query)

    try:
        resp = httpx.get(
            settings.BIZNO_API_URL,
            params={
                "query": query,
                "key":   settings.BIZNO_API_KEY,
                "type":  "json",
                "perPage": 5,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        # 비즈노 응답 → 통일 포맷으로 변환
        raw_items = data.get("items", data.get("data", []))
        candidates = [_normalize_bizno_item(item) for item in raw_items]

        logger.info("[MCP/비즈노] 후보 %d건 반환", len(candidates))
        return {"candidates": candidates, "total": len(candidates), "source": "bizno_api"}

    except httpx.HTTPError as e:
        logger.error("[MCP/비즈노] API 오류: %s", e)
        return {"error": str(e), "candidates": [], "source": "bizno_api"}


def _normalize_bizno_item(item: dict) -> dict:
    """비즈노 응답 항목을 내부 통일 포맷으로 변환합니다."""
    return {
        "registration_number": str(item.get("BIZNO", item.get("bizno", ""))).replace("-", ""),
        "business_name":       item.get("CMPNM_NM", item.get("cmpnm_nm", "")),
        "business_status":     item.get("BSN_STATE_NM", item.get("bsn_state_nm", "unknown")),
        "tax_type":            item.get("INDUTY_NM", item.get("induty_nm", "")),
        "representative":      item.get("RPRSV_NM", item.get("rprsv_nm")),
        "address":             item.get("ADRES", item.get("adres")),
        "phone":               item.get("TEL_NO", item.get("tel_no")),
        "source": "bizno_api",
    }


def _bizno_mock(query: str) -> dict:
    """비즈노 API 키 없을 때 반환하는 mock 데이터"""
    return {
        "candidates": [
            {
                "registration_number": "1234567890",
                "business_name": query,
                "business_status": "계속사업자",
                "tax_type": "일반과세자",
                "representative": "홍길동",
                "address": "서울특별시 강남구 테헤란로 123",
                "phone": "02-1234-5678",
                "source": "mock",
            },
            {
                "registration_number": "9876543210",
                "business_name": f"{query} 2호점",
                "business_status": "휴업자",
                "tax_type": "간이과세자",
                "representative": "김철수",
                "address": "서울특별시 서초구 반포대로 456",
                "phone": None,
                "source": "mock",
            },
        ],
        "total": 2,
        "source": "mock",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tool 2: 국세청 최종 상태 검증
# ──────────────────────────────────────────────────────────────────────────────

def _nts_verify_business_status(registration_numbers: list[str]) -> dict:
    """
    국세청 공공데이터 API로 사업자등록번호 배열의 상태를 검증합니다.
    NTS_SERVICE_KEY가 없으면 mock 데이터를 반환합니다.
    """
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
    """국세청 API 키 없을 때 반환하는 mock 데이터"""
    results = []
    for i, num in enumerate(registration_numbers):
        results.append({
            "b_no":      num,
            "b_stt":     "계속사업자" if i == 0 else "휴업자",
            "b_stt_cd":  "01"         if i == 0 else "02",
            "tax_type":  "일반과세자",
            "tax_type_cd": "1",
            "end_dt":    "",
            "utcc_yn":   "N",
        })
    return {"verified": False, "source": "mock", "results": results}