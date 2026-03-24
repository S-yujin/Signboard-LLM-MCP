"""
services/mcp_client.py
MCP Tool 정의 + 실제 외부 API 호출 디스패처.

두 개의 MCP 도구를 정의합니다:
  1. lookup_business_candidates  — 상호명/전화/업종으로 사업자 후보 조회
  2. verify_business_status      — 국세청 API로 사업자등록번호 상태 검증
"""
import json
from typing import Any

import httpx

from config import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Anthropic tool_use 스펙에 맞는 MCP 도구 정의
# ──────────────────────────────────────────────────────────────────────────────

MCP_TOOLS: list[dict] = [
    {
        "name": "lookup_business_candidates",
        "description": (
            "상호명·전화번호·업종 키워드로 사업자 후보를 조회합니다. "
            "내부 DB 또는 외부 검색 API에서 사업자등록번호 후보 목록을 반환합니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "business_name": {"type": "string", "description": "상호명 (필수)"},
                "phone": {"type": "string", "description": "전화번호 (선택)"},
                "industry": {"type": "string", "description": "업종 키워드 (선택)"},
                "address": {"type": "string", "description": "주소 조각 (선택)"},
            },
            "required": ["business_name"],
        },
    },
    {
        "name": "verify_business_status",
        "description": (
            "사업자등록번호 배열을 국세청 API에 전달하여 "
            "각 번호의 과세유형·상태·대표자명을 검증합니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "registration_numbers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "사업자등록번호 배열 (하이픈 없이 10자리, 예: ['1234567890'])",
                }
            },
            "required": ["registration_numbers"],
        },
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# 도구 실행 함수
# ──────────────────────────────────────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_input: dict) -> Any:
    """
    tool_name에 따라 적절한 실행 함수를 호출하고 결과를 반환합니다.
    결과는 문자열(JSON)로 직렬화되어 Anthropic tool_result content로 사용됩니다.
    """
    if tool_name == "lookup_business_candidates":
        result = _lookup_business_candidates(**tool_input)
    elif tool_name == "verify_business_status":
        result = _verify_business_status(**tool_input)
    else:
        result = {"error": f"알 수 없는 도구: {tool_name}"}

    return json.dumps(result, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# 내부 구현: 후보 조회
# ──────────────────────────────────────────────────────────────────────────────

def _lookup_business_candidates(
    business_name: str,
    phone: str | None = None,
    industry: str | None = None,
    address: str | None = None,
) -> dict:
    """
    사업자 후보 목록을 조회합니다.
    실제 API 키가 설정된 경우 외부 API를 호출하고,
    그렇지 않으면 mock 데이터를 반환합니다.
    """
    logger.info("[MCP] 후보 조회 — 상호명: %s / 전화: %s", business_name, phone)

    if settings.BUSINESS_SEARCH_API_KEY:
        return _call_search_api(business_name, phone, industry, address)

    # ── Mock 데이터 ─────────────────────────────────────────────────────────
    logger.debug("[MCP] API 키 없음 — mock 데이터 반환")
    return {
        "candidates": [
            {
                "registration_number": "1234567890",
                "business_name": business_name,
                "representative": "홍길동",
                "address": address or "서울특별시 강남구 테헤란로 123",
                "industry": industry or "한식음식점업",
                "phone": phone or "02-1234-5678",
                "source": "mock_db",
            },
            {
                "registration_number": "9876543210",
                "business_name": f"{business_name} 2호점",
                "representative": "김철수",
                "address": "서울특별시 서초구 반포대로 456",
                "industry": industry or "한식음식점업",
                "phone": None,
                "source": "mock_db",
            },
        ],
        "total": 2,
    }


def _call_search_api(business_name, phone, industry, address) -> dict:
    """실제 사업자 검색 API 호출 (BUSINESS_SEARCH_API_URL 환경변수 기반)"""
    params = {"name": business_name}
    if phone:
        params["phone"] = phone
    if industry:
        params["industry"] = industry
    if address:
        params["address"] = address

    try:
        resp = httpx.get(
            f"{settings.BUSINESS_SEARCH_API_URL}/businesses",
            params=params,
            headers={"Authorization": f"Bearer {settings.BUSINESS_SEARCH_API_KEY}"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        logger.error("[MCP] 검색 API 오류: %s", e)
        return {"error": str(e), "candidates": []}


# ──────────────────────────────────────────────────────────────────────────────
# 내부 구현: 국세청 상태 검증
# ──────────────────────────────────────────────────────────────────────────────

def _verify_business_status(registration_numbers: list[str]) -> dict:
    """
    국세청 공공데이터 API로 사업자 상태를 검증합니다.
    NTS_SERVICE_KEY가 없으면 mock 데이터를 반환합니다.
    """
    logger.info("[MCP] 상태 검증 — 번호: %s", registration_numbers)

    if settings.NTS_SERVICE_KEY:
        return _call_nts_api(registration_numbers)

    # ── Mock ────────────────────────────────────────────────────────────────
    logger.debug("[MCP] NTS 키 없음 — mock 상태 반환")
    mock_results = []
    for num in registration_numbers:
        is_first = (num == registration_numbers[0])
        mock_results.append({
            "b_no": num,
            "b_stt": "계속사업자" if is_first else "휴업자",
            "b_stt_cd": "01" if is_first else "02",
            "tax_type": "일반과세자",
            "tax_type_cd": "1",
            "end_dt": "",
            "utcc_yn": "N",
        })
    return {"verified": False, "source": "mock", "results": mock_results}


def _call_nts_api(registration_numbers: list[str]) -> dict:
    """국세청 사업자등록 상태조회 API 실제 호출"""
    url = f"{settings.NTS_API_BASE}/status"
    params = {"serviceKey": settings.NTS_SERVICE_KEY}
    body = {"b_no": registration_numbers}

    try:
        resp = httpx.post(url, params=params, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "verified": True,
            "source": "nts_api",
            "results": data.get("data", []),
        }
    except httpx.HTTPError as e:
        logger.error("[MCP] 국세청 API 오류: %s", e)
        return {"verified": False, "error": str(e), "results": []}