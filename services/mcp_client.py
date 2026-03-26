"""
services/mcp_client.py

MCP Tool 정의 + 실제 외부 API 호출 디스패처.

두 개의 MCP 도구:
  1. bizno_search_candidates    — 비즈노 API (bizno.net/api/fapi) 상호명 검색
  2. nts_verify_business_status — 국세청 API 사업자 상태 최종 검증
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
    elif tool_name == "nts_verify_business_status":
        result = _nts_verify_business_status(**tool_input)
    else:
        result = {"error": f"알 수 없는 도구: {tool_name}"}
    return json.dumps(result, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# Tool 1: 비즈노 후보 검색
# API: GET https://bizno.net/api/fapi
# 파라미터: key, q(검색어), gb(1:사업자번호/2:법인번호/3:상호명), type=json, status=Y
# ──────────────────────────────────────────────────────────────────────────────

def _bizno_search_candidates(query: str, area: str | None = None) -> dict:
    """
    비즈노 API로 상호명 검색하여 사업자 후보 목록을 반환합니다.

    응답 필드:
        company   : 상호명
        bno       : 사업자등록번호 (하이픈 포함, 예: 123-45-67890)
        cno       : 법인등록번호
        bstt      : 사업자상태 (계속사업자 / 휴업자 / 폐업자)
        bsttcd    : 상태코드
        taxtype   : 과세유형
        TaxTypeCd : 과세유형코드
        EndDt     : 폐업일
    """
    logger.info("[MCP/비즈노] 검색어: %s / 지역: %s", query, area)

    if not settings.BIZNO_API_KEY:
        logger.debug("[MCP/비즈노] API 키 없음 — mock 반환")
        return _bizno_mock(query)

    params = {
        "key":     settings.BIZNO_API_KEY,
        "q":       query,
        "gb":      "3",          # 3: 상호명 검색
        "type":    "json",
        "status":  "Y",          # 국세청 실시간 상태 조회 포함
        "pagecnt": "5",          # 최대 5건
    }
    if area:
        params["area"] = area

    try:
        resp = httpx.get(
            settings.BIZNO_API_URL,
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        # 응답 자체가 None이거나 dict가 아닌 경우 방어
        if not isinstance(data, dict):
            logger.warning("[MCP/비즈노] 응답이 dict가 아님: %s", type(data))
            return {"error": "잘못된 응답 형식", "candidates": []}

        if data.get("resultCode") != 0:
            logger.warning("[MCP/비즈노] 결과코드 오류: %s", data.get("resultMsg"))
            return {"error": data.get("resultMsg", "알 수 없는 오류"), "candidates": []}

        raw_items = data.get("items") or []   # None → []
        candidates = [_normalize_bizno_item(item) for item in raw_items if isinstance(item, dict)]

        logger.info("[MCP/비즈노] 후보 %d건 / 전체 %s건", len(candidates), data.get("totalCount"))
        return {
            "candidates":   candidates,
            "total":        data.get("totalCount", len(candidates)),
            "source":       "bizno_api",
        }

    except httpx.HTTPError as e:
        logger.error("[MCP/비즈노] API 오류: %s", e)
        return {"error": str(e), "candidates": [], "source": "bizno_api"}


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
# Tool 2: 국세청 최종 상태 검증
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