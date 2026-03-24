"""
schemas/output_schema.py
파이프라인 최종 출력 및 MCP 중간 결과의 Pydantic 스키마
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from schemas.extraction_schema import SignboardExtraction


# ── 열거형 ────────────────────────────────────────────────────────────────────

class PipelineStatus(str, Enum):
    VERIFIED = "verified"       # 사업자등록번호 + 상태 모두 확인
    PARTIAL  = "partial"        # 후보는 있으나 상태 검증 불완전
    NOT_FOUND = "not_found"     # 상호명 인식 실패 또는 후보 없음
    ERROR    = "error"          # 파이프라인 오류


class BusinessStatus(str, Enum):
    ACTIVE   = "계속사업자"
    SUSPENDED = "휴업자"
    CLOSED   = "폐업자"
    UNKNOWN  = "unknown"


# ── 후보 사업자 ────────────────────────────────────────────────────────────────

class BusinessCandidate(BaseModel):
    """MCP 조회 + 국세청 검증을 거친 사업자 후보 한 건"""
    registration_number: str = Field(..., description="사업자등록번호 (10자리, 하이픈 없음)")
    business_name: str = Field(..., description="상호명")
    representative: Optional[str] = Field(None, description="대표자명")
    address: Optional[str] = Field(None, description="사업장 주소")
    industry: Optional[str] = Field(None, description="업종/업태")
    phone: Optional[str] = Field(None, description="전화번호")

    # 국세청 검증 결과
    business_status: BusinessStatus = Field(BusinessStatus.UNKNOWN, description="사업자 상태")
    tax_type: Optional[str] = Field(None, description="과세유형 (일반/간이/면세)")
    status_verified: bool = Field(False, description="국세청 API 검증 여부")

    # 신뢰도
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="간판 정보와의 일치 신뢰도")
    source: str = Field("unknown", description="조회 출처 (mock_db / search_api 등)")

    def is_active(self) -> bool:
        return self.business_status == BusinessStatus.ACTIVE


# ── NTS 원시 응답 ──────────────────────────────────────────────────────────────

class NtsStatusItem(BaseModel):
    """국세청 API 응답 항목 원형"""
    b_no: str
    b_stt: str = ""
    b_stt_cd: str = ""
    tax_type: str = ""
    tax_type_cd: str = ""
    end_dt: str = ""
    utcc_yn: str = "N"


# ── 최종 파이프라인 출력 ───────────────────────────────────────────────────────

class PipelineResult(BaseModel):
    """파이프라인 전체 최종 출력 구조체"""
    pipeline_version: str = Field("1.0.0")
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    image_source: str = Field(..., description="분석한 이미지 경로 또는 URL")

    status: PipelineStatus = Field(..., description="파이프라인 처리 결과 상태")
    source_signboard: Optional[SignboardExtraction] = Field(None, description="LLM 간판 추출 결과")
    candidates: List[BusinessCandidate] = Field(default_factory=list)
    best_match: Optional[BusinessCandidate] = Field(None, description="최고 신뢰도 후보")
    warnings: List[str] = Field(default_factory=list, description="처리 중 발생한 경고 메시지")

    def model_dump_json_pretty(self) -> str:
        import json
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )