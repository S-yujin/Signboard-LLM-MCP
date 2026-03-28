"""
schemas/output_schema.py

파이프라인 최종 출력 스키마 (Pydantic v2).

논문 3.4절 '구조화된 JSON 출력':
    검증된 정보를 바탕으로 사업자명, 전화번호, 업종 등이 포함된
    규격화된 JSON 포맷으로 출력하며 디지털 지도 인덱싱 효율을 높임.

클래스 구조 (integrator.py 실제 사용 기준):
    BusinessStatus      — 사업자 상태 열거형
    PipelineStatus      — 파이프라인 최종 상태 열거형
    BusinessCandidate   — Bizno + 국세청 검증 결과 단일 후보
    PipelineResult      — 파이프라인 전체 출력 (최종 JSON)

설계 원칙:
    1. integrator.py / app.py 가 import하는 클래스명과 완전히 일치
    2. 논문 수식(confFINAL) 결과를 confidence_scores 필드로 포함
    3. 사업자등록번호 자동 정규화 (XXX-XX-XXXXX 포맷)
    4. 디지털 지도 DB 인덱싱을 위한 필드 구성
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────────────────────────────────────
# 열거형
# ──────────────────────────────────────────────────────────────────────────────

class BusinessStatus(str, Enum):
    """
    국세청 사업자 상태.
    integrator.py의 _STATUS_CODE_MAP / _STATUS_NAME_MAP과 1:1 대응.
    """
    ACTIVE    = "계속사업자"
    SUSPENDED = "휴업자"
    CLOSED    = "폐업자"
    UNKNOWN   = "unknown"


class PipelineStatus(str, Enum):
    """
    파이프라인 최종 처리 상태.

    verified  : Bizno 후보 확보 + 국세청 상태 확인 완료
    partial   : Bizno 후보는 있으나 국세청 검증 불완전
    not_found : 5단계 fallback을 모두 시도했음에도 후보 없음
    error     : 파이프라인 내부 오류
    """
    VERIFIED  = "verified"
    PARTIAL   = "partial"
    NOT_FOUND = "not_found"
    ERROR     = "error"


# ──────────────────────────────────────────────────────────────────────────────
# 공통 정규화 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_regno(v: Optional[str]) -> Optional[str]:
    """사업자등록번호를 'XXX-XX-XXXXX' 형식으로 정규화."""
    if not v:
        return None
    digits = re.sub(r"\D", "", str(v))
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
    return v  # 10자리가 아니면 원본 유지


# ──────────────────────────────────────────────────────────────────────────────
# BusinessCandidate — Bizno + 국세청 통합 후보
# integrator.py의 _parse_candidate()가 생성하는 핵심 클래스
# ──────────────────────────────────────────────────────────────────────────────

class BusinessCandidate(BaseModel):
    """
    Bizno.net 검색 + 국세청 상태 검증을 통합한 단일 사업자 후보.

    논문 Step 2·3의 출력 단위이며, integrator.py의 _parse_candidate()가 생성합니다.
    confidence_score는 _compute_confidence()에서 재계산된 conf_MCP 값입니다.

    디지털 지도 인덱싱 필드:
        registration_number — DB 기본키로 활용 가능한 사업자등록번호
        business_status     — 영업 중 여부 필터링 기준
        industry            — 업종 기반 카테고리 인덱싱
    """
    # ── 식별 정보 ──────────────────────────────────────────────────────────────
    registration_number : Optional[str]   = Field(None,  description="사업자등록번호 (XXX-XX-XXXXX)")
    business_name       : str             = Field("",    description="공식 상호명")
    representative      : Optional[str]   = Field(None,  description="대표자명")

    # ── 위치·연락처 ────────────────────────────────────────────────────────────
    address             : Optional[str]   = Field(None,  description="사업장 주소")
    phone               : Optional[str]   = Field(None,  description="대표 전화번호")

    # ── 업종 정보 ──────────────────────────────────────────────────────────────
    industry            : Optional[str]   = Field(None,  description="업종·업태")

    # ── 국세청 검증 상태 ───────────────────────────────────────────────────────
    business_status     : BusinessStatus  = Field(BusinessStatus.UNKNOWN, description="사업자 상태")
    tax_type            : Optional[str]   = Field(None,  description="과세 유형 (일반/간이/면세)")
    status_verified     : bool            = Field(False, description="국세청 API 검증 완료 여부")

    # ── conf_MCP (논문 수식 핵심 입력값) ──────────────────────────────────────
    confidence_score    : float           = Field(0.0, ge=0.0, le=1.0,
                                                  description="MCP 매칭 신뢰도 (conf_MCP, integrator 재계산)")

    # ── 출처 ───────────────────────────────────────────────────────────────────
    source              : str             = Field("unknown",
                                                  description="데이터 출처 (bizno_api / nts_api / mock)")

    @field_validator("registration_number", mode="before")
    @classmethod
    def fmt_regno(cls, v: Optional[str]) -> Optional[str]:
        return _normalize_regno(v)

    @property
    def is_active(self) -> bool:
        """현재 영업 중 여부 (business_status 기준)."""
        return self.business_status == BusinessStatus.ACTIVE

    class Config:
        use_enum_values = False   # Enum 객체 그대로 유지 (integrator 비교 로직에 필요)


# ──────────────────────────────────────────────────────────────────────────────
# PipelineResult — 파이프라인 최종 출력
# ──────────────────────────────────────────────────────────────────────────────

class PipelineResult(BaseModel):
    """
    파이프라인 전체 실행 결과 (논문 3.4절 '구조화된 JSON 출력').

    최종 JSON 구조:
    {
      "pipeline_version": "1.0.0",
      "processed_at": "2026-03-28T...",
      "image_source": "./sample_data/shop.jpg",
      "status": "verified",
      "source_signboard": {               <- Step 1: LLM 추출 원본
        "business_name": "CAFE 051",
        "confidence": { "business_name": 1.0, ... }
      },
      "candidates": [ ... ],              <- Step 2: Bizno 후보 (BusinessCandidate)
      "best_match": { ... },              <- Step 3: 최고 conf_MCP 후보
      "confidence_scores": {              <- Step 5: confFINAL 수식 결과
        "conf_ext": 1.0,
        "conf_llm": 0.997,
        "conf_mcp": 0.90,
        "conf_final": 0.958,
        "weights": { "w_ext": 0.3, "w_llm": 0.3, "w_mcp": 0.4 },
        "debug": {
          "raw_levenshtein_similarity": 1.0,
          "probabilistic_fusion_score": 0.993,
          "attribute_completeness": 1.0
        }
      },
      "warnings": []
    }
    """
    # ── 메타 ───────────────────────────────────────────────────────────────────
    pipeline_version : str                        = Field("1.0.0", description="파이프라인 버전")
    processed_at     : Optional[datetime]         = Field(None,    description="처리 시각 (UTC)")
    image_source     : str                        = Field(...,     description="분석한 이미지 경로 또는 URL")

    # ── 파이프라인 상태 ────────────────────────────────────────────────────────
    status           : PipelineStatus             = Field(...,     description="파이프라인 최종 상태")

    # ── Step 1: LLM 추출 결과 ─────────────────────────────────────────────────
    # extraction_schema.SignboardExtraction 객체를 그대로 수용
    source_signboard : Optional[object]           = Field(None,    description="LLM 1차 추출 결과 (SignboardExtraction)")

    # ── Step 2: Bizno 후보 목록 ───────────────────────────────────────────────
    candidates       : List[BusinessCandidate]    = Field(default_factory=list,
                                                          description="Bizno 검색 후보 목록")

    # ── Step 3: 최종 best_match ───────────────────────────────────────────────
    best_match       : Optional[BusinessCandidate] = Field(None,   description="가장 높은 conf_MCP 후보")

    # ── Step 5: confFINAL 수식 결과 ───────────────────────────────────────────
    # app.py의 _inject_confidence()가 dict 형태로 주입
    confidence_scores: Optional[dict]             = Field(None,    description="논문 제안 신뢰도 수식 결과 (confFINAL)")

    # ── 경고 메시지 ────────────────────────────────────────────────────────────
    warnings         : List[str]                  = Field(default_factory=list,
                                                          description="처리 중 경고 메시지")

    class Config:
        arbitrary_types_allowed = True   # SignboardExtraction 객체 허용
        use_enum_values = True