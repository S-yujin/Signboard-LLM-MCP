"""
schemas/output_schema.py

논문 '구조화된 JSON 출력' 섹션을 구현하는 Pydantic v2 스키마 모음.

설계 원칙:
    1. 엄격한 타입 강제 (strict mode) → 디지털 지도 인덱싱 효율성 보장
    2. 국세청 / Bizno.net API 응답 필드를 완전히 커버하는 스키마
    3. 신뢰도 수식(confFINAL) 결과를 포함하는 최종 출력 구조
    4. DB 인덱싱을 위한 정규화된 필드(사업자등록번호, 업종코드 등)

논문 4.4절 '구조화된 JSON 출력':
    검증된 정보를 바탕으로 사업자명, 전화번호, 업종 등이 포함된
    규격화된 JSON 포맷으로 출력하며 디지털 지도 인덱싱 효율을 높임.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────────────────────────────────────
# 열거형 (Enum)
# ──────────────────────────────────────────────────────────────────────────────

class PipelineStatus(str, Enum):
    """파이프라인 최종 상태."""
    VERIFIED  = "verified"    # 국세청 검증 완료
    PARTIAL   = "partial"     # Bizno 후보 확인됐으나 국세청 미검증
    NOT_FOUND = "not_found"   # 추출은 됐으나 MCP 후보 없음
    ERROR     = "error"       # 파이프라인 오류


class BusinessStatusCode(str, Enum):
    """국세청 사업자 상태 코드."""
    ACTIVE   = "01"   # 계속사업자
    CLOSED   = "02"   # 폐업자
    SUSPENDED = "03"  # 휴업자


class TaxType(str, Enum):
    """과세 유형."""
    GENERAL   = "일반과세자"
    SIMPLIFIED = "간이과세자"
    EXEMPT    = "면세사업자"
    UNKNOWN   = "알수없음"


# ──────────────────────────────────────────────────────────────────────────────
# 필드 수준 공통 설정
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_phone(v: Optional[str]) -> Optional[str]:
    """전화번호를 'XXX-XXXX-XXXX' 또는 'XX-XXXX-XXXX' 형식으로 정규화."""
    if not v:
        return None
    digits = re.sub(r"\D", "", v)
    if len(digits) == 11:
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
    if len(digits) == 10:
        return f"{digits[:2]}-{digits[2:6]}-{digits[6:]}"
    return v  # 정규화 불가 시 원본 반환


def _normalize_regno(v: Optional[str]) -> Optional[str]:
    """사업자등록번호를 'XXX-XX-XXXXX' 형식으로 정규화."""
    if not v:
        return None
    digits = re.sub(r"\D", "", v)
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
    return v


# ──────────────────────────────────────────────────────────────────────────────
# 1. 간판 이미지 추출 결과 (LLM 1차 출력)
# ──────────────────────────────────────────────────────────────────────────────

class ExtractionConfidence(BaseModel):
    """VLM이 각 속성 추출 시 반환하는 신뢰도 점수 (0.0 ~ 1.0)."""
    business_name : float = Field(0.0, ge=0.0, le=1.0, description="상호명 추출 신뢰도")
    phone         : float = Field(0.0, ge=0.0, le=1.0, description="전화번호 추출 신뢰도")
    industry      : float = Field(0.0, ge=0.0, le=1.0, description="업종 추출 신뢰도")
    address       : float = Field(0.0, ge=0.0, le=1.0, description="주소 추출 신뢰도")


class SignboardExtraction(BaseModel):
    """
    Step 1: LLM/VLM이 간판 이미지에서 추출한 원시(raw) 사업자 정보.

    논문 3.1절 '다중 가설 기반 속성 추출'의 출력 스키마.
    """
    business_name : Optional[str]  = Field(None, description="간판에서 인식된 상호명")
    phone         : Optional[str]  = Field(None, description="전화번호 (원본)")
    industry      : Optional[str]  = Field(None, description="업종·업태 (예: 한식, 카페)")
    address       : Optional[str]  = Field(None, description="주소 (간판에 표시된 경우)")
    extra_text    : Optional[str]  = Field(None, description="기타 추출된 텍스트")
    confidence    : ExtractionConfidence = Field(
        default_factory=ExtractionConfidence,
        description="속성별 VLM 추출 신뢰도",
    )

    @field_validator("phone", mode="before")
    @classmethod
    def normalize_phone(cls, v: Optional[str]) -> Optional[str]:
        return _normalize_phone(v)

    def is_extractable(self) -> bool:
        """상호명이 인식된 경우만 파이프라인을 계속 진행합니다."""
        return bool(self.business_name and self.business_name.strip())


# ──────────────────────────────────────────────────────────────────────────────
# 2. Bizno.net API 응답 스키마 (Step 2: 후보 조회)
# ──────────────────────────────────────────────────────────────────────────────

class BiznoCandidate(BaseModel):
    """
    Bizno.net API fapi 엔드포인트 응답의 단일 후보 항목.

    API 문서: https://bizno.net/api/fapi?key=&q=&gb=3&type=json
    """
    registration_number  : Optional[str]  = Field(None, description="사업자등록번호 (XXX-XX-XXXXX)")
    business_name        : Optional[str]  = Field(None, description="상호명")
    representative_name  : Optional[str]  = Field(None, description="대표자명")
    industry_code        : Optional[str]  = Field(None, description="업종 코드 (표준산업분류)")
    industry_name        : Optional[str]  = Field(None, description="업종명")
    business_type        : Optional[str]  = Field(None, description="업태 (예: 소매)")
    address              : Optional[str]  = Field(None, description="사업장 주소")
    phone                : Optional[str]  = Field(None, description="대표 전화번호")
    open_date            : Optional[str]  = Field(None, description="개업일 (YYYYMMDD)")
    confidence_score     : float          = Field(0.0, ge=0.0, le=1.0, description="MCP 매칭 신뢰도 (conf_MCP)")

    @field_validator("registration_number", mode="before")
    @classmethod
    def normalize_regno(cls, v: Optional[str]) -> Optional[str]:
        return _normalize_regno(v)

    @field_validator("phone", mode="before")
    @classmethod
    def normalize_phone(cls, v: Optional[str]) -> Optional[str]:
        return _normalize_phone(v)


# ──────────────────────────────────────────────────────────────────────────────
# 3. 국세청 API 응답 스키마 (Step 3: 상태 검증)
# ──────────────────────────────────────────────────────────────────────────────

class NtsVerificationResult(BaseModel):
    """
    국세청 사업자등록정보 진위확인 및 상태조회 API 응답 스키마.

    API: https://api.odcloud.kr/api/nts-businessman/v1/status
    """
    registration_number  : Optional[str]           = Field(None, description="사업자등록번호")
    business_status      : Optional[str]            = Field(None, description="사업자 상태 (계속/폐업/휴업)")
    business_status_code : Optional[BusinessStatusCode] = Field(None, description="상태 코드 (01/02/03)")
    tax_type             : Optional[TaxType]        = Field(TaxType.UNKNOWN, description="과세 유형")
    tax_type_code        : Optional[str]            = Field(None, description="과세 유형 코드")
    close_date           : Optional[str]            = Field(None, description="폐업일 (YYYYMMDD, 폐업 시)")
    is_active            : bool                     = Field(False, description="현재 영업 중 여부")

    @model_validator(mode="after")
    def set_is_active(self) -> "NtsVerificationResult":
        """business_status_code 기준으로 is_active 자동 설정."""
        self.is_active = (self.business_status_code == BusinessStatusCode.ACTIVE)
        return self


# ──────────────────────────────────────────────────────────────────────────────
# 4. 최종 검증 매칭 결과 (best_match)
# ──────────────────────────────────────────────────────────────────────────────

class VerifiedBusinessInfo(BaseModel):
    """
    Bizno 후보 + 국세청 검증을 통합한 최종 사업자 정보.

    디지털 지도 DB 인덱싱을 위한 정규화된 필드를 포함합니다.
    """
    # ── 식별 정보 ──────────────────────────────────────────────────────────
    registration_number  : Optional[str]  = Field(None, description="사업자등록번호 (DB 인덱스 키)")
    business_name        : Optional[str]  = Field(None, description="공식 상호명")
    representative_name  : Optional[str]  = Field(None, description="대표자명")

    # ── 업종 정보 ──────────────────────────────────────────────────────────
    industry_code        : Optional[str]  = Field(None, description="표준산업분류 코드 (인덱싱용)")
    industry_name        : Optional[str]  = Field(None, description="업종명")
    business_type        : Optional[str]  = Field(None, description="업태")

    # ── 연락처 / 위치 ──────────────────────────────────────────────────────
    address              : Optional[str]  = Field(None, description="사업장 주소")
    phone                : Optional[str]  = Field(None, description="대표 전화번호")
    open_date            : Optional[str]  = Field(None, description="개업일")

    # ── 국세청 검증 상태 ───────────────────────────────────────────────────
    business_status      : Optional[str]  = Field(None, description="사업자 상태 (국세청 원문)")
    business_status_code : Optional[str]  = Field(None, description="상태 코드")
    tax_type             : Optional[str]  = Field(None, description="과세 유형")
    is_active            : bool           = Field(False, description="현재 영업 중 여부")
    status_verified      : bool           = Field(False, description="국세청 검증 완료 여부")

    # ── 신뢰도 ─────────────────────────────────────────────────────────────
    confidence_score     : float          = Field(0.0, ge=0.0, le=1.0, description="MCP 매칭 신뢰도 (conf_MCP)")

    @field_validator("registration_number", mode="before")
    @classmethod
    def normalize_regno(cls, v: Optional[str]) -> Optional[str]:
        return _normalize_regno(v)

    @field_validator("phone", mode="before")
    @classmethod
    def normalize_phone(cls, v: Optional[str]) -> Optional[str]:
        return _normalize_phone(v)


# ──────────────────────────────────────────────────────────────────────────────
# 5. 신뢰도 수식 결과 스키마
# ──────────────────────────────────────────────────────────────────────────────

class ConfidenceWeights(BaseModel):
    """논문 수식 가중치."""
    w_ext : float = Field(..., ge=0.0, le=1.0, description="conf_EXT 가중치 (기본 0.30)")
    w_llm : float = Field(..., ge=0.0, le=1.0, description="conf_LLM 가중치 (기본 0.30)")
    w_mcp : float = Field(..., ge=0.0, le=1.0, description="conf_MCP 가중치 (기본 0.40)")

    @model_validator(mode="after")
    def check_sum(self) -> "ConfidenceWeights":
        total = self.w_ext + self.w_llm + self.w_mcp
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"가중치 합계는 1.0이어야 합니다. 현재: {total:.6f}")
        return self


class ConfidenceDebugInfo(BaseModel):
    """conf_LLM 계산 중간값 (논문 재현 및 ablation 분석용)."""
    raw_levenshtein_similarity   : float = Field(0.0, ge=0.0, le=1.0, description="Levenshtein 정규화 유사도 (선형)")
    probabilistic_fusion_score   : float = Field(0.0, ge=0.0, le=1.0, description="시그모이드 변환 후 확률적 융합 점수")
    attribute_completeness       : float = Field(0.0, ge=0.0, le=1.0, description="속성 완전성 (채워진 필드 비율)")


class ConfidenceScores(BaseModel):
    """
    논문 제안 신뢰도 수식 전체 결과.

        confFINAL = w_EXT * conf_EXT + w_LLM * conf_LLM + w_MCP * conf_MCP

    conf_EXT : VLM 1차 추출 신뢰도
    conf_LLM : Levenshtein 기반 확률적 융합 (Probabilistic Fusion)
    conf_MCP : 국세청/Bizno 외부 검증 매칭 점수
    """
    conf_ext   : float = Field(..., ge=0.0, le=1.0, description="VLM 1차 추출 신뢰도")
    conf_llm   : float = Field(..., ge=0.0, le=1.0, description="LLM 문맥 타당성 점수 (확률적 융합)")
    conf_mcp   : float = Field(..., ge=0.0, le=1.0, description="MCP 외부 검증 점수")
    conf_final : float = Field(..., ge=0.0, le=1.0, description="최종 신뢰도 (가중 합산)")
    weights    : ConfidenceWeights
    debug      : ConfidenceDebugInfo = Field(default_factory=ConfidenceDebugInfo)


# ──────────────────────────────────────────────────────────────────────────────
# 6. 최종 파이프라인 결과 (PipelineResult)
# ──────────────────────────────────────────────────────────────────────────────

class PipelineResult(BaseModel):
    """
    파이프라인 전체 실행 결과.

    논문 3.4절 '구조화된 JSON 출력'의 최종 스키마.
    디지털 지도 DB 인덱싱에 바로 사용할 수 있는 형태로 설계됨.

    JSON 예시:
    {
        "image_source": "./sample_data/shop.jpg",
        "status": "verified",
        "source_signboard": { ... },          ← LLM 추출 원본
        "candidates": [ ... ],                ← Bizno 후보 목록
        "best_match": { ... },                ← 최종 선택된 사업자 정보
        "confidence_scores": { ... },         ← confFINAL 수식 결과
        "warnings": []
    }
    """
    image_source      : str                             = Field(..., description="처리된 이미지 경로 또는 URL")
    status            : PipelineStatus                  = Field(..., description="파이프라인 최종 상태")
    source_signboard  : Optional[SignboardExtraction]   = Field(None, description="LLM 1차 추출 결과")
    candidates        : list[BiznoCandidate]            = Field(default_factory=list, description="Bizno 후보 목록")
    best_match        : Optional[VerifiedBusinessInfo]  = Field(None, description="최종 검증된 사업자 정보")
    confidence_scores : Optional[ConfidenceScores]      = Field(None, description="논문 제안 신뢰도 수식 결과")
    warnings          : list[str]                       = Field(default_factory=list, description="처리 중 경고 메시지")

    class Config:
        use_enum_values = True

    @model_validator(mode="after")
    def check_best_match_consistency(self) -> "PipelineResult":
        """
        best_match가 있을 때 status 일관성 검증.
        verified 상태라면 best_match가 반드시 존재해야 합니다.
        """
        if self.status == PipelineStatus.VERIFIED and self.best_match is None:
            raise ValueError("status='verified'이면 best_match가 있어야 합니다.")
        return self