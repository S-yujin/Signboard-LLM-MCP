"""
schemas/extraction_schema.py
LLM이 반환하는 간판 추출 결과의 Pydantic 스키마
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ConfidenceScores(BaseModel):
    """각 필드의 LLM 인식 신뢰도 (0.0 ~ 1.0)"""
    business_name: float = Field(default=0.0, ge=0.0, le=1.0)
    phone: float = Field(default=0.0, ge=0.0, le=1.0)
    address: float = Field(default=0.0, ge=0.0, le=1.0)


class SignboardExtraction(BaseModel):
    """
    간판 이미지에서 LLM이 추출한 정보 구조체.
    모든 필드는 인식 실패 시 None을 허용합니다.
    """
    business_name: Optional[str] = Field(None, description="상호명")
    phone: Optional[str] = Field(None, description="정규화된 전화번호")
    industry: Optional[str] = Field(None, description="업종/업태 키워드")
    address: Optional[str] = Field(None, description="주소 조각")
    extra_keywords: List[str] = Field(
        default_factory=list,
        description="기타 키워드(최대 5개 권장)"
    )
    confidence: ConfidenceScores = Field(default_factory=ConfidenceScores)

    @field_validator("business_name", "phone", "industry", "address", mode="before")
    @classmethod
    def empty_string_to_none(cls, v):
        """빈 문자열을 None으로 변환합니다."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("extra_keywords", mode="before")
    @classmethod
    def normalize_keywords(cls, v):
        """키워드 개수 제한 및 공백 제거"""
        if v is None:
            return []

        if not isinstance(v, list):
            return []

        cleaned = []
        seen = set()

        for item in v:
            if not isinstance(item, str):
                continue

            item = item.strip()
            if not item:
                continue

            if item not in seen:
                cleaned.append(item)
                seen.add(item)

            if len(cleaned) >= 5:
                break

        return cleaned

    def is_extractable(self) -> bool:
        """최소한 상호명이라도 추출되었는지 확인합니다."""
        return self.business_name is not None

    def to_search_params(self) -> dict:
        """MCP 후보 조회에 사용할 검색 파라미터를 반환합니다."""
        return {
            "business_name": self.business_name,
            "phone": self.phone,
            "industry": self.industry,
            "address": self.address,
        }