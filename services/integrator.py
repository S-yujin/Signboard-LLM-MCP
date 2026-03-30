"""
services/integrator.py
[최종 통합] — GPS/POI 제거 버전

LLM 추출 결과와 MCP 검증 결과를 합쳐 PipelineResult를 구성합니다.

신뢰도 수식 (3항):
    conf_FINAL = w_BRAND * S_brand + w_BRANCH * S_branch + w_STATUS * S_status

GPS 항(S_gps)은 본 수식에서 완전히 제거되었습니다.
"""
from datetime import datetime
from typing import Optional

from schemas.extraction_schema import SignboardExtraction
from schemas.output_schema import (
    BusinessCandidate,
    BusinessStatus,
    PipelineResult,
    PipelineStatus,
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# 국세청 상태 코드 → BusinessStatus 매핑
_STATUS_CODE_MAP: dict[str, BusinessStatus] = {
    "01": BusinessStatus.ACTIVE,
    "02": BusinessStatus.SUSPENDED,
    "03": BusinessStatus.CLOSED,
}
_STATUS_NAME_MAP: dict[str, BusinessStatus] = {
    "계속사업자": BusinessStatus.ACTIVE,
    "휴업자":     BusinessStatus.SUSPENDED,
    "폐업자":     BusinessStatus.CLOSED,
}


# ──────────────────────────────────────────────────────────────────────────────
# 신뢰도 계산 (3항 수식)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_confidence(
    candidate_name: str,
    extracted_name: str,
    business_status: BusinessStatus,
    status_verified: bool = False,
) -> float:
    """
    제안 수식 v2로 conf_FINAL을 계산합니다:

        conf_FINAL = w_BRAND * S_brand + w_BRANCH * S_branch + w_STATUS * S_status
    """
    from services.confidence import ConfidenceInputV2, compute_confidence_v2

    inp = ConfidenceInputV2(
        extracted_name  = extracted_name,
        candidate_name  = candidate_name,
        business_status = (
            business_status.value
            if hasattr(business_status, "value")
            else str(business_status)
        ),
        status_verified = status_verified,
    )
    result = compute_confidence_v2(inp)

    logger.debug(
        "[신뢰도] %s → S_brand=%.3f S_branch=%s S_status=%.3f → %.4f %s",
        candidate_name,
        result.s_brand,
        f"{result.s_branch:.3f}" if result.s_branch is not None else "N/A",
        result.s_status,
        result.conf_final,
        result.warnings,
    )

    return result.conf_final


# ──────────────────────────────────────────────────────────────────────────────
# PipelineResult 조립
# ──────────────────────────────────────────────────────────────────────────────

def build_pipeline_result(
    image_source: str,
    extraction: SignboardExtraction,
    agent_output: dict,
) -> PipelineResult:
    """
    LLM 추출 결과 + MCP 에이전트 출력을 합쳐 최종 PipelineResult를 반환합니다.

    Args:
        image_source : 분석한 이미지 경로 또는 URL
        extraction   : LLM 간판 추출 결과
        agent_output : run_verification_agent() 반환값
    """
    warnings: list[str] = list(agent_output.get("warnings", []))

    raw_status = agent_output.get("status", "partial")
    try:
        status = PipelineStatus(raw_status)
    except ValueError:
        status = PipelineStatus.PARTIAL
        warnings.append(f"알 수 없는 status 값: {raw_status}")

    candidates: list[BusinessCandidate] = []
    for item in agent_output.get("candidates", []):
        try:
            candidate = _parse_candidate(item, extraction)
            candidates.append(candidate)
        except Exception as e:
            logger.warning("후보 파싱 실패 (건너뜀): %s — %s", item, e)
            warnings.append(f"후보 파싱 오류: {e}")

    best_match: BusinessCandidate | None = None
    raw_best = agent_output.get("best_match")
    if raw_best:
        try:
            best_match = _parse_candidate(raw_best, extraction)
            regno = best_match.registration_number
            matched = next(
                (c for c in candidates if c.registration_number == regno), None
            )
            if matched:
                best_match = matched
        except Exception:
            pass

    if candidates:
        best_by_score = max(candidates, key=lambda c: c.confidence_score)
        if best_match is None or best_by_score.confidence_score > best_match.confidence_score:
            best_match = best_by_score

    logger.info(
        "[통합] status=%s / 후보=%d건 / best=%s",
        status.value,
        len(candidates),
        best_match.business_name if best_match else "없음",
    )

    return PipelineResult(
        pipeline_version="2.0.0",
        processed_at=datetime.utcnow(),
        image_source=image_source,
        status=status,
        source_signboard=extraction,
        candidates=candidates,
        best_match=best_match,
        warnings=warnings,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _parse_candidate(
    item: dict,
    extraction: SignboardExtraction | None = None,
) -> BusinessCandidate:
    """딕셔너리에서 BusinessCandidate Pydantic 모델을 생성합니다."""
    raw_stt_name = item.get("business_status", "unknown")
    raw_stt_cd   = item.get("b_stt_cd", "")
    business_status = (
        _STATUS_CODE_MAP.get(raw_stt_cd)
        or _STATUS_NAME_MAP.get(raw_stt_name)
        or BusinessStatus.UNKNOWN
    )

    candidate_name  = str(item.get("business_name", ""))
    status_verified = bool(item.get("status_verified", False))

    if extraction and extraction.business_name:
        confidence_score = _compute_confidence(
            candidate_name  = candidate_name,
            extracted_name  = extraction.business_name,
            business_status = business_status,
            status_verified = status_verified,
        )
    else:
        confidence_score = float(item.get("confidence_score", 0.0))

    return BusinessCandidate(
        registration_number = str(item.get("registration_number", "")),
        business_name       = candidate_name,
        representative      = item.get("representative"),
        address             = item.get("address"),
        industry            = item.get("industry"),
        phone               = item.get("phone"),
        business_status     = business_status,
        tax_type            = item.get("tax_type"),
        status_verified     = status_verified,
        confidence_score    = confidence_score,
        source              = str(item.get("source", "unknown")),
    )