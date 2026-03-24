"""
services/integrator.py
[최종 통합]
LLM 추출 결과와 MCP 검증 결과를 합쳐 PipelineResult를 구성합니다.
"""
from datetime import datetime

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
    "휴업자":    BusinessStatus.SUSPENDED,
    "폐업자":    BusinessStatus.CLOSED,
}


def build_pipeline_result(
    image_source: str,
    extraction: SignboardExtraction,
    agent_output: dict,
) -> PipelineResult:
    """
    LLM 추출 결과 + MCP 에이전트 출력을 합쳐 최종 PipelineResult를 반환합니다.

    Args:
        image_source: 분석한 이미지 경로 또는 URL
        extraction:   LLM 간판 추출 결과
        agent_output: run_verification_agent() 반환값

    Returns:
        PipelineResult
    """
    warnings: list[str] = list(agent_output.get("warnings", []))

    # ── 상태 결정 ──────────────────────────────────────────────────────────────
    raw_status = agent_output.get("status", "partial")
    try:
        status = PipelineStatus(raw_status)
    except ValueError:
        status = PipelineStatus.PARTIAL
        warnings.append(f"알 수 없는 status 값: {raw_status}")

    # ── 후보 파싱 ──────────────────────────────────────────────────────────────
    candidates: list[BusinessCandidate] = []
    for item in agent_output.get("candidates", []):
        try:
            candidate = _parse_candidate(item)
            candidates.append(candidate)
        except Exception as e:
            logger.warning("후보 파싱 실패 (건너뜀): %s — %s", item, e)
            warnings.append(f"후보 파싱 오류: {e}")

    # ── best_match 결정 ────────────────────────────────────────────────────────
    best_match: BusinessCandidate | None = None
    raw_best = agent_output.get("best_match")
    if raw_best:
        try:
            best_match = _parse_candidate(raw_best)
        except Exception:
            # candidates 중 최고 confidence_score를 best_match로 사용
            if candidates:
                best_match = max(candidates, key=lambda c: c.confidence_score)
    elif candidates:
        best_match = max(candidates, key=lambda c: c.confidence_score)

    logger.info(
        "[통합] status=%s / 후보=%d건 / best=%s",
        status.value,
        len(candidates),
        best_match.business_name if best_match else "없음",
    )

    return PipelineResult(
        pipeline_version="1.0.0",
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

def _parse_candidate(item: dict) -> BusinessCandidate:
    """딕셔너리에서 BusinessCandidate Pydantic 모델을 생성합니다."""
    # business_status 정규화
    raw_stt_name = item.get("business_status", "unknown")
    raw_stt_cd   = item.get("b_stt_cd", "")
    business_status = (
        _STATUS_CODE_MAP.get(raw_stt_cd)
        or _STATUS_NAME_MAP.get(raw_stt_name)
        or BusinessStatus.UNKNOWN
    )

    return BusinessCandidate(
        registration_number=str(item.get("registration_number", "")),
        business_name=str(item.get("business_name", "")),
        representative=item.get("representative"),
        address=item.get("address"),
        industry=item.get("industry"),
        phone=item.get("phone"),
        business_status=business_status,
        tax_type=item.get("tax_type"),
        status_verified=bool(item.get("status_verified", False)),
        confidence_score=float(item.get("confidence_score", 0.0)),
        source=str(item.get("source", "unknown")),
    )