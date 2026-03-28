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


def _compute_confidence(
    candidate_name: str,
    extracted_name: str,
    extracted_address: str | None,
    business_status: BusinessStatus,
    status_verified: bool = False,
) -> float:
    """
    conf_MCP 계산: 외부 API(Bizno + 국세청) 검증 결과를 수치화합니다.

    논문 수식에서 conf_MCP 항에 해당하며, 다음 기준으로 산정됩니다:

    [상호명 일치도]  최대 0.50
      +0.50  정규화 후 완전 일치  (_normalize_name 적용 — 한글↔영문 동의어 포함)
      +0.35  한쪽이 다른 쪽을 포함
      +0.20  정규화 Levenshtein 유사도 ≥ 0.6

    [국세청 검증 보너스]  최대 0.40
      +0.40  국세청 API verified + 계속사업자   ← 외부 Ground Truth 확인
      +0.20  국세청 API verified + 휴/폐업
      +0.10  verified=False이지만 Bizno 기준 계속사업자

    [지역 일치]  최대 0.10
      +0.10  추출 주소의 토큰이 후보명에 포함

    총합 최대 1.0으로 클램핑.

    설계 의도:
      - 국세청 verified + 계속사업자 조합에 큰 점수(0.40)를 부여해
        MCP 연동의 실질적 가치를 conf_MCP에 반영
      - 단순 이름 매칭만으로는 0.5를 넘기 어려워 외부 검증 없이
        높은 신뢰도가 부여되는 것을 방지 (hallucination 억제 효과)
    """
    import re
    from services.confidence import _normalize_name, levenshtein_similarity

    score = 0.0

    # ── 상호명 일치도 (최대 0.50) ─────────────────────────────────────────────
    norm_c = _normalize_name(candidate_name)
    norm_e = _normalize_name(extracted_name)

    if norm_c and norm_e:
        if norm_c == norm_e:
            score += 0.50                          # 완전 일치
        elif norm_e in norm_c or norm_c in norm_e:
            score += 0.35                          # 포함 관계
        else:
            sim = levenshtein_similarity(extracted_name, candidate_name)
            if sim >= 0.6:
                score += 0.20                      # 유사도 ≥ 0.6

    # ── 국세청 검증 보너스 (최대 0.40) ───────────────────────────────────────
    if status_verified:
        if business_status == BusinessStatus.ACTIVE:
            score += 0.40                          # verified + 계속사업자
        else:
            score += 0.20                          # verified + 휴/폐업
    else:
        if business_status == BusinessStatus.ACTIVE:
            score += 0.10                          # Bizno 기준 계속사업자만

    # ── 지역 일치 (최대 0.10) ────────────────────────────────────────────────
    if extracted_address:
        tokens = re.findall(r"[가-힣a-zA-Z]{2,}", extracted_address)
        for token in tokens:
            if token.lower() in candidate_name.lower():
                score += 0.10
                break

    return round(min(score, 1.0), 2)


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
            candidate = _parse_candidate(item, extraction)
            candidates.append(candidate)
        except Exception as e:
            logger.warning("후보 파싱 실패 (건너뜀): %s — %s", item, e)
            warnings.append(f"후보 파싱 오류: {e}")

    # ── best_match 결정 ────────────────────────────────────────────────────────
    best_match: BusinessCandidate | None = None
    raw_best = agent_output.get("best_match")
    if raw_best:
        try:
            best_match = _parse_candidate(raw_best, extraction)
            # candidates에서 같은 등록번호로 교체 (재계산된 점수 반영)
            regno = best_match.registration_number
            matched = next((c for c in candidates if c.registration_number == regno), None)
            if matched:
                best_match = matched
        except Exception:
            pass

    # best_match를 candidates 중 최고 confidence_score로 결정
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

def _parse_candidate(item: dict, extraction: SignboardExtraction | None = None) -> BusinessCandidate:
    """딕셔너리에서 BusinessCandidate Pydantic 모델을 생성합니다."""
    # business_status 정규화
    raw_stt_name = item.get("business_status", "unknown")
    raw_stt_cd   = item.get("b_stt_cd", "")
    business_status = (
        _STATUS_CODE_MAP.get(raw_stt_cd)
        or _STATUS_NAME_MAP.get(raw_stt_name)
        or BusinessStatus.UNKNOWN
    )

    candidate_name = str(item.get("business_name", ""))

    # confidence_score: LLM 점수 대신 Python에서 직접 재계산 (conf_MCP)
    status_verified = bool(item.get("status_verified", False))
    if extraction and extraction.business_name:
        confidence_score = _compute_confidence(
            candidate_name=candidate_name,
            extracted_name=extraction.business_name,
            extracted_address=extraction.address,
            business_status=business_status,
            status_verified=status_verified,
        )
    else:
        confidence_score = float(item.get("confidence_score", 0.0))

    return BusinessCandidate(
        registration_number=str(item.get("registration_number", "")),
        business_name=candidate_name,
        representative=item.get("representative"),
        address=item.get("address"),
        industry=item.get("industry"),
        phone=item.get("phone"),
        business_status=business_status,
        tax_type=item.get("tax_type"),
        status_verified=bool(item.get("status_verified", False)),
        confidence_score=confidence_score,
        source=str(item.get("source", "unknown")),
    )