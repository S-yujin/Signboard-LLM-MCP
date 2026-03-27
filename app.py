"""
app.py

간판 분석 → 사업자 검증 파이프라인 진입점

사용법:
    python app.py <이미지_경로_또는_URL> [--output output.json]

예시:
    python app.py ./sample_data/test_signboard.jpg
    python app.py https://example.com/sign.png --output result.json

NOTE:
    현재 구현은 단일 뷰 이미지 기반으로 동작합니다.
    다중 뷰(Multiple Views) 기반의 신뢰도 보정은 향후 연구 과제(Future Work)입니다.
    입력 인터페이스는 List[str] 형태로 설계되어 있어 다중 뷰 확장에 대응 가능합니다.
"""

import argparse
import sys
from pathlib import Path

from config import settings
from schemas.output_schema import PipelineStatus
from services.llm_extractor import extract_from_signboard
from services.verifier import run_verification_agent
from services.integrator import build_pipeline_result
from services.confidence import ConfidenceInput, compute_confidence
from utils.json_utils import pretty_json, save_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 속성 완전성 계산
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS = ["business_name", "phone", "industry", "address"]

def _attribute_completeness(signboard_dict: dict) -> float:
    """추출된 사업자 속성 중 값이 채워진 필드의 비율을 반환합니다."""
    if not signboard_dict:
        return 0.0
    filled = sum(
        1 for f in _REQUIRED_FIELDS
        if signboard_dict.get(f) not in (None, "", [])
    )
    return filled / len(_REQUIRED_FIELDS)


# ──────────────────────────────────────────────────────────────────────────────
# 신뢰도 수식 주입
# ──────────────────────────────────────────────────────────────────────────────

def _inject_confidence(result: dict) -> dict:
    """
    pipeline_result dict에 논문 제안 신뢰도 수식(confFINAL)을 계산해 주입합니다.

        confFINAL = w_EXT * conf_EXT + w_LLM * conf_LLM + w_MCP * conf_MCP

    conf_LLM은 Levenshtein 유사도를 시그모이드로 변환한 확률적 융합 점수입니다.
    """
    sb   = result.get("source_signboard") or {}
    best = result.get("best_match") or {}

    inp = ConfidenceInput(
        conf_ext=( sb.get("confidence") or {}).get("business_name", 0.0),
        extracted_name=sb.get("business_name") or "",
        candidate_name=best.get("business_name") or "",
        attribute_completeness=_attribute_completeness(sb),
        conf_mcp=best.get("confidence_score", 0.0),
    )
    result["confidence_scores"] = compute_confidence(inp).as_dict()
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(image_source: str | list[str]) -> dict:
    """
    간판 이미지에 대한 전체 파이프라인을 실행합니다.

    Pipeline:
        Step 1 [LLM]    이미지 → 상호명/전화/업종/주소 JSON 추출
        Step 2 [MCP]    lookup_business_candidates → 후보 조회 (Bizno.net)
        Step 3 [MCP]    verify_business_status     → 국세청 상태 검증
        Step 4 [통합]   PipelineResult 조립
        Step 5 [신뢰도] confFINAL = w_EXT*conf_EXT + w_LLM*conf_LLM + w_MCP*conf_MCP

    Args:
        image_source: 로컬 파일 경로 또는 공개 이미지 URL.
                      List[str]를 전달할 경우 첫 번째 요소를 단일 이미지로 처리합니다.
                      (다중 뷰 통합은 Future Work)

    Returns:
        PipelineResult + confidence_scores 가 포함된 dict
    """
    # List 입력 대응 (인터페이스는 다중 뷰 확장 가능 구조 유지)
    if isinstance(image_source, list):
        source = image_source[0]
    else:
        source = image_source

    logger.info("=" * 60)
    logger.info("파이프라인 시작: %s", source)
    logger.info("=" * 60)

    # ── Step 1: LLM 이미지 분석 ──────────────────────────────────────────────
    logger.info("[Step 1] LLM 간판 분석 중...")
    extraction = extract_from_signboard(source)

    if not extraction.is_extractable():
        logger.warning("[Step 1] 상호명 인식 실패 — 조기 종료.")
        from schemas.output_schema import PipelineResult
        result = PipelineResult(
            image_source=source,
            status=PipelineStatus.NOT_FOUND,
            source_signboard=extraction,
            warnings=["간판에서 상호명을 인식할 수 없습니다."],
        )
        return _inject_confidence(result.model_dump(mode="json"))

    # ── Step 2 & 3: MCP 에이전트 ─────────────────────────────────────────────
    logger.info("[Step 2-3] MCP 에이전트 실행 중...")
    agent_output = run_verification_agent(extraction)

    # ── Step 4: 최종 통합 ─────────────────────────────────────────────────────
    logger.info("[Step 4] 결과 통합 중...")
    pipeline_result = build_pipeline_result(source, extraction, agent_output)
    result = pipeline_result.model_dump(mode="json")

    # ── Step 5: 신뢰도 수식 계산 ──────────────────────────────────────────────
    result = _inject_confidence(result)

    logger.info("=" * 60)
    logger.info(
        "파이프라인 완료 — status: %s / 후보: %d건 / confFINAL: %.4f",
        result.get("status", "?"),
        len(result.get("candidates") or []),
        (result.get("confidence_scores") or {}).get("conf_final", 0.0),
    )
    logger.info("=" * 60)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="간판 이미지에서 사업자 정보를 추출하고 검증합니다."
    )
    parser.add_argument("image", help="간판 이미지 경로 또는 URL")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="결과를 저장할 JSON 파일 경로 (미지정 시 stdout 출력)",
    )
    args = parser.parse_args()

    try:
        settings.validate()
    except EnvironmentError as e:
        logger.error(str(e))
        sys.exit(1)

    result = run_pipeline(args.image)

    if args.output:
        save_json(result, args.output)
        logger.info("결과 저장 완료: %s", args.output)
    else:
        print(pretty_json(result))


if __name__ == "__main__":
    main()