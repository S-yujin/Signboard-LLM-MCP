"""
app.py
간판 분석 → 사업자 검증 파이프라인 진입점

사용법:
    python app.py <이미지_경로_또는_URL> [--output output.json]

예시:
    python app.py ./sample_data/test_signboard.jpg
    python app.py https://example.com/sign.png --output result.json
"""
import argparse
import sys
from pathlib import Path

from config import settings
from schemas.output_schema import PipelineStatus
from services.llm_extractor import extract_from_signboard
from services.verifier import run_verification_agent
from services.integrator import build_pipeline_result
from utils.json_utils import pretty_json, save_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_pipeline(image_source: str) -> dict:
    """
    간판 이미지 한 장에 대한 전체 파이프라인을 실행합니다.

    Pipeline:
        Step 1 [LLM]  이미지 → 상호명/전화/업종/주소 JSON 추출
        Step 2 [MCP]  lookup_business_candidates 도구 → 후보 조회
        Step 3 [MCP]  verify_business_status 도구  → 국세청 상태 검증
        Step 4 [통합] PipelineResult 조립 후 반환

    Args:
        image_source: 로컬 파일 경로 또는 공개 이미지 URL

    Returns:
        PipelineResult를 dict로 직렬화한 값
    """
    logger.info("=" * 60)
    logger.info("파이프라인 시작: %s", image_source)
    logger.info("=" * 60)

    # ── Step 1: LLM 이미지 분석 ────────────────────────────────────────────────
    logger.info("[Step 1] LLM 간판 분석 중...")
    extraction = extract_from_signboard(image_source)

    if not extraction.is_extractable():
        logger.warning("[Step 1] 상호명을 인식하지 못했습니다. 파이프라인 조기 종료.")
        from schemas.output_schema import PipelineResult
        result = PipelineResult(
            image_source=image_source,
            status=PipelineStatus.NOT_FOUND,
            source_signboard=extraction,
            warnings=["간판에서 상호명을 인식할 수 없습니다."],
        )
        return result.model_dump(mode="json")

    # ── Step 2 & 3: MCP 에이전트 (후보 조회 + 상태 검증) ──────────────────────
    logger.info("[Step 2-3] MCP 에이전트 실행 중...")
    agent_output = run_verification_agent(extraction)

    # ── Step 4: 최종 통합 ─────────────────────────────────────────────────────
    logger.info("[Step 4] 결과 통합 중...")
    pipeline_result = build_pipeline_result(image_source, extraction, agent_output)

    logger.info("=" * 60)
    logger.info(
        "파이프라인 완료 — status: %s / 후보: %d건",
        pipeline_result.status.value,
        len(pipeline_result.candidates),
    )
    logger.info("=" * 60)

    return pipeline_result.model_dump(mode="json")


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

    # 설정 유효성 검사
    try:
        settings.validate()
    except EnvironmentError as e:
        logger.error(str(e))
        sys.exit(1)

    # 파이프라인 실행
    result = run_pipeline(args.image)

    # 출력
    output_json = pretty_json(result)
    if args.output:
        save_json(result, args.output)
        logger.info("결과 저장 완료: %s", args.output)
    else:
        print(output_json)


if __name__ == "__main__":
    main()