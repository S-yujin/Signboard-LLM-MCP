"""
evaluate.py

여러 간판 이미지를 배치 실행하고 논문용 실험 결과를 생성합니다.

사용법:
    python evaluate.py                          # sample_data 전체 배치 평가
    python evaluate.py --image-dir ./my_images  # 폴더 지정
    python evaluate.py --output results.csv     # CSV 경로 지정
    python evaluate.py --delay 7                # 요청 간 딜레이(초) 조정
    python evaluate.py --single ./img.jpg       # 단일 이미지 평가

출력:
    - eval_results.csv   : 이미지별 상세 결과 (논문 Table용)
    - ablation_report.csv: Baseline vs 제안 수식 비교 (논문 4절용)
"""

import argparse
import csv
import time
import re
from pathlib import Path
from dataclasses import dataclass, field

from app import run_pipeline
from config import settings
from services.confidence import (
    ConfidenceInput,
    compute_confidence,
    levenshtein_similarity,
    W_EXT, W_LLM, W_MCP,
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_DELAY_SEC = 7
MAX_RETRY = 3


# ──────────────────────────────────────────────────────────────────────────────
# 결과 데이터 구조
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalRecord:
    image_file: str = ""

    # 추출 정보
    extracted_name: str = ""
    extracted_phone: str = ""
    extracted_industry: str = ""
    extracted_address: str = ""

    # ── 논문 제안 수식: 3항 신뢰도 ────────────────────────────────────────
    conf_ext: float = 0.0               # VLM 1차 추출 신뢰도
    conf_llm: float = 0.0               # 시그모이드 확률적 융합 점수
    conf_mcp: float = 0.0               # 외부 API 매칭 점수
    conf_final: float = 0.0             # confFINAL = 0.3*EXT + 0.3*LLM + 0.4*MCP
    raw_levenshtein_similarity: float = 0.0
    probabilistic_fusion_score: float = 0.0
    attribute_completeness: float = 0.0

    # ── Baseline 수식: 기존 MUM+LLM (2항) ─────────────────────────────────
    # confFINAL_baseline = w_MUM * conf_MUM + w_LLM * conf_LLM
    # conf_MUM ≈ conf_ext (동일 추출 모델 가정)
    # conf_LLM_baseline ≈ conf_llm
    # w_MUM = 0.5, w_LLM = 0.5 (기존 논문 기본값)
    conf_final_baseline: float = 0.0    # 기존 2항 수식 결과
    precision_gain: float = 0.0         # conf_final - conf_final_baseline
    # ─────────────────────────────────────────────────────────────────────

    # 파이프라인 결과
    pipeline_status: str = ""
    candidate_count: int = 0
    best_match_name: str = ""
    best_match_regno: str = ""
    best_match_status: str = ""
    best_match_tax: str = ""
    best_confidence: float = 0.0
    status_verified: bool = False

    # ── Hallucination 분석 ────────────────────────────────────────────────
    # hallucination: LLM이 추출한 이름이 MCP에서 전혀 검증 안 된 경우
    is_hallucination: bool = False      # MCP 검증 전 hallucination 의심 여부
    hallucination_suppressed: bool = False  # MCP 연동 후 수정 여부
    # ─────────────────────────────────────────────────────────────────────

    elapsed_sec: float = 0.0
    retry_count: int = 0
    error_msg: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 신뢰도 계산 (기존 2항 수식: MUM + LLM)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_baseline_confidence(conf_mum: float, conf_llm: float) -> float:
    """
    기존 소스 논문 수식 (2항):

        confFINAL_baseline = w_MUM * conf_MUM + w_LLM * conf_LLM
                           = 0.5 * conf_MUM + 0.5 * conf_LLM

    MCP 항이 없으므로 외부 검증 없이 모델 내부 점수만 사용.
    """
    W_MUM_BASE = 0.5
    W_LLM_BASE = 0.5
    return W_MUM_BASE * conf_mum + W_LLM_BASE * conf_llm


# ──────────────────────────────────────────────────────────────────────────────
# Hallucination 판정
# ──────────────────────────────────────────────────────────────────────────────

_HALLUCINATION_SIM_THRESHOLD = 0.4  # Levenshtein 유사도 < 이 값이면 hallucination 의심

def _detect_hallucination(
    extracted_name: str,
    candidate_name: str,
    pipeline_status: str,
) -> tuple[bool, bool]:
    """
    Hallucination 의심 여부 및 MCP 연동 후 억제 여부를 판정합니다.

    판정 기준:
        - is_hallucination  : 추출명과 MCP 후보명 간 Levenshtein 유사도가
                              임계값 미만이거나 MCP 후보 자체가 없는 경우
        - hallucination_suppressed : MCP 연동을 통해 실제 검증 데이터가
                                     확보된 경우 (status == verified/partial)

    Returns:
        (is_hallucination, hallucination_suppressed)
    """
    if not extracted_name:
        return False, False

    if not candidate_name:
        # MCP 후보가 아예 없으면 hallucination 의심
        is_h = True
        suppressed = False
    else:
        sim = levenshtein_similarity(extracted_name, candidate_name)
        is_h = sim < _HALLUCINATION_SIM_THRESHOLD
        # MCP가 verified/partial을 반환했으면 억제 성공
        suppressed = is_h and pipeline_status in ("verified", "partial")

    return is_h, suppressed


# ──────────────────────────────────────────────────────────────────────────────
# 재시도 딜레이 파싱
# ──────────────────────────────────────────────────────────────────────────────

def _parse_retry_delay(error_msg: str, fallback: int = 60) -> int:
    m = re.search(r"retryDelay.*?(\d+)s", error_msg)
    return int(m.group(1)) + 3 if m else fallback


# ──────────────────────────────────────────────────────────────────────────────
# 단일 이미지 평가
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS = ["business_name", "phone", "industry", "address"]

def _attribute_completeness(sb: dict) -> float:
    filled = sum(1 for f in _REQUIRED_FIELDS if sb.get(f) not in (None, "", []))
    return filled / len(_REQUIRED_FIELDS)


def evaluate_one(image_path: str, delay_sec: int = DEFAULT_DELAY_SEC) -> EvalRecord:
    rec = EvalRecord(image_file=Path(image_path).name)
    t0 = time.time()

    for attempt in range(1, MAX_RETRY + 1):
        try:
            result = run_pipeline(image_path)
            rec.elapsed_sec = round(time.time() - t0, 2)
            rec.retry_count = attempt - 1

            sb   = result.get("source_signboard") or {}
            best = result.get("best_match") or {}

            # 추출 정보
            rec.extracted_name     = sb.get("business_name") or ""
            rec.extracted_phone    = sb.get("phone") or ""
            rec.extracted_industry = sb.get("industry") or ""
            rec.extracted_address  = sb.get("address") or ""

            rec.pipeline_status = result.get("status", "")
            rec.candidate_count = len(result.get("candidates") or [])

            if best:
                rec.best_match_name   = best.get("business_name", "")
                rec.best_match_regno  = best.get("registration_number", "")
                rec.best_match_status = best.get("business_status", "")
                rec.best_match_tax    = best.get("tax_type", "")
                rec.best_confidence   = best.get("confidence_score", 0.0)
                rec.status_verified   = best.get("status_verified", False)

            # ── 논문 제안 수식: 3항 신뢰도 계산 ────────────────────────────
            completeness = _attribute_completeness(sb)
            conf_ext_raw = (sb.get("confidence") or {}).get("business_name", 0.0)

            conf_inp = ConfidenceInput(
                conf_ext=conf_ext_raw,
                extracted_name=rec.extracted_name,
                candidate_name=rec.best_match_name,
                attribute_completeness=completeness,
                conf_mcp=rec.best_confidence,
            )
            conf_res = compute_confidence(conf_inp)

            rec.conf_ext   = conf_res.conf_ext
            rec.conf_llm   = conf_res.conf_llm
            rec.conf_mcp   = conf_res.conf_mcp
            rec.conf_final = conf_res.conf_final
            rec.raw_levenshtein_similarity = conf_res.raw_levenshtein_similarity
            rec.probabilistic_fusion_score = conf_res.probabilistic_fusion_score
            rec.attribute_completeness     = conf_res.attribute_completeness

            # ── Baseline 수식: 2항 (MUM + LLM) ─────────────────────────────
            # conf_MUM ≈ conf_ext (동일 추출 단계 가정)
            rec.conf_final_baseline = _compute_baseline_confidence(
                conf_mum=rec.conf_ext,
                conf_llm=rec.conf_llm,
            )
            rec.precision_gain = round(rec.conf_final - rec.conf_final_baseline, 4)

            # ── Hallucination 분석 ──────────────────────────────────────────
            rec.is_hallucination, rec.hallucination_suppressed = _detect_hallucination(
                extracted_name=rec.extracted_name,
                candidate_name=rec.best_match_name,
                pipeline_status=rec.pipeline_status,
            )

            return rec

        except Exception as e:
            err_str = str(e)

            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = _parse_retry_delay(err_str, fallback=65)
                logger.warning("[평가] 429 — %d초 대기 (시도 %d/%d)", wait, attempt, MAX_RETRY)
                print(f"  ⏳ 429 할당량 초과 — {wait}초 대기 중...")
                time.sleep(wait)
                continue

            if "503" in err_str or "UNAVAILABLE" in err_str:
                logger.warning("[평가] 503 — 30초 대기 (시도 %d/%d)", attempt, MAX_RETRY)
                print("  ⏳ 503 서버 과부하 — 30초 대기 중...")
                time.sleep(30)
                continue

            rec.elapsed_sec     = round(time.time() - t0, 2)
            rec.pipeline_status = "error"
            rec.retry_count     = attempt - 1
            rec.error_msg       = err_str[:200]
            logger.error("[평가] 오류 [%s]: %s", image_path, err_str[:200])
            return rec

    rec.elapsed_sec     = round(time.time() - t0, 2)
    rec.pipeline_status = "error"
    rec.retry_count     = MAX_RETRY
    rec.error_msg       = f"최대 재시도({MAX_RETRY}회) 초과"
    return rec


# ──────────────────────────────────────────────────────────────────────────────
# 배치 평가
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    image_dir: str,
    output_csv: str | None = None,
    delay_sec: int = DEFAULT_DELAY_SEC,
) -> list[EvalRecord]:
    image_paths = sorted([
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in SUPPORTED_EXT
    ])

    if not image_paths:
        print(f"[!] 이미지 파일이 없습니다: {image_dir}")
        return []

    print(f"\n총 {len(image_paths)}장 평가 시작 (요청 간 {delay_sec}초 간격)")
    print("=" * 70)

    records: list[EvalRecord] = []
    for i, path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {path.name} 처리 중...")
        rec = evaluate_one(str(path), delay_sec)
        records.append(rec)
        _print_record(rec)

        if i < len(image_paths):
            print(f"  ⏸ {delay_sec}초 대기...")
            time.sleep(delay_sec)

    _print_summary(records)

    if output_csv:
        _save_main_csv(records, output_csv)
        ablation_path = str(Path(output_csv).with_name("ablation_report.csv"))
        _save_ablation_csv(records, ablation_path)
        print(f"\n📄 결과 CSV  : {output_csv}")
        print(f"📄 Ablation  : {ablation_path}")

    return records


# ──────────────────────────────────────────────────────────────────────────────
# 출력 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _print_record(rec: EvalRecord) -> None:
    icon = {"verified": "✅", "partial": "⚠️ ", "not_found": "❌", "error": "💥"}.get(
        rec.pipeline_status, "?"
    )
    retry_str = f" (재시도 {rec.retry_count}회)" if rec.retry_count else ""
    hall_str  = " 🔴H" if rec.is_hallucination else ""
    supp_str  = "→억제✓" if rec.hallucination_suppressed else ""

    print(
        f"  {icon}{hall_str}{supp_str} 추출: {rec.extracted_name or '(실패)':<20}"
        f" | confFINAL: {rec.conf_final:.3f}"
        f" (base: {rec.conf_final_baseline:.3f}, Δ{rec.precision_gain:+.3f})"
        f" | {rec.elapsed_sec}s{retry_str}"
    )
    if rec.error_msg:
        print(f"    오류: {rec.error_msg}")


def _print_summary(records: list[EvalRecord]) -> None:
    total = len(records)
    if not total:
        return

    verified  = sum(1 for r in records if r.pipeline_status == "verified")
    partial   = sum(1 for r in records if r.pipeline_status == "partial")
    not_found = sum(1 for r in records if r.pipeline_status == "not_found")
    errors    = sum(1 for r in records if r.pipeline_status == "error")
    extracted    = sum(1 for r in records if r.extracted_name)
    nts_verified = sum(1 for r in records if r.status_verified)

    # 신뢰도 평균
    avg = lambda attr: sum(getattr(r, attr) for r in records) / total
    avg_conf         = avg("conf_final")
    avg_base         = avg("conf_final_baseline")
    avg_gain         = avg("precision_gain")
    avg_conf_ext     = avg("conf_ext")
    avg_conf_llm     = avg("conf_llm")
    avg_conf_mcp     = avg("conf_mcp")
    avg_lev          = avg("raw_levenshtein_similarity")
    avg_prob         = avg("probabilistic_fusion_score")
    avg_time         = avg("elapsed_sec")
    avg_cand         = avg("candidate_count")

    # Hallucination 통계
    total_hall  = sum(1 for r in records if r.is_hallucination)
    supp_hall   = sum(1 for r in records if r.hallucination_suppressed)
    supp_rate   = (supp_hall / total_hall * 100) if total_hall else 0.0

    print("\n" + "=" * 70)
    print("📊 실험 결과 요약  [논문 4절 결과 및 고찰]")
    print("=" * 70)
    print(f"  총 이미지 수          : {total}장")
    print(f"  상호명 추출 성공      : {extracted}장 ({extracted/total*100:.1f}%)")
    print(f"  검증 완료 (verified)  : {verified}장 ({verified/total*100:.1f}%)")
    print(f"  부분 검증 (partial)   : {partial}장 ({partial/total*100:.1f}%)")
    print(f"  후보 없음 (not_found) : {not_found}장 ({not_found/total*100:.1f}%)")
    print(f"  오류 (error)          : {errors}장 ({errors/total*100:.1f}%)")
    print(f"  국세청 검증 완료      : {nts_verified}장 ({nts_verified/total*100:.1f}%)")
    print(f"  평균 후보 수          : {avg_cand:.1f}건")
    print(f"  평균 처리 시간        : {avg_time:.1f}초")

    print()
    print("  ── 신뢰도 비교 (Ablation Study) ───────────────────────────────")
    print(f"  {'':32} {'Baseline':>10}  {'제안':>10}  {'향상(Δ)':>10}")
    print(f"  {'confFINAL (평균)':32} {avg_base:>10.4f}  {avg_conf:>10.4f}  {avg_gain:>+10.4f}")
    print()
    print(f"  conf_EXT  (VLM 추출,    w={W_EXT:.2f})  : {avg_conf_ext:.4f}")
    print(f"  conf_LLM  (확률적 융합, w={W_LLM:.2f})  : {avg_conf_llm:.4f}")
    print(f"    └ Levenshtein 유사도 (raw)      : {avg_lev:.4f}")
    print(f"    └ Probabilistic Fusion (sigmoid): {avg_prob:.4f}")
    print(f"  conf_MCP  (외부 검증,   w={W_MCP:.2f})  : {avg_conf_mcp:.4f}")

    print()
    print("  ── Hallucination 억제 분석 ─────────────────────────────────────")
    print(f"  Hallucination 의심 건수 : {total_hall}건 ({total_hall/total*100:.1f}%)")
    print(f"  MCP 연동 후 억제 성공   : {supp_hall}건 ({supp_rate:.1f}%  of hallucination)")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# CSV 저장
# ──────────────────────────────────────────────────────────────────────────────

_MAIN_FIELDS = [
    "image_file",
    "extracted_name", "extracted_phone", "extracted_industry", "extracted_address",
    # 제안 수식
    "conf_ext", "conf_llm", "conf_mcp", "conf_final",
    "raw_levenshtein_similarity", "probabilistic_fusion_score", "attribute_completeness",
    # baseline 비교
    "conf_final_baseline", "precision_gain",
    # 파이프라인
    "pipeline_status", "candidate_count",
    "best_match_name", "best_match_regno", "best_match_status",
    "best_match_tax", "best_confidence", "status_verified",
    # hallucination
    "is_hallucination", "hallucination_suppressed",
    # 실행 정보
    "elapsed_sec", "retry_count", "error_msg",
]

def _save_main_csv(records: list[EvalRecord], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_MAIN_FIELDS)
        w.writeheader()
        for rec in records:
            w.writerow({k: getattr(rec, k) for k in _MAIN_FIELDS})


def _save_ablation_csv(records: list[EvalRecord], path: str) -> None:
    """
    논문 Table 생성용 Ablation 리포트.
    이미지별로 Baseline vs 제안 수식을 나란히 비교합니다.
    """
    ablation_fields = [
        "image_file",
        "extracted_name", "best_match_name",
        "raw_levenshtein_similarity", "probabilistic_fusion_score",
        "conf_ext", "conf_llm", "conf_mcp",
        "conf_final_baseline",   # 기존 수식 (MUM+LLM)
        "conf_final",            # 제안 수식 (EXT+LLM+MCP)
        "precision_gain",        # 향상도 Δ
        "is_hallucination", "hallucination_suppressed",
        "pipeline_status", "status_verified",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=ablation_fields)
        w.writeheader()
        for rec in records:
            w.writerow({k: getattr(rec, k) for k in ablation_fields})


# ──────────────────────────────────────────────────────────────────────────────
# 단일 이미지 누적 평가
# ──────────────────────────────────────────────────────────────────────────────

def run_single(image_path: str, output_csv: str) -> None:
    path = Path(image_path)
    if not path.exists():
        print(f"[!] 파일이 없습니다: {image_path}")
        return

    print(f"\n[단일 실행] {path.name} 처리 중...")
    rec = evaluate_one(str(path))
    _print_record(rec)

    csv_path   = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_MAIN_FIELDS)
        if not file_exists:
            w.writeheader()
        w.writerow({k: getattr(rec, k) for k in _MAIN_FIELDS})

    print(f"📄 CSV 누적 저장: {output_csv}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="간판 이미지 배치 평가 + 논문용 리포트 생성")
    parser.add_argument("--image-dir", "-d", default="./sample_data", help="이미지 폴더")
    parser.add_argument("--single",    "-s", default=None,            help="단일 이미지 평가")
    parser.add_argument("--output",    "-o", default="./outputs/eval_results.csv", help="CSV 경로")
    parser.add_argument("--delay",     "-t", type=int, default=DEFAULT_DELAY_SEC,  help="요청 간 딜레이(초)")
    args = parser.parse_args()

    settings.validate()

    if args.single:
        run_single(args.single, args.output)
    else:
        run_evaluation(args.image_dir, args.output, args.delay)


if __name__ == "__main__":
    main()