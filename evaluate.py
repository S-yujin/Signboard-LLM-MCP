"""
evaluate.py
여러 간판 이미지를 배치 실행하고 논문용 실험 결과 표를 생성합니다.

사용법:
    # sample_data 폴더 안의 모든 이미지 평가
    python evaluate.py

    # 특정 폴더 지정
    python evaluate.py --image-dir ./my_images

    # 결과를 CSV로 저장
    python evaluate.py --output results.csv
"""
import argparse
import csv
import time
from pathlib import Path
from dataclasses import dataclass, field

from app import run_pipeline
from config import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# ──────────────────────────────────────────────────────────────────────────────
# 결과 데이터 구조
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalRecord:
    # 입력
    image_file:       str  = ""

    # Step 1: LLM 추출 결과
    extracted_name:   str  = ""
    extracted_phone:  str  = ""
    extracted_industry: str = ""
    extracted_address: str = ""
    name_confidence:  float = 0.0

    # Step 2-3: 검증 결과
    pipeline_status:  str  = ""       # verified / partial / not_found / error
    candidate_count:  int  = 0
    best_match_name:  str  = ""
    best_match_regno: str  = ""
    best_match_status: str = ""       # 계속사업자 / 휴업자 / 폐업자
    best_match_tax:   str  = ""
    best_confidence:  float = 0.0
    status_verified:  bool = False    # 국세청 검증 여부

    # 메타
    elapsed_sec:      float = 0.0
    error_msg:        str  = ""


# ──────────────────────────────────────────────────────────────────────────────
# 단일 이미지 평가
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_one(image_path: str) -> EvalRecord:
    rec = EvalRecord(image_file=Path(image_path).name)
    t0 = time.time()

    try:
        result = run_pipeline(image_path)
        rec.elapsed_sec = round(time.time() - t0, 2)

        # ── Step 1 추출 결과 ───────────────────────────────────────────────────
        sb = result.get("source_signboard") or {}
        rec.extracted_name     = sb.get("business_name") or ""
        rec.extracted_phone    = sb.get("phone") or ""
        rec.extracted_industry = sb.get("industry") or ""
        rec.extracted_address  = sb.get("address") or ""
        rec.name_confidence    = sb.get("confidence", {}).get("business_name", 0.0)

        # ── 검증 결과 ─────────────────────────────────────────────────────────
        rec.pipeline_status  = result.get("status", "")
        rec.candidate_count  = len(result.get("candidates", []))

        best = result.get("best_match")
        if best:
            rec.best_match_name   = best.get("business_name", "")
            rec.best_match_regno  = best.get("registration_number", "")
            rec.best_match_status = best.get("business_status", "")
            rec.best_match_tax    = best.get("tax_type", "")
            rec.best_confidence   = best.get("confidence_score", 0.0)
            rec.status_verified   = best.get("status_verified", False)

    except Exception as e:
        rec.elapsed_sec = round(time.time() - t0, 2)
        rec.pipeline_status = "error"
        rec.error_msg = str(e)
        logger.error("평가 오류 [%s]: %s", image_path, e)

    return rec


# ──────────────────────────────────────────────────────────────────────────────
# 배치 평가 + 출력
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(image_dir: str, output_csv: str | None = None) -> list[EvalRecord]:
    image_paths = sorted([
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in SUPPORTED_EXT
    ])

    if not image_paths:
        print(f"[!] 이미지 파일이 없습니다: {image_dir}")
        return []

    print(f"\n총 {len(image_paths)}장 평가 시작\n" + "=" * 70)

    records: list[EvalRecord] = []
    for i, path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {path.name} 처리 중...")
        rec = evaluate_one(str(path))
        records.append(rec)
        _print_record(rec)

    # ── 요약 통계 ─────────────────────────────────────────────────────────────
    _print_summary(records)

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    if output_csv:
        _save_csv(records, output_csv)
        print(f"\n📄 CSV 저장 완료: {output_csv}")

    return records


def _print_record(rec: EvalRecord) -> None:
    status_icon = {
        "verified":  "✅",
        "partial":   "⚠️ ",
        "not_found": "❌",
        "error":     "💥",
    }.get(rec.pipeline_status, "?")

    print(
        f"  {status_icon} 추출: {rec.extracted_name or '(실패)':<20} "
        f"| 후보: {rec.candidate_count}건 "
        f"| best: {rec.best_match_name or '-':<20} "
        f"| {rec.best_match_status or '-':<8} "
        f"| {rec.elapsed_sec}s"
    )
    if rec.error_msg:
        print(f"     오류: {rec.error_msg}")


def _print_summary(records: list[EvalRecord]) -> None:
    total = len(records)
    if total == 0:
        return

    verified   = sum(1 for r in records if r.pipeline_status == "verified")
    partial    = sum(1 for r in records if r.pipeline_status == "partial")
    not_found  = sum(1 for r in records if r.pipeline_status == "not_found")
    errors     = sum(1 for r in records if r.pipeline_status == "error")
    extracted  = sum(1 for r in records if r.extracted_name)
    nts_verified = sum(1 for r in records if r.status_verified)
    avg_time   = sum(r.elapsed_sec for r in records) / total
    avg_cand   = sum(r.candidate_count for r in records) / total

    print("\n" + "=" * 70)
    print("실험 결과 요약")
    print("=" * 70)
    print(f"  총 이미지 수         : {total}장")
    print(f"  상호명 추출 성공      : {extracted}장  ({extracted/total*100:.1f}%)")
    print(f"  검증 완료 (verified) : {verified}장  ({verified/total*100:.1f}%)")
    print(f"  부분 검증 (partial)  : {partial}장  ({partial/total*100:.1f}%)")
    print(f"  후보 없음 (not_found): {not_found}장  ({not_found/total*100:.1f}%)")
    print(f"  오류 (error)         : {errors}장  ({errors/total*100:.1f}%)")
    print(f"  국세청 검증 완료      : {nts_verified}장  ({nts_verified/total*100:.1f}%)")
    print(f"  평균 후보 수          : {avg_cand:.1f}건")
    print(f"  평균 처리 시간        : {avg_time:.1f}초")
    print("=" * 70)


def _save_csv(records: list[EvalRecord], path: str) -> None:
    fieldnames = [
        "image_file",
        "extracted_name", "extracted_phone", "extracted_industry", "extracted_address",
        "name_confidence",
        "pipeline_status", "candidate_count",
        "best_match_name", "best_match_regno", "best_match_status",
        "best_match_tax", "best_confidence", "status_verified",
        "elapsed_sec", "error_msg",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: getattr(rec, k) for k in fieldnames})


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="간판 이미지 배치 평가 스크립트")
    parser.add_argument(
        "--image-dir", "-d",
        default="./sample_data",
        help="이미지 폴더 경로 (기본: ./sample_data)",
    )
    parser.add_argument(
        "--output", "-o",
        default="./outputs/eval_results.csv",
        help="결과 CSV 저장 경로 (기본: ./outputs/eval_results.csv)",
    )
    args = parser.parse_args()

    settings.validate()
    run_evaluation(args.image_dir, args.output)


if __name__ == "__main__":
    main()