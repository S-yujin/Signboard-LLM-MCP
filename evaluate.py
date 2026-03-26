"""
evaluate.py
여러 간판 이미지를 배치 실행하고 논문용 실험 결과 표를 생성합니다.

사용법:
    python evaluate.py                              # sample_data 폴더 전체
    python evaluate.py --image-dir ./my_images      # 폴더 지정
    python evaluate.py --output results.csv         # CSV 저장 경로 지정
    python evaluate.py --delay 7                    # 요청 간 딜레이(초) 조정
"""
import argparse
import csv
import time
import re
from pathlib import Path
from dataclasses import dataclass

from app import run_pipeline
from config import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# 무료 티어: 분당 10회 → 요청 간 7초 간격이면 안전
DEFAULT_DELAY_SEC   = 7
MAX_RETRY           = 3     # 429/503 발생 시 최대 재시도 횟수


# ──────────────────────────────────────────────────────────────────────────────
# 결과 데이터 구조
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalRecord:
    image_file:        str   = ""
    extracted_name:    str   = ""
    extracted_phone:   str   = ""
    extracted_industry: str  = ""
    extracted_address: str   = ""
    name_confidence:   float = 0.0
    pipeline_status:   str   = ""
    candidate_count:   int   = 0
    best_match_name:   str   = ""
    best_match_regno:  str   = ""
    best_match_status: str   = ""
    best_match_tax:    str   = ""
    best_confidence:   float = 0.0
    status_verified:   bool  = False
    elapsed_sec:       float = 0.0
    retry_count:       int   = 0
    error_msg:         str   = ""


# ──────────────────────────────────────────────────────────────────────────────
# 재시도 딜레이 파싱 (retryDelay: '38s' → 38)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_retry_delay(error_msg: str, fallback: int = 60) -> int:
    """오류 메시지에서 retryDelay 초를 파싱합니다."""
    m = re.search(r"retryDelay.*?(\d+)s", error_msg)
    if m:
        return int(m.group(1)) + 3   # 여유 3초 추가
    return fallback


# ──────────────────────────────────────────────────────────────────────────────
# 단일 이미지 평가 (재시도 포함)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_one(image_path: str, delay_sec: int = DEFAULT_DELAY_SEC) -> EvalRecord:
    rec = EvalRecord(image_file=Path(image_path).name)
    t0 = time.time()

    for attempt in range(1, MAX_RETRY + 1):
        try:
            result = run_pipeline(image_path)
            rec.elapsed_sec  = round(time.time() - t0, 2)
            rec.retry_count  = attempt - 1

            sb = result.get("source_signboard") or {}
            rec.extracted_name      = sb.get("business_name") or ""
            rec.extracted_phone     = sb.get("phone") or ""
            rec.extracted_industry  = sb.get("industry") or ""
            rec.extracted_address   = sb.get("address") or ""
            rec.name_confidence     = (sb.get("confidence") or {}).get("business_name", 0.0)

            rec.pipeline_status = result.get("status", "")
            rec.candidate_count = len(result.get("candidates") or [])

            best = result.get("best_match")
            if best:
                rec.best_match_name   = best.get("business_name", "")
                rec.best_match_regno  = best.get("registration_number", "")
                rec.best_match_status = best.get("business_status", "")
                rec.best_match_tax    = best.get("tax_type", "")
                rec.best_confidence   = best.get("confidence_score", 0.0)
                rec.status_verified   = best.get("status_verified", False)

            return rec   # 성공

        except Exception as e:
            err_str = str(e)

            # 429 RESOURCE_EXHAUSTED — 할당량 초과, retryDelay만큼 대기 후 재시도
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = _parse_retry_delay(err_str, fallback=65)
                logger.warning(
                    "[평가] 429 할당량 초과 (시도 %d/%d) — %d초 대기 후 재시도...",
                    attempt, MAX_RETRY, wait,
                )
                print(f"  ⏳ 429 할당량 초과 — {wait}초 대기 중...")
                time.sleep(wait)
                continue

            # 503 UNAVAILABLE — 서버 과부하, 30초 대기 후 재시도
            if "503" in err_str or "UNAVAILABLE" in err_str:
                wait = 30
                logger.warning(
                    "[평가] 503 서버 과부하 (시도 %d/%d) — %d초 대기 후 재시도...",
                    attempt, MAX_RETRY, wait,
                )
                print(f"  ⏳ 503 서버 과부하 — {wait}초 대기 중...")
                time.sleep(wait)
                continue

            # 그 외 오류 — 재시도 없이 기록
            rec.elapsed_sec     = round(time.time() - t0, 2)
            rec.pipeline_status = "error"
            rec.retry_count     = attempt - 1
            rec.error_msg       = err_str[:200]
            logger.error("[평가] 오류 [%s]: %s", image_path, err_str[:200])
            return rec

    # MAX_RETRY 모두 소진
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

        # 마지막 이미지가 아니면 딜레이
        if i < len(image_paths):
            print(f"  ⏸  다음 요청까지 {delay_sec}초 대기...")
            time.sleep(delay_sec)

    _print_summary(records)

    if output_csv:
        _save_csv(records, output_csv)
        print(f"\n📄 CSV 저장 완료: {output_csv}")

    return records


# ──────────────────────────────────────────────────────────────────────────────
# 출력 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _print_record(rec: EvalRecord) -> None:
    icon = {"verified": "✅", "partial": "⚠️ ", "not_found": "❌", "error": "💥"}.get(rec.pipeline_status, "?")
    retry_str = f" (재시도 {rec.retry_count}회)" if rec.retry_count else ""
    print(
        f"  {icon} 추출: {rec.extracted_name or '(실패)':<20} "
        f"| 후보: {rec.candidate_count}건 "
        f"| best: {rec.best_match_name or '-':<20} "
        f"| {rec.best_match_status or '-':<8} "
        f"| {rec.elapsed_sec}s{retry_str}"
    )
    if rec.error_msg:
        print(f"     오류: {rec.error_msg}")


def _print_summary(records: list[EvalRecord]) -> None:
    total = len(records)
    if not total:
        return

    verified     = sum(1 for r in records if r.pipeline_status == "verified")
    partial      = sum(1 for r in records if r.pipeline_status == "partial")
    not_found    = sum(1 for r in records if r.pipeline_status == "not_found")
    errors       = sum(1 for r in records if r.pipeline_status == "error")
    extracted    = sum(1 for r in records if r.extracted_name)
    nts_verified = sum(1 for r in records if r.status_verified)
    avg_time     = sum(r.elapsed_sec for r in records) / total
    avg_cand     = sum(r.candidate_count for r in records) / total

    print("\n" + "=" * 70)
    print("📊 실험 결과 요약")
    print("=" * 70)
    print(f"  총 이미지 수          : {total}장")
    print(f"  상호명 추출 성공       : {extracted}장  ({extracted/total*100:.1f}%)")
    print(f"  검증 완료 (verified)  : {verified}장  ({verified/total*100:.1f}%)")
    print(f"  부분 검증 (partial)   : {partial}장  ({partial/total*100:.1f}%)")
    print(f"  후보 없음 (not_found) : {not_found}장  ({not_found/total*100:.1f}%)")
    print(f"  오류 (error)          : {errors}장  ({errors/total*100:.1f}%)")
    print(f"  국세청 검증 완료       : {nts_verified}장  ({nts_verified/total*100:.1f}%)")
    print(f"  평균 후보 수           : {avg_cand:.1f}건")
    print(f"  평균 처리 시간         : {avg_time:.1f}초")
    print("=" * 70)


def _save_csv(records: list[EvalRecord], path: str) -> None:
    fields = [
        "image_file",
        "extracted_name", "extracted_phone", "extracted_industry", "extracted_address",
        "name_confidence", "pipeline_status", "candidate_count",
        "best_match_name", "best_match_regno", "best_match_status",
        "best_match_tax", "best_confidence", "status_verified",
        "elapsed_sec", "retry_count", "error_msg",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in records:
            w.writerow({k: getattr(rec, k) for k in fields})


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def run_single(image_path: str, output_csv: str) -> None:
    """
    이미지 한 장만 평가하고 CSV에 누적 저장합니다.

    사용법:
        python evaluate.py --single ./sample_data/foo.jpg
        python evaluate.py --single ./sample_data/foo.jpg --output results.csv
    """
    path = Path(image_path)
    if not path.exists():
        print(f"[!] 파일이 없습니다: {image_path}")
        return

    print(f"\n[단일 실행] {path.name} 처리 중...")
    rec = evaluate_one(str(path))
    _print_record(rec)

    # CSV가 이미 있으면 헤더 없이 행만 추가, 없으면 새로 생성
    fields = [
        "image_file",
        "extracted_name", "extracted_phone", "extracted_industry", "extracted_address",
        "name_confidence", "pipeline_status", "candidate_count",
        "best_match_name", "best_match_regno", "best_match_status",
        "best_match_tax", "best_confidence", "status_verified",
        "elapsed_sec", "retry_count", "error_msg",
    ]
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()
        w.writerow({k: getattr(rec, k) for k in fields})

    print(f"📄 CSV 누적 저장: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="간판 이미지 배치 평가")
    parser.add_argument("--image-dir", "-d", default="./sample_data",
                        help="배치 평가할 이미지 폴더")
    parser.add_argument("--single",    "-s", default=None,
                        help="이미지 한 장만 평가 (CSV에 누적 저장)")
    parser.add_argument("--output",    "-o", default="./outputs/eval_results.csv",
                        help="CSV 저장 경로")
    parser.add_argument("--delay",     "-t", type=int, default=DEFAULT_DELAY_SEC,
                        help=f"요청 간 딜레이(초), 기본값 {DEFAULT_DELAY_SEC}")
    args = parser.parse_args()

    settings.validate()

    if args.single:
        # 한 장 모드
        run_single(args.single, args.output)
    else:
        # 배치 모드
        run_evaluation(args.image_dir, args.output, args.delay)


if __name__ == "__main__":
    main()