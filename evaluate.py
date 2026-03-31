"""
evaluate.py

여러 간판 이미지를 배치 실행하고 논문용 실험 결과를 생성합니다.

사용법:
    python evaluate.py                          # sample_data 전체 배치 평가
    python evaluate.py --image-dir ./my_images  # 폴더 지정
    python evaluate.py --output results.csv     # CSV 경로 지정
    python evaluate.py --delay 7                # 요청 간 딜레이(초) 조정
    python evaluate.py --single ./img.jpg       # 단일 이미지 평가 (CSV 누적)
    python evaluate.py --html ./report.html     # HTML 논문용 표 별도 저장

출력:
    - eval_results.csv      : 이미지별 상세 결과 (논문 Table용 원본 데이터)
    - ablation_report.csv   : Baseline vs 제안 수식 Ablation 비교
    - report.html           : 논문 삽입용 HTML 결과 표 (브라우저에서 바로 확인)
"""

import argparse
import csv
import time
import re
import html
from pathlib import Path
from dataclasses import dataclass

from app import run_pipeline
from config import settings
from services.confidence import (
    ConfidenceInput,
    ConfidenceInputV2,
    compute_confidence,
    compute_confidence_v2,
    levenshtein_similarity,
    W_EXT, W_LLM, W_MCP,
    W_BRAND, W_BRANCH, W_STATUS,
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

    # ── LLM 추출 정보 ─────────────────────────────────────────────────────────
    extracted_name: str = ""
    extracted_phone: str = ""
    extracted_industry: str = ""
    extracted_address: str = ""

    # ── 제안 수식 v2: 3항 신뢰도 ─────────────────────────────────────────────
    # conf_FINAL = w_BRAND*S_brand + w_BRANCH*S_branch + w_STATUS*S_status
    s_brand:  float = 0.0          # 브랜드명 Jaro-Winkler 유사도
    s_branch: float = 0.0          # 지점명 일치 점수 (None이면 0으로 저장)
    s_branch_active: bool = True   # S_branch 항 활성 여부
    s_status: float = 0.0          # 국세청 상태 점수
    conf_final: float = 0.0        # confFINAL (제안 수식)
    jaro_winkler_raw: float = 0.0  # 정규화 전 Jaro-Winkler (디버그)

    # ── Baseline 수식: 기존 2항 (MUM + LLM) ──────────────────────────────────
    # confFINAL_baseline = 0.5 * conf_MUM + 0.5 * conf_LLM
    conf_ext: float = 0.0
    conf_llm: float = 0.0
    conf_mcp: float = 0.0
    conf_final_baseline: float = 0.0
    raw_levenshtein_similarity: float = 0.0
    probabilistic_fusion_score: float = 0.0
    attribute_completeness: float = 0.0
    precision_gain: float = 0.0    # conf_final − conf_final_baseline

    # ── 파이프라인 결과 ───────────────────────────────────────────────────────
    pipeline_status: str = ""
    candidate_count: int = 0
    best_match_name: str = ""
    best_match_regno: str = ""
    best_match_status: str = ""
    best_match_tax: str = ""
    best_confidence: float = 0.0
    status_verified: bool = False

    # ── Hallucination 분석 ────────────────────────────────────────────────────
    is_hallucination: bool = False
    hallucination_suppressed: bool = False

    # ── 실행 메타 ─────────────────────────────────────────────────────────────
    elapsed_sec: float = 0.0
    retry_count: int = 0
    error_msg: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 신뢰도 (기존 2항 수식: MUM + LLM)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_baseline_confidence(conf_mum: float, conf_llm: float) -> float:
    """
    기존 소스 논문 수식 (2항):
        confFINAL_baseline = 0.5 * conf_MUM + 0.5 * conf_LLM
    MCP(외부 검증) 항이 없어 API 연동 이전 상태를 대표합니다.
    """
    return 0.5 * conf_mum + 0.5 * conf_llm


# ──────────────────────────────────────────────────────────────────────────────
# Hallucination 판정
# ──────────────────────────────────────────────────────────────────────────────

_HALLUCINATION_SIM_THRESHOLD = 0.4


def _detect_hallucination(
    extracted_name: str,
    candidate_name: str,
    pipeline_status: str,
) -> tuple[bool, bool]:
    """
    Hallucination 의심 여부 및 MCP 연동 후 억제 여부를 판정합니다.

    Returns:
        (is_hallucination, hallucination_suppressed)
    """
    if not extracted_name:
        return False, False

    if not candidate_name:
        return True, False

    sim = levenshtein_similarity(extracted_name, candidate_name)
    is_h = sim < _HALLUCINATION_SIM_THRESHOLD
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

            # ── LLM 추출 정보 ─────────────────────────────────────────────
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

            # ── 제안 수식 v2: S_brand + S_branch + S_status ───────────────
            v2_inp = ConfidenceInputV2(
                extracted_name  = rec.extracted_name,
                candidate_name  = rec.best_match_name,
                business_status = rec.best_match_status or "unknown",
                status_verified = rec.status_verified,
            )
            v2_res = compute_confidence_v2(v2_inp)

            rec.s_brand          = v2_res.s_brand
            rec.s_branch         = v2_res.s_branch if v2_res.s_branch is not None else 0.0
            rec.s_branch_active  = v2_res.s_branch is not None
            rec.s_status         = v2_res.s_status
            rec.conf_final       = v2_res.conf_final
            rec.jaro_winkler_raw = v2_res.jaro_winkler_raw

            # ── Baseline 수식: conf_EXT + conf_LLM ────────────────────────
            completeness  = _attribute_completeness(sb)
            conf_ext_raw  = (sb.get("confidence") or {}).get("business_name", 0.0)

            old_inp = ConfidenceInput(
                conf_ext              = conf_ext_raw,
                extracted_name        = rec.extracted_name,
                candidate_name        = rec.best_match_name,
                attribute_completeness= completeness,
                conf_mcp              = rec.best_confidence,
            )
            old_res = compute_confidence(old_inp)

            rec.conf_ext                  = old_res.conf_ext
            rec.conf_llm                  = old_res.conf_llm
            rec.conf_mcp                  = old_res.conf_mcp
            rec.raw_levenshtein_similarity = old_res.raw_levenshtein_similarity
            rec.probabilistic_fusion_score = old_res.probabilistic_fusion_score
            rec.attribute_completeness    = old_res.attribute_completeness

            rec.conf_final_baseline = _compute_baseline_confidence(
                conf_mum=rec.conf_ext,
                conf_llm=rec.conf_llm,
            )
            rec.precision_gain = round(rec.conf_final - rec.conf_final_baseline, 4)

            # ── Hallucination 분석 ────────────────────────────────────────
            rec.is_hallucination, rec.hallucination_suppressed = _detect_hallucination(
                extracted_name  = rec.extracted_name,
                candidate_name  = rec.best_match_name,
                pipeline_status = rec.pipeline_status,
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
    html_path: str | None = None,
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
        print(f"\n📄 결과 CSV    : {output_csv}")
        print(f"📄 Ablation CSV: {ablation_path}")

    target_html = html_path or (
        str(Path(output_csv).with_name("report.html")) if output_csv else "./outputs/report.html"
    )
    _save_html_report(records, target_html)
    print(f"📄 HTML 리포트  : {target_html}")

    return records


# ──────────────────────────────────────────────────────────────────────────────
# 출력 헬퍼 (콘솔)
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
        f" | confFINAL: {rec.conf_final:.4f}"
        f"  (base: {rec.conf_final_baseline:.4f}, Δ{rec.precision_gain:+.4f})"
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

    avg = lambda attr: sum(getattr(r, attr) for r in records) / total
    avg_conf  = avg("conf_final")
    avg_base  = avg("conf_final_baseline")
    avg_gain  = avg("precision_gain")
    avg_sbrand  = avg("s_brand")
    avg_sstat   = avg("s_status")
    avg_jw      = avg("jaro_winkler_raw")
    avg_time    = avg("elapsed_sec")
    avg_cand    = avg("candidate_count")

    # S_branch 평균 (활성 샘플만)
    active_branch = [r for r in records if r.s_branch_active]
    avg_sbranch = (
        sum(r.s_branch for r in active_branch) / len(active_branch)
        if active_branch else float("nan")
    )

    total_hall = sum(1 for r in records if r.is_hallucination)
    supp_hall  = sum(1 for r in records if r.hallucination_suppressed)
    supp_rate  = (supp_hall / total_hall * 100) if total_hall else 0.0

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
    print("  ── 신뢰도 비교 (Ablation Study) ────────────────────────────────")
    print(f"  {'':32} {'Baseline':>10}  {'제안':>10}  {'향상(Δ)':>10}")
    print(f"  {'confFINAL (평균)':32} {avg_base:>10.4f}  {avg_conf:>10.4f}  {avg_gain:>+10.4f}")
    print()
    print(f"  S_brand  (Jaro-Winkler, w={W_BRAND:.2f})  평균: {avg_sbrand:.4f}")
    sbranch_str = f"{avg_sbranch:.4f}" if not (avg_sbranch != avg_sbranch) else "N/A (지점명 없음)"
    print(f"  S_branch (지점명 일치, w={W_BRANCH:.2f})  평균: {sbranch_str}"
          f"  (활성 {len(active_branch)}/{total}건)")
    print(f"  S_status (국세청 상태,  w={W_STATUS:.2f})  평균: {avg_sstat:.4f}")
    print(f"    └ Jaro-Winkler raw (정규화 전)         : {avg_jw:.4f}")

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
    # 제안 수식 v2
    "s_brand", "s_branch", "s_branch_active", "s_status", "conf_final",
    "jaro_winkler_raw",
    # baseline 비교
    "conf_ext", "conf_llm", "conf_mcp",
    "raw_levenshtein_similarity", "probabilistic_fusion_score", "attribute_completeness",
    "conf_final_baseline", "precision_gain",
    # 파이프라인
    "pipeline_status", "candidate_count",
    "best_match_name", "best_match_regno", "best_match_status",
    "best_match_tax", "best_confidence", "status_verified",
    # hallucination
    "is_hallucination", "hallucination_suppressed",
    # 실행
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
    """논문 Ablation Study용 CSV."""
    ablation_fields = [
        "image_file",
        "extracted_name", "best_match_name",
        # 제안 수식 3항
        "s_brand", "s_branch", "s_branch_active", "s_status",
        "conf_final",           # 제안 수식 결과
        "jaro_winkler_raw",
        # baseline
        "conf_ext", "conf_llm",
        "conf_final_baseline",  # 기존 수식 결과
        "precision_gain",       # 향상도 Δ
        # 검증 상태
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
# HTML 논문용 표 생성
# ──────────────────────────────────────────────────────────────────────────────

def _status_badge(status: str) -> str:
    color = {
        "verified":  "#27ae60",
        "partial":   "#e67e22",
        "not_found": "#e74c3c",
        "error":     "#7f8c8d",
    }.get(status, "#95a5a6")
    label = {
        "verified":  "✔ verified",
        "partial":   "⚠ partial",
        "not_found": "✗ not_found",
        "error":     "💥 error",
    }.get(status, status)
    return (
        f'<span style="background:{color};color:#fff;padding:2px 7px;'
        f'border-radius:4px;font-size:0.82em;white-space:nowrap">{label}</span>'
    )


def _score_bar(value: float, max_val: float = 1.0, color: str = "#2980b9") -> str:
    """수치를 미니 막대 그래프로 시각화합니다."""
    pct = min(100, max(0, value / max_val * 100))
    return (
        f'<div style="display:flex;align-items:center;gap:6px">'
        f'<div style="width:70px;background:#ecf0f1;border-radius:3px;height:10px">'
        f'<div style="width:{pct:.1f}%;background:{color};height:10px;border-radius:3px"></div>'
        f'</div>'
        f'<span style="font-size:0.88em;color:#2c3e50">{value:.4f}</span>'
        f'</div>'
    )


def _gain_cell(gain: float) -> str:
    if gain > 0.005:
        return f'<td style="color:#27ae60;font-weight:700;text-align:center">+{gain:.4f}</td>'
    elif gain < -0.005:
        return f'<td style="color:#e74c3c;font-weight:700;text-align:center">{gain:.4f}</td>'
    else:
        return f'<td style="color:#7f8c8d;text-align:center">{gain:.4f}</td>'


def _save_html_report(records: list[EvalRecord], path: str) -> None:
    """
    논문 삽입용 HTML 결과 표를 생성합니다.

    포함 테이블:
        Table 1 — 이미지별 상세 결과 (제안 수식 3항 + confFINAL)
        Table 2 — Ablation Study: Baseline vs 제안 수식
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    total = len(records)
    if total == 0:
        return

    # ── 기초 통계 계산 ────────────────────────────────────────────────────────
    verified  = sum(1 for r in records if r.pipeline_status == "verified")
    partial   = sum(1 for r in records if r.pipeline_status == "partial")
    not_found = sum(1 for r in records if r.pipeline_status == "not_found")
    errors    = sum(1 for r in records if r.pipeline_status == "error")
    extracted    = sum(1 for r in records if r.extracted_name)
    nts_verified = sum(1 for r in records if r.status_verified)

    avg = lambda attr: sum(getattr(r, attr) for r in records) / total
    avg_conf     = avg("conf_final")
    avg_base     = avg("conf_final_baseline")
    avg_gain     = avg("precision_gain")
    avg_sbrand   = avg("s_brand")
    avg_sstat    = avg("s_status")
    avg_jw       = avg("jaro_winkler_raw")
    avg_time     = avg("elapsed_sec")
    avg_conf_ext = avg("conf_ext")
    avg_conf_llm = avg("conf_llm")

    active_branch = [r for r in records if r.s_branch_active]
    avg_sbranch = (
        sum(r.s_branch for r in active_branch) / len(active_branch)
        if active_branch else None
    )
    sbranch_avg_str = f"{avg_sbranch:.4f}" if avg_sbranch is not None else "N/A"

    total_hall = sum(1 for r in records if r.is_hallucination)
    supp_hall  = sum(1 for r in records if r.hallucination_suppressed)

    # ── Precision / Recall / F1 ───────────────────────────────────────────────
    tp = verified + partial
    fp = not_found
    fn = total - tp - fp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score  = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    # ── Precision / Recall / F1 카드 ─────────────────────────────────────────
    metrics_cards = f"""  <div class="stat-card" style="border-color:#e67e22">
    <div class="label">Precision (정밀도)</div>
    <div class="value">{precision:.4f}</div>
    <div class="sub">TP / (TP + FP)</div>
  </div>
  <div class="stat-card" style="border-color:#e67e22">
    <div class="label">Recall (재현율)</div>
    <div class="value">{recall:.4f}</div>
    <div class="sub">TP / (TP + FN)</div>
  </div>
  <div class="stat-card" style="border-color:#d35400">
    <div class="label">F1-Score</div>
    <div class="value" style="color:#d35400">{f1_score:.4f}</div>
    <div class="sub">Harmonic Mean</div>
  </div>"""

    # ── Table 1 행 생성 ───────────────────────────────────────────────────────
    rows_t1 = []
    for i, r in enumerate(records, 1):
        hall_icon = "🔴" if r.is_hallucination else ""
        supp_icon = "✓" if r.hallucination_suppressed else ""
        branch_str = (
            f"{r.s_branch:.4f}" if r.s_branch_active else
            '<span style="color:#aaa;font-size:0.82em">N/A</span>'
        )
        rows_t1.append(
            f"<tr>"
            f"<td style='text-align:center'>{i}</td>"
            f"<td>{html.escape(r.image_file)}</td>"
            f"<td>{html.escape(r.extracted_name or '—')}</td>"
            f"<td>{html.escape(r.best_match_name or '—')}</td>"
            f"<td>{_score_bar(r.s_brand, color='#2980b9')}</td>"
            f"<td style='text-align:center'>{branch_str}</td>"
            f"<td>{_score_bar(r.s_status, color='#8e44ad')}</td>"
            f"<td>{_score_bar(r.conf_final, color='#16a085')}</td>"
            f"<td>{_status_badge(r.pipeline_status)}</td>"
            f"<td style='text-align:center'>{'✓' if r.status_verified else '—'}</td>"
            f"<td style='text-align:center'>{hall_icon}{supp_icon}</td>"
            f"<td style='text-align:center;color:#7f8c8d'>{r.elapsed_sec}s</td>"
            f"</tr>"
        )

    # ── Table 2 행 생성 (Ablation) ────────────────────────────────────────────
    rows_t2 = []
    for i, r in enumerate(records, 1):
        rows_t2.append(
            f"<tr>"
            f"<td style='text-align:center'>{i}</td>"
            f"<td>{html.escape(r.image_file)}</td>"
            f"<td>{html.escape(r.extracted_name or '—')}</td>"
            f"<td style='text-align:center'>{r.conf_ext:.4f}</td>"
            f"<td style='text-align:center'>{r.conf_llm:.4f}</td>"
            f"<td style='text-align:center;font-weight:600'>{r.conf_final_baseline:.4f}</td>"
            f"<td style='text-align:center'>{r.s_brand:.4f}</td>"
            f"<td style='text-align:center'>{'N/A' if not r.s_branch_active else f'{r.s_branch:.4f}'}</td>"
            f"<td style='text-align:center'>{r.s_status:.4f}</td>"
            f"<td style='text-align:center;font-weight:600'>{r.conf_final:.4f}</td>"
            + _gain_cell(r.precision_gain)
            + f"</tr>"
        )

    # ── 평균 행 (Table 2 footer) ──────────────────────────────────────────────
    footer_t2 = (
        f"<tr style='background:#eaf4fb;font-weight:700'>"
        f"<td colspan='3' style='text-align:right'>평균 (Mean)</td>"
        f"<td style='text-align:center'>{avg_conf_ext:.4f}</td>"
        f"<td style='text-align:center'>{avg_conf_llm:.4f}</td>"
        f"<td style='text-align:center'>{avg_base:.4f}</td>"
        f"<td style='text-align:center'>{avg_sbrand:.4f}</td>"
        f"<td style='text-align:center'>{sbranch_avg_str}</td>"
        f"<td style='text-align:center'>{avg_sstat:.4f}</td>"
        f"<td style='text-align:center'>{avg_conf:.4f}</td>"
        + _gain_cell(avg_gain)
        + f"</tr>"
    )

    # ── HTML 전체 렌더링 ──────────────────────────────────────────────────────
    html_str = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>간판 분석 파이프라인 — 실험 결과 리포트</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    font-size: 13.5px; color: #2c3e50; padding: 32px 40px;
    background: #f7f9fc; line-height: 1.6;
  }}
  h1 {{ font-size: 1.6em; margin-bottom: 6px; color: #1a252f; }}
  h2 {{ font-size: 1.15em; margin: 30px 0 10px; color: #2c3e50;
       border-left: 4px solid #2980b9; padding-left: 10px; }}
  .meta {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 24px; }}
  .formula {{
    background: #eaf4fb; border-left: 4px solid #2980b9;
    padding: 10px 16px; border-radius: 4px; font-family: monospace;
    font-size: 1em; margin: 12px 0 20px; color: #1a252f;
  }}
  table {{
    width: 100%; border-collapse: collapse; background: #fff;
    border-radius: 8px; overflow: hidden;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08); margin-bottom: 8px;
  }}
  th {{
    background: #2c3e50; color: #fff; padding: 9px 10px;
    font-size: 0.82em; text-align: center; white-space: nowrap;
  }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #ecf0f1; vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f0f8ff; }}
  .stat-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 14px; margin: 16px 0 28px;
  }}
  .stat-card {{
    background: #fff; border-radius: 8px; padding: 14px 18px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.07); border-top: 4px solid #2980b9;
  }}
  .stat-card .label {{ font-size: 0.8em; color: #7f8c8d; margin-bottom: 4px; }}
  .stat-card .value {{ font-size: 1.45em; font-weight: 700; color: #1a252f; }}
  .stat-card .sub   {{ font-size: 0.78em; color: #95a5a6; margin-top: 2px; }}
  .note {{ font-size: 0.82em; color: #7f8c8d; margin-top: 6px; }}
  .highlight {{ color: #27ae60; font-weight: 700; }}
  @media print {{
    body {{ padding: 10px; background: #fff; }}
    table {{ box-shadow: none; }}
  }}
</style>
</head>
<body>

<h1>간판 분석 파이프라인 &mdash; 실험 결과 리포트</h1>
<div class="meta">
  생성 시각: {time.strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
  총 이미지: <strong>{total}장</strong> &nbsp;|&nbsp;
  평균 처리 시간: <strong>{avg_time:.1f}초/장</strong>
</div>

<div class="formula">
  conf_FINAL = <strong>w<sub>BRAND</sub></strong> &times; S_brand
             + <strong>w<sub>BRANCH</sub></strong> &times; S_branch
             + <strong>w<sub>STATUS</sub></strong> &times; S_status
  &nbsp;&nbsp; (w = {W_BRAND:.2f} / {W_BRANCH:.2f} / {W_STATUS:.2f})
</div>

<!-- ─── 요약 카드 ──────────────────────────────────────────────────────── -->
<h2>요약 통계 (Summary Statistics)</h2>
<div class="stat-grid">
  <div class="stat-card">
    <div class="label">상호명 추출 성공률</div>
    <div class="value">{extracted/total*100:.1f}<span style="font-size:0.6em">%</span></div>
    <div class="sub">{extracted} / {total} 장</div>
  </div>
  <div class="stat-card" style="border-color:#27ae60">
    <div class="label">Verified (국세청 확인)</div>
    <div class="value" style="color:#27ae60">{verified/total*100:.1f}<span style="font-size:0.6em">%</span></div>
    <div class="sub">{verified} 장 verified / {nts_verified} 장 NTS 완료</div>
  </div>
  <div class="stat-card" style="border-color:#e67e22">
    <div class="label">Partial / Not-found</div>
    <div class="value" style="color:#e67e22">{partial} / {not_found}</div>
    <div class="sub">부분 검증 / 후보 없음</div>
  </div>
  <div class="stat-card" style="border-color:#16a085">
    <div class="label">평균 confFINAL (제안)</div>
    <div class="value" style="color:#16a085">{avg_conf:.4f}</div>
    <div class="sub">Baseline {avg_base:.4f} &nbsp; Δ<span class="highlight">+{avg_gain:.4f}</span></div>
  </div>
  <div class="stat-card" style="border-color:#2980b9">
    <div class="label">평균 S_brand (JW)</div>
    <div class="value">{avg_sbrand:.4f}</div>
    <div class="sub">Jaro-Winkler raw {avg_jw:.4f}</div>
  </div>
  <div class="stat-card" style="border-color:#8e44ad">
    <div class="label">평균 S_status</div>
    <div class="value">{avg_sstat:.4f}</div>
    <div class="sub">국세청 검증 상태 점수</div>
  </div>
  <div class="stat-card" style="border-color:#e74c3c">
    <div class="label">Hallucination 억제</div>
    <div class="value">{supp_hall} / {total_hall}</div>
    <div class="sub">의심 {total_hall}건 중 {supp_hall}건 MCP 억제</div>
  </div>
{metrics_cards}
</div>

<!-- ─── Table 1: 이미지별 상세 결과 ────────────────────────────────────── -->
<h2>Table 1 &nbsp; 이미지별 상세 실험 결과</h2>
<p class="note">S_brand: 브랜드 Jaro-Winkler 유사도 &nbsp;|&nbsp;
  S_branch: 지점명 일치 점수 (N/A = 지점명 없어 비활성) &nbsp;|&nbsp;
  S_status: 국세청 상태 점수 &nbsp;|&nbsp;
  H: Hallucination 의심(🔴) / 억제(✓)</p>
<table>
<thead>
<tr>
  <th>#</th>
  <th>이미지</th>
  <th>추출 상호명</th>
  <th>비즈노 매칭명</th>
  <th>S_brand</th>
  <th>S_branch</th>
  <th>S_status</th>
  <th>confFINAL</th>
  <th>상태</th>
  <th>NTS</th>
  <th>H</th>
  <th>시간</th>
</tr>
</thead>
<tbody>
{''.join(rows_t1)}
</tbody>
</table>

<!-- ─── Table 2: Ablation Study ─────────────────────────────────────────── -->
<h2>Table 2 &nbsp; Ablation Study — Baseline vs 제안 수식</h2>
<p class="note">
  Baseline: conf_FINAL<sub>base</sub> = 0.5&times;conf_EXT + 0.5&times;conf_LLM &nbsp;|&nbsp;
  제안: conf_FINAL = {W_BRAND}&times;S_brand + {W_BRANCH}&times;S_branch + {W_STATUS}&times;S_status
</p>
<table>
<thead>
<tr>
  <th rowspan="2">#</th>
  <th rowspan="2">이미지</th>
  <th rowspan="2">추출 상호명</th>
  <th colspan="3" style="background:#7f8c8d">Baseline (2항)</th>
  <th colspan="4" style="background:#16a085">제안 수식 (3항)</th>
  <th rowspan="2">Δ gain</th>
</tr>
<tr>
  <th style="background:#95a5a6">conf_EXT</th>
  <th style="background:#95a5a6">conf_LLM</th>
  <th style="background:#95a5a6">confFINAL<br><small>baseline</small></th>
  <th style="background:#1abc9c">S_brand</th>
  <th style="background:#1abc9c">S_branch</th>
  <th style="background:#1abc9c">S_status</th>
  <th style="background:#1abc9c">confFINAL<br><small>제안</small></th>
</tr>
</thead>
<tbody>
{''.join(rows_t2)}
</tbody>
<tfoot>
{footer_t2}
</tfoot>
</table>

<p class="note" style="margin-top:8px">
  * Δ gain = confFINAL(제안) − confFINAL(Baseline). 녹색: 제안 수식 우위, 적색: Baseline 우위.
</p>

</body>
</html>
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html_str)


# ──────────────────────────────────────────────────────────────────────────────
# 단일 이미지 누적 평가
# ──────────────────────────────────────────────────────────────────────────────

def run_single(image_path: str, output_csv: str, html_path: str | None = None) -> None:
    path = Path(image_path)
    if not path.exists():
        print(f"[!] 파일이 없습니다: {image_path}")
        return

    print(f"\n[단일 실행] {path.name} 처리 중...")
    rec = evaluate_one(str(path))
    _print_record(rec)

    # CSV 누적 저장
    csv_path    = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_MAIN_FIELDS)
        if not file_exists:
            w.writeheader()
        w.writerow({k: getattr(rec, k) for k in _MAIN_FIELDS})

    print(f"📄 CSV 누적 저장: {output_csv}")

    # HTML 갱신 (전체 CSV를 다시 읽어 재생성)
    target_html = html_path or str(csv_path.with_name("report.html"))
    all_records = _load_records_from_csv(output_csv)
    _save_html_report(all_records, target_html)
    print(f"📄 HTML 리포트  : {target_html}")


def _load_records_from_csv(csv_path: str) -> list[EvalRecord]:
    """CSV 파일에서 EvalRecord 목록을 복원합니다 (HTML 재생성용)."""
    records: list[EvalRecord] = []
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = EvalRecord()
                for field_name in _MAIN_FIELDS:
                    raw = row.get(field_name, "")
                    attr = getattr(rec, field_name)
                    try:
                        if isinstance(attr, bool):
                            setattr(rec, field_name, raw.lower() in ("true", "1", "yes"))
                        elif isinstance(attr, float):
                            setattr(rec, field_name, float(raw) if raw else 0.0)
                        elif isinstance(attr, int):
                            setattr(rec, field_name, int(raw) if raw else 0)
                        else:
                            setattr(rec, field_name, raw)
                    except (ValueError, TypeError):
                        pass
                records.append(rec)
    except Exception as e:
        logger.warning("CSV 복원 실패: %s", e)
    return records


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="간판 이미지 배치 평가 + 논문용 리포트 생성"
    )
    parser.add_argument("--image-dir", "-d", default="./sample_data",
                        help="이미지 폴더 (기본: ./sample_data)")
    parser.add_argument("--single",    "-s", default=None,
                        help="단일 이미지 평가 (CSV 누적)")
    parser.add_argument("--output",    "-o", default="./outputs/eval_results.csv",
                        help="메인 CSV 저장 경로")
    parser.add_argument("--html",      "-H", default=None,
                        help="HTML 리포트 저장 경로 (미지정 시 CSV 동일 폴더)")
    parser.add_argument("--delay",     "-t", type=int, default=DEFAULT_DELAY_SEC,
                        help="요청 간 딜레이(초, 기본: 7)")
    args = parser.parse_args()

    settings.validate()

    if args.single:
        run_single(args.single, args.output, args.html)
    else:
        run_evaluation(args.image_dir, args.output, args.html, args.delay)


if __name__ == "__main__":
    main()