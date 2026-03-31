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
from dataclasses import dataclass, field

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

# 제안 수식 가중치 (논문 수식)
_W_EXT = W_EXT    # conf_EXT: LLM 1차 추출 기본 신뢰도
_W_LLM = W_LLM    # conf_LLM: 문맥 타당성 (속성 완성도 + 확률적 융합)
_W_MCP = W_MCP    # conf_MCP: 비즈노/국세청 외부 DB 일치 점수
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

    # ── 제안 수식 (논문): 3항 신뢰도 ─────────────────────────────────────────
    # conf_FINAL = w_EXT*conf_EXT + w_LLM*conf_LLM + w_MCP*conf_MCP
    conf_ext: float = 0.0          # LLM 1차 추출 기본 신뢰도 (VLM confidence)
    conf_llm: float = 0.0          # 문맥 타당성 (속성 완성도 + probabilistic fusion)
    conf_mcp: float = 0.0          # 외부 DB 일치 점수 (비즈노/국세청)
    conf_final: float = 0.0        # confFINAL (제안 수식)
    raw_levenshtein_similarity: float = 0.0
    probabilistic_fusion_score: float = 0.0
    attribute_completeness: float = 0.0

    # ── Baseline 수식 (기존 v2): S_brand + S_branch + S_status ───────────────
    s_brand:  float = 0.0          # 브랜드명 Jaro-Winkler 유사도
    s_branch: float = 0.0          # 지점명 일치 점수 (None이면 0으로 저장)
    s_branch_active: bool = True   # S_branch 항 활성 여부
    s_status: float = 0.0          # 국세청 상태 점수
    conf_final_baseline: float = 0.0  # confFINAL (Baseline v2 수식)
    jaro_winkler_raw: float = 0.0  # 정규화 전 Jaro-Winkler (디버그)
    precision_gain: float = 0.0    # conf_final(제안) − conf_final_baseline

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
# Baseline 신뢰도 (기존 수식: w_MUM*conf_MUM + w_LLM*conf_LLM)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_baseline_confidence(conf_mum: float, conf_llm: float) -> float:
    """
    기존 소스 논문 수식 (2항):
        confFINAL_baseline = w_MUM * conf_MUM + w_LLM * conf_LLM
                           = 0.5 * conf_MUM + 0.5 * conf_LLM

    conf_MUM: MUM/BERT 계열 추출 모델의 신뢰도 → 여기서는 conf_EXT(VLM 추출 기본 신뢰도)로 대응
    conf_LLM: LLM 검증자의 문맥 타당성 점수 → probabilistic_fusion 기반 conf_LLM으로 대응
    MCP(외부 API 검증) 항이 없는 것이 Baseline과 제안 수식의 핵심 차이.
    """
    return 0.5 * conf_mum + 0.5 * conf_llm


# ──────────────────────────────────────────────────────────────────────────────
# 제안 수식 신뢰도 (논문 3항: w_EXT*conf_EXT + w_LLM*conf_LLM + w_MCP*conf_MCP)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_proposed_confidence(
    conf_ext: float,
    conf_llm: float,
    conf_mcp: float,
) -> float:
    """
    논문 제안 수식 (3항):
        conf_FINAL = w_EXT * conf_EXT + w_LLM * conf_LLM + w_MCP * conf_MCP

    conf_EXT: LLM/VLM이 이미지에서 정보를 1차 추출할 때의 기본 신뢰도
    conf_LLM: 추출된 정보의 문맥적 타당성을 LLM이 내부적으로 평가한 점수
              (속성 완성도 + probabilistic_fusion 기반)
    conf_MCP: 국세청/Bizno.net 실제 사업자 데이터와 추출 데이터 간의 일치 점수
    """
    total_w = _W_EXT + _W_LLM + _W_MCP
    w_ext = _W_EXT / total_w
    w_llm = _W_LLM / total_w
    w_mcp = _W_MCP / total_w
    result = w_ext * conf_ext + w_llm * conf_llm + w_mcp * conf_mcp
    return round(max(0.0, min(1.0, result)), 4)


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

            # ── 제안 수식 (논문): w_EXT*conf_EXT + w_LLM*conf_LLM + w_MCP*conf_MCP
            completeness = _attribute_completeness(sb)
            conf_ext_raw = (sb.get("confidence") or {}).get("business_name", 0.0)

            proposed_inp = ConfidenceInput(
                conf_ext               = conf_ext_raw,
                extracted_name         = rec.extracted_name,
                candidate_name         = rec.best_match_name,
                attribute_completeness = completeness,
                conf_mcp               = rec.best_confidence,
            )
            proposed_res = compute_confidence(proposed_inp)

            rec.conf_ext                   = proposed_res.conf_ext
            rec.conf_llm                   = proposed_res.conf_llm   # 문맥 타당성 (속성 완성도 + probabilistic fusion)
            rec.conf_mcp                   = proposed_res.conf_mcp   # 비즈노/국세청 외부 DB 일치 점수
            rec.raw_levenshtein_similarity  = proposed_res.raw_levenshtein_similarity
            rec.probabilistic_fusion_score  = proposed_res.probabilistic_fusion_score
            rec.attribute_completeness     = proposed_res.attribute_completeness

            # 제안 수식 최종값: w_EXT*conf_EXT + w_LLM*conf_LLM + w_MCP*conf_MCP
            rec.conf_final = _compute_proposed_confidence(
                conf_ext = rec.conf_ext,
                conf_llm = rec.conf_llm,
                conf_mcp = rec.conf_mcp,
            )

            # ── Baseline (기존 수식): w_MUM*conf_MUM + w_LLM*conf_LLM ──────
            # conf_MUM → conf_EXT로 대응 (MUM/BERT 계열 추출 신뢰도)
            # conf_LLM → LLM 문맥 타당성 점수 (동일)
            # MCP 항 없음이 Baseline과의 핵심 차이
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
            rec.jaro_winkler_raw = v2_res.jaro_winkler_raw

            # Baseline = w_MUM*conf_MUM + w_LLM*conf_LLM (MCP 항 없음)
            rec.conf_final_baseline = _compute_baseline_confidence(
                conf_mum = rec.conf_ext,
                conf_llm = rec.conf_llm,
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
        f" | confFINAL(제안): {rec.conf_final:.4f}"
        f"  [EXT={rec.conf_ext:.3f} LLM={rec.conf_llm:.3f} MCP={rec.conf_mcp:.3f}]"
        f"  (Baseline: {rec.conf_final_baseline:.4f}, Δ{rec.precision_gain:+.4f})"
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
    print(f"  ── 제안 수식 항별 평균 (w_EXT={_W_EXT:.2f} / w_LLM={_W_LLM:.2f} / w_MCP={_W_MCP:.2f}) ──")
    avg_conf_ext = sum(r.conf_ext for r in records) / total
    avg_conf_llm = sum(r.conf_llm for r in records) / total
    avg_conf_mcp = sum(r.conf_mcp for r in records) / total
    print(f"  conf_EXT (VLM 1차 추출 신뢰도)     평균: {avg_conf_ext:.4f}")
    print(f"  conf_LLM (문맥 타당성)              평균: {avg_conf_llm:.4f}")
    print(f"  conf_MCP (비즈노/국세청 일치 점수)  평균: {avg_conf_mcp:.4f}")
    print()
    print(f"  ── Baseline 항별 평균 (S_brand/branch/status) ──────────────────")
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
    # 제안 수식 (논문): w_EXT*conf_EXT + w_LLM*conf_LLM + w_MCP*conf_MCP
    "conf_ext", "conf_llm", "conf_mcp",
    "raw_levenshtein_similarity", "probabilistic_fusion_score", "attribute_completeness",
    "conf_final",
    # Baseline: w_MUM*conf_MUM + w_LLM*conf_LLM (S_brand+S_branch+S_status)
    "s_brand", "s_branch", "s_branch_active", "s_status",
    "jaro_winkler_raw",
    "conf_final_baseline",
    "precision_gain",
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
        # 제안 수식 3항 (논문)
        "conf_ext", "conf_llm", "conf_mcp",
        "conf_final",           # 제안 수식 결과
        # baseline (기존 수식: S_brand+S_branch+S_status)
        "s_brand", "s_branch", "s_branch_active", "s_status",
        "conf_final_baseline",  # 기존 수식 결과
        "precision_gain",       # 향상도 Δ (제안 - baseline)
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
        Table 3 — 요약 통계 (논문 결과 섹션용)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    total = len(records)
    if total == 0:
        return

    # ── 요약 통계 계산 ────────────────────────────────────────────────────────
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

    active_branch = [r for r in records if r.s_branch_active]
    avg_sbranch = (
        sum(r.s_branch for r in active_branch) / len(active_branch)
        if active_branch else None
    )

    total_hall = sum(1 for r in records if r.is_hallucination)
    supp_hall  = sum(1 for r in records if r.hallucination_suppressed)

    # ── Table 1 행 생성 (제안 수식: conf_EXT / conf_LLM / conf_MCP / conf_final) ──
    rows_t1 = []
    for i, r in enumerate(records, 1):
        hall_icon = "🔴" if r.is_hallucination else ""
        supp_icon = "✓" if r.hallucination_suppressed else ""
        rows_t1.append(
            f"<tr>"
            f"<td style='text-align:center'>{i}</td>"
            f"<td>{html.escape(r.image_file)}</td>"
            f"<td>{html.escape(r.extracted_name or '—')}</td>"
            f"<td>{html.escape(r.best_match_name or '—')}</td>"
            f"<td style='text-align:center'>{r.conf_ext:.4f}</td>"
            f"<td style='text-align:center'>{r.conf_llm:.4f}</td>"
            f"<td>{_score_bar(r.conf_mcp, color='#27ae60')}</td>"
            f"<td>{_score_bar(r.conf_final, color='#16a085')}</td>"
            f"<td>{_status_badge(r.pipeline_status)}</td>"
            f"<td style='text-align:center'>{'✓' if r.status_verified else '—'}</td>"
            f"<td style='text-align:center'>{hall_icon}{supp_icon}</td>"
            f"<td style='text-align:center;color:#7f8c8d'>{r.elapsed_sec}s</td>"
            f"</tr>"
        )

    # ── Table 2 행 생성 (Ablation: Baseline(S_brand+branch+status) vs 제안(EXT+LLM+MCP)) ──
    rows_t2 = []
    for i, r in enumerate(records, 1):
        branch_str = f"{r.s_branch:.4f}" if r.s_branch_active else "N/A"
        rows_t2.append(
            f"<tr>"
            f"<td style='text-align:center'>{i}</td>"
            f"<td>{html.escape(r.image_file)}</td>"
            f"<td>{html.escape(r.extracted_name or '—')}</td>"
            f"<td style='text-align:center'>{r.s_brand:.4f}</td>"
            f"<td style='text-align:center'>{branch_str}</td>"
            f"<td style='text-align:center'>{r.s_status:.4f}</td>"
            f"<td style='text-align:center;font-weight:600'>{r.conf_final_baseline:.4f}</td>"
            f"<td style='text-align:center'>{r.conf_ext:.4f}</td>"
            f"<td style='text-align:center'>{r.conf_llm:.4f}</td>"
            f"<td style='text-align:center'>{r.conf_mcp:.4f}</td>"
            f"<td style='text-align:center;font-weight:600'>{r.conf_final:.4f}</td>"
            + _gain_cell(r.precision_gain)
            + f"</tr>"
        )

    # ── 평균 행 (Table 2 footer) ──────────────────────────────────────────────
    sbranch_avg_str = f"{avg_sbranch:.4f}" if avg_sbranch is not None else "N/A"
    avg_conf_ext = avg("conf_ext")
    avg_conf_llm = avg("conf_llm")
    avg_conf_mcp = avg("conf_mcp")
    footer_t2 = (
        f"<tr style='background:#eaf4fb;font-weight:700'>"
        f"<td colspan='3' style='text-align:right'>평균 (Mean)</td>"
        f"<td style='text-align:center'>{avg_sbrand:.4f}</td>"
        f"<td style='text-align:center'>{sbranch_avg_str}</td>"
        f"<td style='text-align:center'>{avg_sstat:.4f}</td>"
        f"<td style='text-align:center'>{avg_base:.4f}</td>"
        f"<td style='text-align:center'>{avg_conf_ext:.4f}</td>"
        f"<td style='text-align:center'>{avg_conf_llm:.4f}</td>"
        f"<td style='text-align:center'>{avg_conf_mcp:.4f}</td>"
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
  <strong>[제안 수식]</strong> &nbsp;
  conf_FINAL = <strong>w<sub>EXT</sub></strong> &times; conf_EXT
             + <strong>w<sub>LLM</sub></strong> &times; conf_LLM
             + <strong>w<sub>MCP</sub></strong> &times; conf_MCP
  &nbsp;&nbsp; (w = {_W_EXT:.2f} / {_W_LLM:.2f} / {_W_MCP:.2f})
  <br>
  <span style="color:#7f8c8d;font-size:0.9em">
  [Baseline] conf_FINAL = 0.5 &times; conf_MUM + 0.5 &times; conf_LLM
  &nbsp;(S_brand + S_branch + S_status, MCP 항 없음)
  </span>
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
    <div class="label">평균 conf_EXT (VLM 추출)</div>
    <div class="value">{sum(r.conf_ext for r in records)/total:.4f}</div>
    <div class="sub">LLM/VLM 1차 추출 기본 신뢰도</div>
  </div>
  <div class="stat-card" style="border-color:#8e44ad">
    <div class="label">평균 conf_MCP (외부 DB)</div>
    <div class="value">{sum(r.conf_mcp for r in records)/total:.4f}</div>
    <div class="sub">비즈노 / 국세청 일치 점수</div>
  </div>
  <div class="stat-card" style="border-color:#e74c3c">
    <div class="label">Hallucination 억제</div>
    <div class="value">{supp_hall} / {total_hall}</div>
    <div class="sub">의심 {total_hall}건 중 {supp_hall}건 MCP 억제</div>
  </div>
</div>

<!-- ─── Table 1: 이미지별 상세 결과 ────────────────────────────────────── -->
<h2>Table 1 &nbsp; 이미지별 상세 실험 결과 (제안 수식)</h2>
<p class="note">
  conf_EXT: VLM 1차 추출 기본 신뢰도 &nbsp;|&nbsp;
  conf_LLM: 문맥 타당성 (속성 완성도 + probabilistic fusion) &nbsp;|&nbsp;
  conf_MCP: 비즈노/국세청 외부 DB 일치 점수 &nbsp;|&nbsp;
  H: Hallucination 의심(🔴) / MCP 억제(✓)</p>
<table>
<thead>
<tr>
  <th>#</th>
  <th>이미지</th>
  <th>추출 상호명</th>
  <th>비즈노 매칭명</th>
  <th>conf_EXT</th>
  <th>conf_LLM</th>
  <th>conf_MCP</th>
  <th>confFINAL<br><small>제안</small></th>
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
  Baseline: conf_FINAL = 0.5&times;conf_MUM + 0.5&times;conf_LLM &nbsp;(S_brand+S_branch+S_status, MCP 항 없음)<br>
  제안: conf_FINAL = {_W_EXT}&times;conf_EXT + {_W_LLM}&times;conf_LLM + {_W_MCP}&times;conf_MCP
</p>
<table>
<thead>
<tr>
  <th rowspan="2">#</th>
  <th rowspan="2">이미지</th>
  <th rowspan="2">추출 상호명</th>
  <th colspan="4" style="background:#7f8c8d">Baseline (2항 — MCP 없음)</th>
  <th colspan="4" style="background:#16a085">제안 수식 (3항 — MCP 포함)</th>
  <th rowspan="2">Δ gain</th>
</tr>
<tr>
  <th style="background:#95a5a6">S_brand</th>
  <th style="background:#95a5a6">S_branch</th>
  <th style="background:#95a5a6">S_status</th>
  <th style="background:#95a5a6">confFINAL<br><small>baseline</small></th>
  <th style="background:#1abc9c">conf_EXT</th>
  <th style="background:#1abc9c">conf_LLM</th>
  <th style="background:#1abc9c">conf_MCP</th>
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