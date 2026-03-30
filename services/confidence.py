"""
services/confidence.py

논문 제안 신뢰도 수식 (GPS 항 제거 — 최종 3항 버전):

    conf_FINAL = w_BRAND * S_brand + w_BRANCH * S_branch + w_STATUS * S_status

각 항의 정의:
    S_brand  : 브랜드명(지점명 제거 후) Jaro-Winkler 유사도
               ← 간판 추출명과 비즈노 후보명의 핵심 브랜드 일치도를 측정
    S_branch : 지점명 일치 여부 점수
               ← 양쪽 모두 지점명 없으면 None → 가중치 0으로 재정규화
               ← 한쪽만 있으면 0.0 (지점 미확정 패널티)
               ← 양쪽 모두 있으면 Jaro-Winkler 직접 비교
    S_status : 국세청/비즈노 검증 상태 점수
               ← 국세청 verified + 계속사업자 → 1.0
               ← 국세청 verified + 휴·폐업자  → 0.5
               ← 비즈노만(미검증) + 계속사업자 → 0.3
               ← 그 외                         → 0.0

가중치 기본값 (합계 = 1.0):
    w_BRAND  = 0.50
    w_BRANCH = 0.20
    w_STATUS = 0.30

지점명 미사용 시:
    - S_branch 항 비활성(w_branch=0), 나머지 재정규화
    - GPS 항은 본 수식에서 완전 제거됨

하위 호환성:
    - levenshtein_distance(), levenshtein_similarity(), probabilistic_fusion()
      등은 evaluate.py에서 계속 사용 가능하도록 유지합니다.
    - ConfidenceInput, ConfidenceResult, compute_confidence() 도 유지합니다.
    - 신규 수식은 ConfidenceInputV2 / compute_confidence_v2() 로 제공합니다.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# 가중치 상수
# ──────────────────────────────────────────────────────────────────────────────

# ── 기존 수식 (conf_EXT + conf_LLM + conf_MCP) — evaluate.py 하위 호환 ──────
W_EXT: float = 0.30
W_LLM: float = 0.30
W_MCP: float = 0.40

# ── 제안 수식 v2 (S_brand + S_branch + S_status) — GPS 항 제거 ───────────────
W_BRAND:  float = 0.50
W_BRANCH: float = 0.20
W_STATUS: float = 0.30

assert abs(W_EXT + W_LLM + W_MCP - 1.0) < 1e-9, "기존 가중치 합계는 1.0이어야 합니다."
assert abs(W_BRAND + W_BRANCH + W_STATUS - 1.0) < 1e-9, "제안 가중치 합계는 1.0이어야 합니다."


# ──────────────────────────────────────────────────────────────────────────────
# 상호명 정규화
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """
    상호명 비교를 위한 정규화.

    처리 순서:
        1. 괄호 안 내용 제거  : "카페051(CAFE051)" → "카페051"
        2. 지점명 키워드 제거 : "~점", "~호점", "~지점", "~본점" 등
        3. 공백·특수문자 제거 : 알파벳·한글·숫자만 남김
        4. 소문자 통일        : "CAFE" → "cafe"
        5. 한글↔영문 동의어 통일 (브랜드명 혼용 표기 대응)
    """
    s = name.strip()
    # 1. 괄호 내용 제거
    s = re.sub(r"[\(\（][^\)\）]*[\)\）]", "", s)
    # 2. 지점명 제거
    s = re.sub(r"[\s\u3000]*\S*(점|호점|지점|본점|지사|센터|직영점|가맹점|분점)$", "", s)
    # 3. 공백·특수문자 제거, 소문자
    s = re.sub(r"[^\w가-힣]", "", s).lower()
    # 4. 한글↔영문 동의어 통일
    _KO_EN_MAP = {
        "카페": "cafe", "베이커리": "bakery", "치킨": "chicken",
        "피자": "pizza", "버거": "burger", "분식": "bunsik",
        "마트": "mart", "편의점": "cvs", "약국": "pharmacy",
        "병원": "hospital", "헤어": "hair", "네일": "nail",
    }
    for ko, en in _KO_EN_MAP.items():
        s = s.replace(ko, en)
    return s.strip()


def _extract_branch(name: str) -> str:
    """
    상호명에서 지점명 키워드를 추출합니다.
    지점명이 없으면 빈 문자열을 반환합니다.

    예)
        "돼지충전소 시민공원점" → "시민공원점"
        "스타벅스 서면1호점"   → "서면1호점"
        "홍길동순대국"         → ""
    """
    m = re.search(r"(\S*(?:점|호점|지점|본점|지사|센터|직영점|가맹점|분점))$", name.strip())
    return m.group(1) if m else ""


# ──────────────────────────────────────────────────────────────────────────────
# 편집 거리 기반 유사도 (하위 호환 유지)
# ──────────────────────────────────────────────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    """두 문자열 간 편집 거리(삽입·삭제·교체 최소 횟수)를 반환합니다."""
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i] + [0] * len(s2)
        for j, c2 in enumerate(s2, 1):
            curr[j] = (
                prev[j - 1] if c1 == c2
                else 1 + min(prev[j], curr[j - 1], prev[j - 1])
            )
        prev = curr
    return prev[-1]


def levenshtein_similarity(s1: str, s2: str, normalize: bool = True) -> float:
    """
    편집 거리 기반 정규화 유사도 [0.0, 1.0].

    Args:
        normalize : True면 _normalize_name() 전처리 후 비교 (기본값).
    """
    if not s1 and not s2:
        return 1.0
    a = _normalize_name(s1) if normalize else s1.strip().lower()
    b = _normalize_name(s2) if normalize else s2.strip().lower()
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(a, b)
    return 1.0 - dist / max_len


# ──────────────────────────────────────────────────────────────────────────────
# Jaro-Winkler 유사도 (논문 제안 수식 핵심)
# ──────────────────────────────────────────────────────────────────────────────

def jaro_similarity(s1: str, s2: str) -> float:
    """Jaro 유사도 [0.0, 1.0]."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i, c1 in enumerate(s1):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, len2)
        for j in range(lo, hi):
            if s2_matches[j] or c1 != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (
        matches / len1
        + matches / len2
        + (matches - transpositions / 2) / matches
    ) / 3.0


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """
    Jaro-Winkler 유사도 [0.0, 1.0].

    Jaro 유사도에 앞부분(prefix) 일치 가중치를 추가합니다.
    브랜드명 앞글자 일치 여부가 중요한 간판 매칭에 적합합니다.

    Args:
        p : prefix 가중치 계수 (표준값 0.1, 최대 0.25)
    """
    jaro = jaro_similarity(s1, s2)
    prefix_len = 0
    for c1, c2 in zip(s1[:4], s2[:4]):
        if c1 == c2:
            prefix_len += 1
        else:
            break
    return jaro + prefix_len * p * (1.0 - jaro)


def name_similarity(s1: str, s2: str, normalize: bool = True) -> float:
    """정규화 후 Jaro-Winkler 유사도."""
    a = _normalize_name(s1) if normalize else s1.strip().lower()
    b = _normalize_name(s2) if normalize else s2.strip().lower()
    return jaro_winkler_similarity(a, b)


# ──────────────────────────────────────────────────────────────────────────────
# 시그모이드 기반 확률적 융합 (기존 — 하위 호환 유지)
# ──────────────────────────────────────────────────────────────────────────────

def sigmoid(x: float, k: float = 10.0, x0: float = 0.5) -> float:
    """일반화 시그모이드 함수."""
    try:
        return 1.0 / (1.0 + math.exp(-k * (x - x0)))
    except OverflowError:
        return 0.0 if x < x0 else 1.0


def probabilistic_fusion(sim_score: float, k: float = 10.0, x0: float = 0.5) -> float:
    """Levenshtein 유사도를 시그모이드로 변환한 확률적 융합 점수."""
    return sigmoid(sim_score, k=k, x0=x0)


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 구조 — 기존 수식 (하위 호환 유지)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ConfidenceInput:
    """신뢰도 계산 입력값 (기존 수식용 — 하위 호환 유지)."""
    conf_ext: float = 0.0
    extracted_name: str = ""
    candidate_name: str = ""
    attribute_completeness: float = 0.0
    sigmoid_k: float = 10.0
    sigmoid_x0: float = 0.5
    conf_mcp: float = 0.0
    w_ext: Optional[float] = None
    w_llm: Optional[float] = None
    w_mcp: Optional[float] = None


@dataclass
class ConfidenceResult:
    """신뢰도 계산 결과 (기존 수식용 — 하위 호환 유지)."""
    conf_ext: float = 0.0
    conf_llm: float = 0.0
    conf_mcp: float = 0.0
    conf_final: float = 0.0
    w_ext: float = W_EXT
    w_llm: float = W_LLM
    w_mcp: float = W_MCP
    raw_levenshtein_similarity: float = 0.0
    probabilistic_fusion_score: float = 0.0
    attribute_completeness: float = 0.0

    def as_dict(self) -> dict:
        return {
            "conf_ext":   round(self.conf_ext, 4),
            "conf_llm":   round(self.conf_llm, 4),
            "conf_mcp":   round(self.conf_mcp, 4),
            "conf_final": round(self.conf_final, 4),
            "weights": {
                "w_ext": round(self.w_ext, 4),
                "w_llm": round(self.w_llm, 4),
                "w_mcp": round(self.w_mcp, 4),
            },
            "debug": {
                "raw_levenshtein_similarity": round(self.raw_levenshtein_similarity, 4),
                "probabilistic_fusion_score": round(self.probabilistic_fusion_score, 4),
                "attribute_completeness":     round(self.attribute_completeness, 4),
            },
        }


def _compute_conf_llm(inp: ConfidenceInput) -> tuple[float, float, float]:
    """conf_LLM: 시그모이드 확률적 융합 점수."""
    ref = inp.candidate_name if inp.candidate_name else inp.extracted_name
    raw_sim = levenshtein_similarity(inp.extracted_name, ref)
    prob_fusion = probabilistic_fusion(raw_sim, k=inp.sigmoid_k, x0=inp.sigmoid_x0)
    completeness = max(0.0, min(1.0, inp.attribute_completeness))
    conf_llm = (prob_fusion + completeness) / 2.0
    return conf_llm, raw_sim, prob_fusion


def compute_confidence(inp: ConfidenceInput) -> ConfidenceResult:
    """
    기존 수식으로 최종 신뢰도를 계산합니다 (하위 호환 유지):
        confFINAL = w_EXT * conf_EXT + w_LLM * conf_LLM + w_MCP * conf_MCP
    """
    w_ext = inp.w_ext if inp.w_ext is not None else W_EXT
    w_llm = inp.w_llm if inp.w_llm is not None else W_LLM
    w_mcp = inp.w_mcp if inp.w_mcp is not None else W_MCP

    total_w = w_ext + w_llm + w_mcp
    if total_w > 0:
        w_ext, w_llm, w_mcp = w_ext / total_w, w_llm / total_w, w_mcp / total_w

    conf_ext = max(0.0, min(1.0, inp.conf_ext))
    conf_mcp = max(0.0, min(1.0, inp.conf_mcp))
    conf_llm, raw_sim, prob_fusion = _compute_conf_llm(inp)

    conf_final = w_ext * conf_ext + w_llm * conf_llm + w_mcp * conf_mcp
    conf_final = max(0.0, min(1.0, conf_final))

    return ConfidenceResult(
        conf_ext=conf_ext,
        conf_llm=conf_llm,
        conf_mcp=conf_mcp,
        conf_final=conf_final,
        w_ext=w_ext,
        w_llm=w_llm,
        w_mcp=w_mcp,
        raw_levenshtein_similarity=raw_sim,
        probabilistic_fusion_score=prob_fusion,
        attribute_completeness=inp.attribute_completeness,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 제안 수식 v2 (GPS 항 완전 제거 — 최종 3항 버전)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ConfidenceInputV2:
    """
    제안 수식 v2 입력값.
    GPS 관련 필드를 완전 제거하고 3항(brand / branch / status)만 사용합니다.
    """
    extracted_name: str = ""          # VLM 추출 상호명
    candidate_name: str = ""          # 비즈노 후보 상호명
    business_status: str = "unknown"  # 사업자 상태 (계속사업자 / 휴업자 / 폐업자)
    status_verified: bool = False     # 국세청 API 검증 완료 여부
    # 가중치 오버라이드 (None이면 전역 상수 사용)
    w_brand:  Optional[float] = None
    w_branch: Optional[float] = None
    w_status: Optional[float] = None


@dataclass
class ConfidenceResultV2:
    """
    제안 수식 v2 계산 결과.
    브랜드 / 지점 / 상태 3항을 분리하여 각 요소의 기여를 투명하게 추적합니다.
    """
    s_brand:  float = 0.0
    s_branch: Optional[float] = None   # 지점명 미사용 시 None
    s_status: float = 0.0
    conf_final: float = 0.0

    w_brand:  float = W_BRAND
    w_branch: float = W_BRANCH
    w_status: float = W_STATUS

    # 디버그용 중간값
    jaro_winkler_raw: float = 0.0
    normalized_brand_a: str = ""
    normalized_brand_b: str = ""
    branch_a: str = ""
    branch_b: str = ""

    warnings: list = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def as_dict(self) -> dict:
        return {
            "s_brand":    round(self.s_brand, 4),
            "s_branch":   round(self.s_branch, 4) if self.s_branch is not None else None,
            "s_status":   round(self.s_status, 4),
            "conf_final": round(self.conf_final, 4),
            "weights": {
                "w_brand":  round(self.w_brand, 4),
                "w_branch": round(self.w_branch, 4),
                "w_status": round(self.w_status, 4),
            },
            "debug": {
                "jaro_winkler_raw":   round(self.jaro_winkler_raw, 4),
                "normalized_brand_a": self.normalized_brand_a,
                "normalized_brand_b": self.normalized_brand_b,
                "branch_a":           self.branch_a,
                "branch_b":           self.branch_b,
            },
            "warnings": self.warnings,
        }


def compute_confidence_v2(inp: ConfidenceInputV2) -> ConfidenceResultV2:
    """
    논문 제안 수식 v2 (GPS 항 완전 제거):

        conf_FINAL = w_BRAND * S_brand + w_BRANCH * S_branch + w_STATUS * S_status

    S_branch가 None(양쪽 지점명 없음)이면 w_branch=0으로 두고 나머지 재정규화.
    """
    w_brand  = inp.w_brand  if inp.w_brand  is not None else W_BRAND
    w_branch = inp.w_branch if inp.w_branch is not None else W_BRANCH
    w_status = inp.w_status if inp.w_status is not None else W_STATUS

    result_warnings: list[str] = []

    # ── S_brand: 브랜드명(지점명 제거) Jaro-Winkler ──────────────────────────
    norm_a = _normalize_name(inp.extracted_name)
    norm_b = _normalize_name(inp.candidate_name)
    s_brand = jaro_winkler_similarity(norm_a, norm_b)
    jw_raw  = jaro_winkler_similarity(
        inp.extracted_name.lower(),
        inp.candidate_name.lower(),
    )

    # ── S_branch: 지점명 일치 점수 ───────────────────────────────────────────
    branch_a = _extract_branch(inp.extracted_name)
    branch_b = _extract_branch(inp.candidate_name)

    s_branch: Optional[float]
    if not branch_a and not branch_b:
        # 양쪽 모두 지점명 없음 → 구분 불필요, 항 비활성
        s_branch = None
        w_branch = 0.0
    elif branch_a and branch_b:
        # 양쪽 모두 지점명 있음 → 직접 비교
        norm_ba = re.sub(r"[^\w가-힣]", "", branch_a).lower()
        norm_bb = re.sub(r"[^\w가-힣]", "", branch_b).lower()
        s_branch = 1.0 if norm_ba == norm_bb else jaro_winkler_similarity(norm_ba, norm_bb)
        if norm_ba != norm_bb:
            result_warnings.append("branch_mismatch")
    else:
        # 한쪽만 지점명 있음 → 지점 미확정 패널티
        s_branch = 0.0
        result_warnings.append("branch_not_verified")

    # ── S_status: 국세청 검증 상태 점수 ─────────────────────────────────────
    s_status = 0.0
    if inp.status_verified:
        s_status = 1.0 if inp.business_status == "계속사업자" else 0.5
    elif inp.business_status == "계속사업자":
        s_status = 0.3

    # ── 가중치 재정규화 (비활성 항 제외) ─────────────────────────────────────
    s_branch_val = s_branch if s_branch is not None else 0.0
    total_w = w_brand + w_branch + w_status
    if total_w > 0 and abs(total_w - 1.0) > 1e-9:
        w_brand  /= total_w
        w_branch /= total_w
        w_status /= total_w

    # ── 최종 합산 ────────────────────────────────────────────────────────────
    conf_final = (
        w_brand  * s_brand
        + w_branch * s_branch_val
        + w_status * s_status
    )
    conf_final = round(max(0.0, min(1.0, conf_final)), 4)

    return ConfidenceResultV2(
        s_brand=round(s_brand, 4),
        s_branch=round(s_branch, 4) if s_branch is not None else None,
        s_status=round(s_status, 4),
        conf_final=conf_final,
        w_brand=round(w_brand, 4),
        w_branch=round(w_branch, 4),
        w_status=round(w_status, 4),
        jaro_winkler_raw=round(jw_raw, 4),
        normalized_brand_a=norm_a,
        normalized_brand_b=norm_b,
        branch_a=branch_a,
        branch_b=branch_b,
        warnings=result_warnings,
    )