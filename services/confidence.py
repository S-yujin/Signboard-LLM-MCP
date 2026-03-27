"""
services/confidence.py

논문 제안 신뢰도 수식 구현:

    confFINAL = w_EXT * conf_EXT + w_LLM * conf_LLM + w_MCP * conf_MCP

각 항의 정의:
    conf_EXT  : VLM(Gemini)이 이미지에서 속성을 1차 추출할 때 반환한 기본 신뢰도
    conf_LLM  : Levenshtein 편집 거리를 시그모이드로 변환한 확률적 융합 점수
                (소스 논문 1의 Probabilistic Fusion 방식 구현)
    conf_MCP  : 국세청·Bizno.net 조회 결과와 추출 데이터 간의 일치 점수

가중치 기본값 (합계 = 1.0):
    w_EXT = 0.30
    w_LLM = 0.30
    w_MCP = 0.40  ← 외부 Ground Truth 검증이므로 가중치 최대

참고 논문:
    - 소스 논문 1: 확률적 융합(Probabilistic Fusion) + Levenshtein Distance
    - 소스 논문 2: LLM Verifier Agent 기반 Precision-Recall 향상
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# 가중치 상수
# ──────────────────────────────────────────────────────────────────────────────

W_EXT: float = 0.30
W_LLM: float = 0.30
W_MCP: float = 0.40

assert abs(W_EXT + W_LLM + W_MCP - 1.0) < 1e-9, "가중치 합계는 반드시 1.0이어야 합니다."


# ──────────────────────────────────────────────────────────────────────────────
# Levenshtein Distance
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
            curr[j] = prev[j - 1] if c1 == c2 else 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[-1]


def _normalize_name(name: str) -> str:
    """
    상호명 비교를 위한 정규화.

    간판 추출명과 API 후보명의 표현 차이를 줄여 Levenshtein 유사도를
    실질적인 브랜드 일치 여부에 집중시킵니다.

    처리 순서:
        1. 괄호 안 내용 제거  : "카페051(CAFE051) 정관모전점" → "카페051  정관모전점"
        2. 지점명 키워드 제거 : "~점", "~호점", "~지점", "~본점", "~지사" 등
        3. 공백·특수문자 제거 : 알파벳·한글·숫자만 남김
        4. 소문자 통일        : "CAFE" → "cafe"

    예)
        "CAFE 051"                    → "cafe051"
        "카페051(CAFE051) 정관모전점"  → "카페051"
        "CAFE051 언양읍성점"           → "cafe051"
        → levenshtein_similarity("cafe051", "카페051") = 0.57  (원본 0.30 대비 향상)
    """
    import re
    s = name.strip()
    # 1. 괄호 내용 제거 (영문 브랜드명 중복 표기 패턴)
    s = re.sub(r"[\(\（][^\)\）]*[\)\）]", "", s)
    # 2. 지점명 제거
    s = re.sub(r"[\s\u3000]*\S*(점|호점|지점|본점|지사|센터|직영점|가맹점|분점)$", "", s)
    # 3. 공백·특수문자 제거, 소문자
    s = re.sub(r"[^\w가-힣]", "", s).lower()
    # 4. 간판에서 자주 쓰이는 한글↔영문 동의어 통일 (conf_LLM 정확도 향상)
    _KO_EN_MAP = {
        "카페": "cafe", "베이커리": "bakery", "치킨": "chicken",
        "피자": "pizza", "버거": "burger", "분식": "bunsik",
        "마트": "mart", "편의점": "cvs", "약국": "pharmacy",
        "병원": "hospital", "헤어": "hair", "네일": "nail",
    }
    for ko, en in _KO_EN_MAP.items():
        s = s.replace(ko, en)
    return s.strip()


def levenshtein_similarity(s1: str, s2: str, normalize: bool = True) -> float:
    """
    편집 거리 기반 정규화 유사도 [0.0, 1.0].
    동일하면 1.0, 완전히 다르면 0.0.

    Args:
        s1, s2    : 비교할 두 문자열
        normalize : True면 _normalize_name() 전처리 후 비교 (기본값).
                    간판 추출명 ↔ API 후보명처럼 표현 형식이 다를 때 사용.
                    False면 원본 문자열 그대로 비교.
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
# 시그모이드 기반 확률적 융합 (Probabilistic Fusion)
# ──────────────────────────────────────────────────────────────────────────────

def sigmoid(x: float, k: float = 10.0, x0: float = 0.5) -> float:
    """
    일반화 시그모이드 함수.

        σ(x) = 1 / (1 + exp(-k * (x - x0)))

    Args:
        x  : 입력값 (0~1 범위의 원시 유사도 점수)
        k  : 기울기 (클수록 임계값 부근에서 급격히 변화, 기본 10.0)
        x0 : 변곡점 (기본 0.5)

    Returns:
        확률적 융합 점수 [0.0, 1.0]

    근거:
        Levenshtein 유사도는 선형 척도여서 0.6과 0.9의 차이가
        실제 매칭 가능성에서 비선형적으로 다름.
        시그모이드로 변환하면 고유사도 구간 신뢰도를 급격히 높이고
        저유사도 구간을 낮춰 확률적 해석이 가능해짐.
    """
    try:
        return 1.0 / (1.0 + math.exp(-k * (x - x0)))
    except OverflowError:
        return 0.0 if x < x0 else 1.0


def probabilistic_fusion(sim_score: float, k: float = 10.0, x0: float = 0.5) -> float:
    """
    Levenshtein 유사도를 시그모이드로 변환하여
    확률적 융합 점수를 반환합니다.

    소스 논문 1의 Probabilistic Fusion 구현:
        P(match | sim) = σ(sim; k, x0)

    Args:
        sim_score : levenshtein_similarity() 결과 [0~1]
        k         : 시그모이드 기울기 (기본 10.0)
        x0        : 변곡점 (기본 0.5)
    """
    return sigmoid(sim_score, k=k, x0=x0)


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ConfidenceInput:
    """신뢰도 계산에 필요한 입력값."""

    conf_ext: float = 0.0               # VLM 1차 추출 신뢰도 (0~1)
    extracted_name: str = ""            # VLM 추출 상호명
    candidate_name: str = ""            # MCP 조회 최상위 후보 상호명
    attribute_completeness: float = 0.0 # 추출 속성 완성도 (채워진 필드 / 전체)

    # 시그모이드 파라미터 (논문 재현 시 조정 가능)
    sigmoid_k: float = 10.0
    sigmoid_x0: float = 0.5

    conf_mcp: float = 0.0               # 외부 API 매칭 점수 (0~1)

    w_ext: Optional[float] = None
    w_llm: Optional[float] = None
    w_mcp: Optional[float] = None


@dataclass
class ConfidenceResult:
    """신뢰도 계산 최종 결과."""

    conf_ext: float = 0.0
    conf_llm: float = 0.0
    conf_mcp: float = 0.0
    conf_final: float = 0.0

    w_ext: float = W_EXT
    w_llm: float = W_LLM
    w_mcp: float = W_MCP

    # 논문 재현 / ablation용 중간값
    raw_levenshtein_similarity: float = 0.0
    probabilistic_fusion_score: float = 0.0
    attribute_completeness: float = 0.0

    def as_dict(self) -> dict:
        return {
            "conf_ext": round(self.conf_ext, 4),
            "conf_llm": round(self.conf_llm, 4),
            "conf_mcp": round(self.conf_mcp, 4),
            "conf_final": round(self.conf_final, 4),
            "weights": {
                "w_ext": round(self.w_ext, 4),
                "w_llm": round(self.w_llm, 4),
                "w_mcp": round(self.w_mcp, 4),
            },
            "debug": {
                "raw_levenshtein_similarity": round(self.raw_levenshtein_similarity, 4),
                "probabilistic_fusion_score": round(self.probabilistic_fusion_score, 4),
                "attribute_completeness": round(self.attribute_completeness, 4),
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# conf_LLM 계산
# ──────────────────────────────────────────────────────────────────────────────

def _compute_conf_llm(inp: ConfidenceInput) -> tuple[float, float, float]:
    """
    conf_LLM: 추출 결과의 문맥적 타당성 점수.

    산출 방법:
        1. raw_sim  = levenshtein_similarity(extracted_name, candidate_name)
        2. prob     = probabilistic_fusion(raw_sim)   ← 시그모이드 변환
        3. conf_LLM = (prob + attribute_completeness) / 2

    Returns:
        (conf_llm, raw_levenshtein_similarity, probabilistic_fusion_score)
    """
    ref = inp.candidate_name if inp.candidate_name else inp.extracted_name
    raw_sim = levenshtein_similarity(inp.extracted_name, ref)
    prob_fusion = probabilistic_fusion(raw_sim, k=inp.sigmoid_k, x0=inp.sigmoid_x0)
    completeness = max(0.0, min(1.0, inp.attribute_completeness))
    conf_llm = (prob_fusion + completeness) / 2.0
    return conf_llm, raw_sim, prob_fusion


# ──────────────────────────────────────────────────────────────────────────────
# 메인 계산 함수
# ──────────────────────────────────────────────────────────────────────────────

def compute_confidence(inp: ConfidenceInput) -> ConfidenceResult:
    """
    논문 제안 수식으로 최종 신뢰도를 계산합니다:

        confFINAL = w_EXT * conf_EXT + w_LLM * conf_LLM + w_MCP * conf_MCP

    conf_LLM은 시그모이드 기반 확률적 융합(Probabilistic Fusion)으로 산출됩니다.
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