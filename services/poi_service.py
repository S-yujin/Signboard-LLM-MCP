"""
services/poi_service.py

[Step 2-1: Geospatial Filtering — 논문 수정안]

카카오 키워드 장소 검색 API를 통해 이미지 GPS 좌표 반경 R미터 내의
POI(Point of Interest) 목록을 수집합니다.

변경 이력:
  - build_poi_coord_map(): 신규 추가
    → POI 상호명을 key, (lat, lon) 을 value로 하는 dict 반환
    → integrator._compute_confidence()에서 후보 상호명으로 POI 좌표를 조회하여
       S_gps(거리 감쇠) 계산에 활용
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

from config import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class POICandidate:
    """카카오 키워드 검색 결과 단일 POI."""
    place_name: str
    address_name: str
    road_address_name: str
    phone: str
    category_name: str
    lat: float
    lon: float
    distance_m: float = 0.0
    place_url: str = ""
    kakao_id: str = ""

    def to_hint_str(self) -> str:
        """verifier 에이전트 컨텍스트 힌트용 한 줄 요약."""
        addr = self.road_address_name or self.address_name
        dist = f"{int(self.distance_m)}m" if self.distance_m > 0 else "?"
        parts = [self.place_name]
        if addr:
            parts.append(addr)
        if self.phone:
            parts.append(self.phone)
        parts.append(f"거리:{dist}")
        return " / ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# 카카오 키워드 검색
# ──────────────────────────────────────────────────────────────────────────────

def search_poi_by_keyword(
    keyword: str,
    lat: float,
    lon: float,
    radius_m: int | None = None,
    max_results: int = 15,
) -> list[POICandidate]:
    """
    카카오 키워드 장소 검색 API로 반경 내 POI를 조회합니다.

    Returns:
        POICandidate 리스트 (distance_m 오름차순)
    """
    if not settings.KAKAO_API_KEY:
        logger.warning("[POI] KAKAO_API_KEY 미설정 — POI 검색 건너뜀")
        return []

    r = radius_m if radius_m is not None else settings.POI_SEARCH_RADIUS_M

    params: dict[str, str] = {
        "query":  keyword,
        "x":      str(lon),   # 카카오는 경도(x), 위도(y) 순서
        "y":      str(lat),
        "radius": str(min(r, 20000)),
        "size":   str(min(max_results, 15)),
        "sort":   "distance",
    }
    url = (
        f"{settings.KAKAO_LOCAL_API_URL}/search/keyword.json"
        f"?{urllib.parse.urlencode(params)}"
    )
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"KakaoAK {settings.KAKAO_API_KEY}"},
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as r_obj:
            data = json.loads(r_obj.read())
    except urllib.error.HTTPError as e:
        logger.warning("[POI] 카카오 API HTTP 오류 %d: %s", e.code, e.reason)
        return []
    except Exception as e:
        logger.warning("[POI] 카카오 API 호출 실패: %s", e)
        return []

    results: list[POICandidate] = []
    for doc in data.get("documents", []):
        try:
            results.append(POICandidate(
                place_name        = doc.get("place_name", ""),
                address_name      = doc.get("address_name", ""),
                road_address_name = doc.get("road_address_name", ""),
                phone             = doc.get("phone", ""),
                category_name     = doc.get("category_name", ""),
                lat               = float(doc.get("y", 0)),
                lon               = float(doc.get("x", 0)),
                distance_m        = float(doc.get("distance", 0)),
                place_url         = doc.get("place_url", ""),
                kakao_id          = doc.get("id", ""),
            ))
        except (ValueError, KeyError) as e:
            logger.debug("[POI] 문서 파싱 오류 (건너뜀): %s", e)

    logger.info(
        "[POI] '%s' 반경 %dm — %d건 검색됨",
        keyword, r, len(results),
    )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 필터링
# ──────────────────────────────────────────────────────────────────────────────

def filter_poi_by_name(
    candidates: list[POICandidate],
    extracted_name: str,
    top_k: int = 5,
    min_similarity: float = 0.25,
) -> list[POICandidate]:
    """
    추출된 상호명과 유사한 POI만 필터링하여 상위 top_k 개를 반환합니다.
    유사도 동점 시 거리 오름차순으로 2차 정렬합니다.
    """
    from services.confidence import levenshtein_similarity

    scored: list[tuple[float, POICandidate]] = []
    for poi in candidates:
        sim = levenshtein_similarity(extracted_name, poi.place_name, normalize=True)
        if sim >= min_similarity:
            scored.append((sim, poi))

    scored.sort(key=lambda x: (-x[0], x[1].distance_m))
    filtered = [poi for _, poi in scored[:top_k]]

    if filtered:
        logger.info(
            "[POI] 필터링 후 상위 %d건: %s",
            len(filtered),
            [f"{p.place_name}({p.distance_m:.0f}m)" for p in filtered],
        )
    else:
        logger.info("[POI] 유사한 POI 없음 (min_sim=%.2f)", min_similarity)

    return filtered


def build_poi_context_hints(pois: list[POICandidate]) -> list[str]:
    """verifier 에이전트 프롬프트용 힌트 문자열 리스트."""
    return [poi.to_hint_str() for poi in pois]


# ──────────────────────────────────────────────────────────────────────────────
# POI 좌표 맵 (신규) — integrator S_gps 계산에 사용
# ──────────────────────────────────────────────────────────────────────────────

def build_poi_coord_map(
    pois: list[POICandidate],
) -> dict[str, tuple[float, float]]:
    """
    POI 목록을 상호명 → (lat, lon) 딕셔너리로 변환합니다.

    integrator._compute_confidence() 에서 Bizno 후보 상호명으로
    POI 좌표를 조회할 때 사용합니다.

    키는 정규화하지 않은 원본 place_name입니다.
    integrator 쪽에서 levenshtein_similarity로 최근접 POI를 찾습니다.

    Returns:
        { place_name: (lat, lon), ... }
    """
    return {poi.place_name: (poi.lat, poi.lon) for poi in pois}


def lookup_nearest_poi_coords(
    candidate_name: str,
    poi_coord_map: dict[str, tuple[float, float]],
    min_similarity: float = 0.35,
) -> Optional[tuple[float, float]]:
    """
    후보 상호명과 가장 유사한 POI의 좌표를 반환합니다.

    Bizno 후보 상호명(예: "유가네 닭갈비")과 POI 명칭(예: "유가네닭갈비 서면롯데점")을
    levenshtein_similarity로 매칭합니다.

    Args:
        candidate_name  : Bizno 후보 상호명
        poi_coord_map   : build_poi_coord_map() 결과
        min_similarity  : 최소 유사도 (이 미만이면 None 반환)

    Returns:
        (lat, lon) 또는 None
    """
    if not poi_coord_map or not candidate_name:
        return None

    from services.confidence import levenshtein_similarity

    best_sim = 0.0
    best_coords: Optional[tuple[float, float]] = None

    for poi_name, coords in poi_coord_map.items():
        sim = levenshtein_similarity(candidate_name, poi_name, normalize=True)
        if sim > best_sim:
            best_sim = sim
            best_coords = coords

    if best_sim >= min_similarity:
        logger.debug(
            "[POI좌표] '%s' ← '%s' (sim=%.3f)",
            candidate_name,
            next(k for k, v in poi_coord_map.items() if v == best_coords),
            best_sim,
        )
        return best_coords

    return None