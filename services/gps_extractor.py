"""
services/gps_extractor.py

이미지 EXIF에서 GPS 좌표를 추출하고 역지오코딩하여 행정구역 정보를 반환합니다.

변경 이력:
  - _reverse_geocode_kakao: api_key 하드코딩 제거 → config.settings.KAKAO_API_KEY 사용
  - KAKAO_API_KEY 미설정 시 Nominatim(OSM) fallback 자동 전환
  - extract_gps_coords(): 역지오코딩 없이 lat/lon만 빠르게 반환하는 경량 함수 유지
"""
from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from typing import Optional

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

logger = logging.getLogger(__name__)


@dataclass
class GpsResult:
    lat: float
    lon: float
    sido: str = ""            # 예: 부산광역시
    sigungu: str = ""         # 예: 부산진구
    eupmyeondong: str = ""    # 예: 서면동


# ──────────────────────────────────────────────────────────────────────────────
# EXIF 파싱
# ──────────────────────────────────────────────────────────────────────────────

def _parse_gps_tag(gps_info: dict) -> Optional[tuple[float, float]]:
    """EXIF GPS 태그 딕셔너리 → (lat, lon) 십진수 변환."""
    def to_decimal(vals, ref):
        d, m, s = [v.numerator / v.denominator for v in vals]
        dec = d + m / 60 + s / 3600
        return -dec if ref in ("S", "W") else dec

    try:
        lat = to_decimal(gps_info[2], gps_info[1])
        lon = to_decimal(gps_info[4], gps_info[3])
        return lat, lon
    except (KeyError, ZeroDivisionError):
        return None


def _read_exif_gps(image_path: str) -> Optional[dict]:
    """이미지 파일에서 GPS EXIF 딕셔너리를 읽어 반환합니다."""
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None
        raw = exif.get(34853, {})  # 34853 = GPSInfo 태그 번호
        if not raw:
            return None
        return {GPSTAGS.get(k, k): v for k, v in raw.items()}
    except Exception as e:
        logger.warning("[GPS] EXIF 읽기 실패: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────────────────────

def extract_gps(image_path: str) -> Optional[GpsResult]:
    """
    이미지 EXIF에서 GPS 정보를 추출하고 역지오코딩합니다.

    카카오 API 키가 설정되어 있으면 카카오 로컬 API로 역지오코딩합니다.
    키가 없으면 Nominatim(OpenStreetMap) fallback을 사용합니다.

    Returns:
        GpsResult(lat, lon, sido, sigungu, eupmyeondong) 또는 None
    """
    gps_raw = _read_exif_gps(image_path)
    if gps_raw is None:
        return None

    coords = _parse_gps_tag(gps_raw)
    if not coords:
        return None

    lat, lon = coords
    result = GpsResult(lat=lat, lon=lon)

    from config import settings
    if settings.KAKAO_API_KEY:
        try:
            _reverse_geocode_kakao(result)
        except Exception as e:
            logger.warning("[GPS] 카카오 역지오코딩 실패 → Nominatim fallback: %s", e)
            try:
                _reverse_geocode_nominatim(result)
            except Exception as e2:
                logger.warning("[GPS] Nominatim fallback도 실패: %s", e2)
    else:
        logger.debug("[GPS] KAKAO_API_KEY 미설정 → Nominatim fallback 사용")
        try:
            _reverse_geocode_nominatim(result)
        except Exception as e:
            logger.warning("[GPS] Nominatim 역지오코딩 실패: %s", e)

    logger.info(
        "[GPS] %.5f, %.5f → %s %s %s",
        lat, lon, result.sido, result.sigungu, result.eupmyeondong,
    )
    return result


def extract_gps_coords(image_path: str) -> Optional[tuple[float, float]]:
    """
    역지오코딩 없이 lat/lon 좌표만 빠르게 반환합니다.

    integrator의 GPS 거리 계산처럼 행정구역 정보가 필요 없고
    좌표만 필요한 호출자를 위한 경량 버전입니다.

    Returns:
        (lat, lon) 튜플 또는 None
    """
    gps_raw = _read_exif_gps(image_path)
    if gps_raw is None:
        return None
    coords = _parse_gps_tag(gps_raw)
    if coords:
        logger.debug("[GPS/경량] %.5f, %.5f", coords[0], coords[1])
    return coords


# ──────────────────────────────────────────────────────────────────────────────
# 역지오코딩 구현
# ──────────────────────────────────────────────────────────────────────────────

def _reverse_geocode_kakao(result: GpsResult) -> None:
    """
    카카오 로컬 API로 역지오코딩합니다.
    API 키는 config.settings.KAKAO_API_KEY에서 읽습니다.

    엔드포인트:
        GET /v2/local/geo/coord2address.json?x={lon}&y={lat}&input_coord=WGS84
    """
    from config import settings

    url = (
        f"{settings.KAKAO_LOCAL_API_URL}/geo/coord2address.json"
        f"?x={result.lon}&y={result.lat}&input_coord=WGS84"
    )
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"KakaoAK {settings.KAKAO_API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        data = json.loads(r.read())

    docs = data.get("documents", [])
    if not docs:
        logger.warning("[GPS] 카카오 역지오코딩 결과 없음 (lat=%.5f, lon=%.5f)", result.lat, result.lon)
        return

    addr = docs[0].get("address") or {}
    result.sido          = addr.get("region_1depth_name", "")
    result.sigungu       = addr.get("region_2depth_name", "")
    result.eupmyeondong  = addr.get("region_3depth_name", "")


def _reverse_geocode_nominatim(result: GpsResult) -> None:
    """
    Nominatim(OpenStreetMap) API로 역지오코딩합니다.
    카카오 키 미설정 시 fallback으로 사용됩니다.
    (정밀도가 카카오보다 낮을 수 있습니다.)
    """
    url = (
        f"https://nominatim.openstreetmap.org/reverse"
        f"?lat={result.lat}&lon={result.lon}&format=json&accept-language=ko"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "signboard-llm/1.0"})
    with urllib.request.urlopen(req, timeout=5) as r:
        data = json.loads(r.read())

    addr = data.get("address", {})
    result.sido          = addr.get("state", "")
    result.sigungu       = addr.get("county") or addr.get("city", "")
    result.eupmyeondong  = addr.get("suburb") or addr.get("quarter", "")