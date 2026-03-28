"""
services/gps_extractor.py

이미지 EXIF에서 GPS 좌표를 추출하고 역지오코딩하여 행정구역 정보를 반환합니다.

변경 이력:
  - GpsResult: lat/lon 필드를 integrator._gps_bonus()가 직접 참조할 수 있도록 유지
  - extract_gps_coords(): lat/lon 튜플만 빠르게 반환하는 경량 함수 추가
    → 역지오코딩 API 호출 없이 integrator에서 GPS 거리 계산에만 쓸 때 사용
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
    sido: str = ""           # 부산광역시
    sigungu: str = ""        # 부산진구
    eupmyeondong: str = ""   # 서면동


def _parse_gps_tag(gps_info: dict) -> Optional[tuple[float, float]]:
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


def extract_gps(image_path: str) -> Optional[GpsResult]:
    """
    이미지 EXIF에서 GPS 정보를 추출하고 역지오코딩합니다.

    Returns:
        GpsResult (lat, lon, sido, sigungu, eupmyeondong) 또는 None
    """
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None
        gps_raw = {GPSTAGS.get(k, k): v for k, v in exif.get(34853, {}).items()}
        coords = _parse_gps_tag(gps_raw)
        if not coords:
            return None
        lat, lon = coords
        result = GpsResult(lat=lat, lon=lon)
        _reverse_geocode_kakao(result)
        logger.info("[GPS] %.5f,%.5f → %s %s", lat, lon, result.sigungu, result.eupmyeondong)
        return result
    except Exception as e:
        logger.warning("[GPS] 추출 실패: %s", e)
        return None


def extract_gps_coords(image_path: str) -> Optional[tuple[float, float]]:
    """
    역지오코딩 없이 lat/lon 좌표만 빠르게 반환합니다.

    integrator의 GPS 거리 계산(_gps_bonus)처럼 행정구역 정보가 필요 없고
    좌표만 필요한 호출자를 위한 경량 버전입니다.

    Returns:
        (lat, lon) 튜플 또는 None
    """
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None
        gps_raw = {GPSTAGS.get(k, k): v for k, v in exif.get(34853, {}).items()}
        coords = _parse_gps_tag(gps_raw)
        if coords:
            logger.debug("[GPS/경량] %.5f, %.5f", coords[0], coords[1])
        return coords
    except Exception as e:
        logger.warning("[GPS/경량] 좌표 추출 실패: %s", e)
        return None


def _reverse_geocode_kakao(result: GpsResult, api_key: str = "YOUR_KAKAO_KEY") -> None:
    """카카오 로컬 API로 역지오코딩합니다."""
    url = (
        f"https://dapi.kakao.com/v2/local/geo/coord2address.json"
        f"?x={result.lon}&y={result.lat}&input_coord=WGS84"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"KakaoAK {api_key}"})
    with urllib.request.urlopen(req, timeout=5) as r:
        data = json.loads(r.read())
    doc = data["documents"][0]["address"]
    result.sido = doc.get("region_1depth_name", "")
    result.sigungu = doc.get("region_2depth_name", "")
    result.eupmyeondong = doc.get("region_3depth_name", "")


def _reverse_geocode_nominatim(result: GpsResult) -> None:
    """카카오 키 없을 때 fallback (정밀도 낮음)."""
    url = (
        f"https://nominatim.openstreetmap.org/reverse"
        f"?lat={result.lat}&lon={result.lon}&format=json&accept-language=ko"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "signboard-llm/1.0"})
    with urllib.request.urlopen(req, timeout=5) as r:
        data = json.loads(r.read())
    addr = data.get("address", {})
    result.sido = addr.get("state", "")
    result.sigungu = addr.get("county") or addr.get("city", "")
    result.eupmyeondong = addr.get("suburb") or addr.get("quarter", "")