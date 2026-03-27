from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import urllib.request, json, logging

logger = logging.getLogger(__name__)

@dataclass
class GpsResult:
    lat: float
    lon: float
    sido: str = ""      # 부산광역시
    sigungu: str = ""   # 부산진구
    eupmyeondong: str = ""  # 서면동

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
        _reverse_geocode_kakao(result)  # 또는 _reverse_geocode_nominatim
        logger.info(f"[GPS] {lat:.5f},{lon:.5f} → {result.sigungu} {result.eupmyeondong}")
        return result
    except Exception as e:
        logger.warning(f"[GPS] 추출 실패: {e}")
        return None

def _reverse_geocode_kakao(result: GpsResult, api_key: str = "YOUR_KAKAO_KEY"):
    url = (f"https://dapi.kakao.com/v2/local/geo/coord2address.json"
           f"?x={result.lon}&y={result.lat}&input_coord=WGS84")
    req = urllib.request.Request(url, headers={"Authorization": f"KakaoAK {api_key}"})
    with urllib.request.urlopen(req, timeout=5) as r:
        data = json.loads(r.read())
    doc = data["documents"][0]["address"]
    result.sido = doc.get("region_1depth_name", "")
    result.sigungu = doc.get("region_2depth_name", "")
    result.eupmyeondong = doc.get("region_3depth_name", "")

def _reverse_geocode_nominatim(result: GpsResult):
    """카카오 키 없을 때 fallback (정밀도 낮음)"""
    url = (f"https://nominatim.openstreetmap.org/reverse"
           f"?lat={result.lat}&lon={result.lon}&format=json&accept-language=ko")
    req = urllib.request.Request(url, headers={"User-Agent": "signboard-llm/1.0"})
    with urllib.request.urlopen(req, timeout=5) as r:
        data = json.loads(r.read())
    addr = data.get("address", {})
    result.sido = addr.get("state", "")
    result.sigungu = addr.get("county") or addr.get("city", "")
    result.eupmyeondong = addr.get("suburb") or addr.get("quarter", "")