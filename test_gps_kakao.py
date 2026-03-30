import requests
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# ==========================================
# [설정] 본인의 카카오 REST API 키를 입력하세요
# ==========================================
KAKAO_REST_API_KEY = "여기에_32자리_영문숫자_조합을_입력하세요"

def get_decimal_from_dms(dms, ref):
    """EXIF DMS 튜플(도, 분, 초)을 실수 좌표로 변환"""
    # dms: (35.0, 9.0, 36.116999) 형태
    degrees = float(dms[0])
    minutes = float(dms[1]) / 60.0
    seconds = float(dms[2]) / 3600.0

    coordinate = degrees + minutes + seconds
    if ref in ['S', 'W']:
        return -coordinate
    return coordinate

def get_exif_location(image_path):
    """이미지에서 GPS 위도, 경도 추출 및 변환"""
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if not exif_data: return None, None

            gps_info = {}
            for tag, value in exif_data.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    for t in value:
                        sub_tag = GPSTAGS.get(t, t)
                        gps_info[sub_tag] = value[t]

            if "GPSLatitude" in gps_info:
                lat = get_decimal_from_dms(gps_info["GPSLatitude"], gps_info["GPSLatitudeRef"])
                lng = get_decimal_from_dms(gps_info["GPSLongitude"], gps_info["GPSLongitudeRef"])
                return lat, lng
    except Exception as e:
        print(f"❌ EXIF 추출 중 오류: {e}")
    return None, None

import requests
import re

# ⚠️ [주의] 여기에 'REST API 키' 라고 한글을 쓰거나 'RESTAPI=' 등을 붙이지 마세요.
# 오직 32자리의 영문+숫자 조합만 " " 사이에 넣으세요.

def search_kakao_poi(lat, lng, keyword="음식점"):
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    
    # 1. 공백만 제거 (정규식 필터링 제거 - 키에 특수문자가 섞일 경우 대비)
    clean_key = KAKAO_REST_API_KEY.strip()
    
    # 2. 헤더 구성
    headers = {
        "Authorization": f"KakaoAK {clean_key}"
    }
    
    params = {
        "query": keyword,
        "x": str(lng),
        "y": str(lat),
        "radius": 200,
        "sort": "distance"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        if response.status_code == 200:
            return response.json().get('documents', [])
        else:
            # 여기서 에러 메시지를 확인하면 문제를 정확히 알 수 있습니다.
            print(f"❌ API 응답 에러 [{response.status_code}]: {response.text}")
            return []
    except Exception as e:
        print(f"❌ 최종 요청 실패: {e}")
        return []

# --- 터미널 실행부 ---
if __name__ == "__main__":
    img_path = "./sample_data/20251211_202540.jpg"
    
    # 1. 좌표 변환 테스트
    lat, lng = get_exif_location(img_path)
    
    if lat and lng:
        print(f"✅ 변환된 실수 좌표: {lat}, {lng}")
        
        # 2. 카카오 API 연동 테스트 (검색어는 임의 지정)
        print(f"📡 카카오 API로 주변 정보 조회 중...")
        store_name = result.source_signboard.business_name
        stores = search_kakao_poi(lat, lng, store_name)
        
        if stores:
            print(f"🏆 주변에서 {len(stores)}개의 매장을 찾았습니다:")
            for s in stores[:5]: # 상위 5개 출력
                print(f"- {s['place_name']} ({s['distance']}m) | {s['address_name']}")
        else:
            print("⚠️ 주변 매장 정보를 가져오지 못했습니다. (API 키 확인 필요)")
    else:
        print("❌ 좌표 변환에 실패했습니다. EXIF 태그 구조를 확인하세요.")