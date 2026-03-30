"""
config.py
환경변수 로드 및 전역 설정 관리
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


class Settings:
    # ── Google Gemini ─────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-3.1-flash-lite-preview"
    GEMINI_MAX_TOKENS: int = 1024

    # ── 비즈노 API (후보 검색) ────────────────────────
    # 가입: https://api.bizno.net/join
    # 무료: 1일 200건 / 상호명·사업자번호 검색 가능
    BIZNO_API_KEY: str = os.getenv("BIZNO_API_KEY", "")
    BIZNO_API_URL: str = "https://bizno.net/api/fapi"   # GET ?key=&q=&gb=3&type=json

    # ── 국세청 API (최종 상태 검증) ───────────────────
    # 발급: https://www.data.go.kr → '사업자등록정보 진위확인 및 상태조회'
    NTS_SERVICE_KEY: str = os.getenv("NTS_SERVICE_KEY", "")
    NTS_API_BASE: str = "https://api.odcloud.kr/api/nts-businessman/v1"

    # ── 카카오 로컬 API  ──────────────────────────────
    # 용도 1) 역지오코딩 : GPS좌표 → 행정구역 주소
    # 용도 2) POI 검색  : GPS 반경 R미터 내 장소 키워드 검색
    KAKAO_API_KEY: str = os.getenv("KAKAO_API_KEY", "")
    KAKAO_LOCAL_API_URL: str = "https://dapi.kakao.com/v2/local"

    # POI 검색 기본 반경(미터). 논문 수식의 R 파라미터 
    # 도심 밀집 상권 : 100~200m / 일반 주택가: 300~500m 권장
    POI_SEARCH_RADIUS_M: int = int(os.getenv("POI_SEARCH_RADIUS_M", "300"))
 
    # GPS 거리 감쇠 스케일 (미터). S_gps = exp(-d / scale).
    # 100m 기준: 100m → 0.37, 200m → 0.14, 500m → 0.007
    GPS_DECAY_SCALE_M: float = float(os.getenv("GPS_DECAY_SCALE_M", "100.0"))

    # ── 일반 ──────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./outputs"))
    PROMPTS_DIR: Path = Path(__file__).parent / "prompts"
    MCP_MAX_ITERATIONS: int = 10
 
    def validate(self) -> None:
        if not self.GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요."
            )
        if not self.KAKAO_API_KEY:
            # 경고만 출력 (GPS 없는 이미지는 카카오 없이도 동작 가능)
            import warnings
            warnings.warn(
                "KAKAO_API_KEY가 설정되지 않았습니다. "
                "역지오코딩 및 POI 검색이 비활성화됩니다.",
                stacklevel=2,
            )


settings = Settings()