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
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"          # 무료 티어 Vision 지원
    GEMINI_MAX_TOKENS: int = 1024

    # ── 국세청 사업자정보 API ──────────────────────────
    NTS_API_BASE: str = "https://api.odcloud.kr/api/nts-businessman/v1"
    NTS_SERVICE_KEY: str = os.getenv("NTS_SERVICE_KEY", "")

    # ── 사업자 검색 API ───────────────────────────────
    BUSINESS_SEARCH_API_URL: str = os.getenv(
        "BUSINESS_SEARCH_API_URL", "https://mock-search-api.example.com/v1"
    )
    BUSINESS_SEARCH_API_KEY: str = os.getenv("BUSINESS_SEARCH_API_KEY", "")

    # ── 일반 ──────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./outputs"))
    PROMPTS_DIR: Path = Path(__file__).parent / "prompts"
    MCP_MAX_ITERATIONS: int = 10

    def validate(self) -> None:
        if not self.GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")


settings = Settings()