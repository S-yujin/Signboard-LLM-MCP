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
    GEMINI_MODEL: str = "gemini-2.5-flash"
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

    # ── 일반 ──────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./outputs"))
    PROMPTS_DIR: Path = Path(__file__).parent / "prompts"
    MCP_MAX_ITERATIONS: int = 10

    def validate(self) -> None:
        if not self.GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")


settings = Settings()