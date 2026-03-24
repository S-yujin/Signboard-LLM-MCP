"""
config.py
환경변수 로드 및 전역 설정 관리
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 기준으로 .env 로드
load_dotenv(Path(__file__).parent / ".env")


class Settings:
    def __init__(self) -> None:
        # ── OpenAI ────────────────────────────────────────
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

        # 모델 선택 (하나만 주석 해제)
        # Vision 지원 + 안정적인 성능 (기본 추천)
        self.LLM_MODEL: str = "gpt-4o"
        # 저비용, 빠른 속도 (이미지 인식 정확도 다소 낮음)
        # self.LLM_MODEL: str = "gpt-4o-mini"
        # 최신 모델, Vision 지원, 텍스트 추론 향상
        # self.LLM_MODEL: str = "gpt-4.1"

        self.OPENAI_MAX_TOKENS: int = 1024

        # ── 국세청 사업자정보 API ──────────────────────────
        self.NTS_API_BASE: str = "https://api.odcloud.kr/api/nts-businessman/v1"
        self.NTS_SERVICE_KEY: str = os.getenv("NTS_SERVICE_KEY", "")

        # ── 사업자 검색 API ───────────────────────────────
        self.BUSINESS_SEARCH_API_URL: str = os.getenv(
            "BUSINESS_SEARCH_API_URL", "https://mock-search-api.example.com/v1"
        )
        self.BUSINESS_SEARCH_API_KEY: str = os.getenv("BUSINESS_SEARCH_API_KEY", "")

        # ── 일반 ──────────────────────────────────────────
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./outputs"))
        self.PROMPTS_DIR: Path = Path(__file__).parent / "prompts"

        # MCP 에이전트 최대 루프 횟수 (무한루프 방지)
        self.MCP_MAX_ITERATIONS: int = 10

    def validate(self) -> None:
        """필수 키가 설정되어 있는지 확인합니다."""
        if not self.OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요."
            )


settings = Settings()