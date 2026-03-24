"""
utils/logging_utils.py
프로젝트 전역 로거 설정
"""
import logging
import sys
from config import settings


def get_logger(name: str) -> logging.Logger:
    """
    모듈 이름을 받아 설정된 로거를 반환합니다.
    LOG_LEVEL 환경변수로 레벨을 제어합니다.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    return logger