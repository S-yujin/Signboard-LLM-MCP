"""
utils/phone_utils.py
전화번호 정규화 및 유효성 검사 유틸리티
"""
import re
from typing import Optional


def normalize_phone(raw: Optional[str]) -> Optional[str]:
    """
    원시 전화번호 문자열을 한국 표준 형식(지역번호-국번-번호)으로 정규화합니다.
    숫자만 추출한 뒤 자리수/접두사 규칙으로 분리합니다.

    >>> normalize_phone("02 1234 5678")
    '02-1234-5678'
    >>> normalize_phone("01012345678")
    '010-1234-5678'
    >>> normalize_phone("0312345678")
    '031-234-5678'
    >>> normalize_phone(None)
    None
    """
    if not raw:
        return None

    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None

    # ── 서울 (02): 9~10자리 ────────────────────────────────────────────────────
    if digits.startswith("02"):
        body = digits[2:]          # 7자리 또는 8자리
        if len(body) == 7:
            return f"02-{body[:3]}-{body[3:]}"
        if len(body) == 8:
            return f"02-{body[:4]}-{body[4:]}"

    # ── 휴대폰 (010/011/016/017/018/019): 10~11자리 ───────────────────────────
    if re.match(r"01[016789]", digits):
        body = digits[3:]
        if len(body) == 7:
            return f"{digits[:3]}-{body[:3]}-{body[3:]}"
        if len(body) == 8:
            return f"{digits[:3]}-{body[:4]}-{body[4:]}"

    # ── 일반 지역번호 (03X/04X/05X/06X): 10~11자리 ───────────────────────────
    if re.match(r"0[3-6]\d", digits):
        area = digits[:3]
        body = digits[3:]
        if len(body) == 7:
            return f"{area}-{body[:3]}-{body[3:]}"
        if len(body) == 8:
            return f"{area}-{body[:4]}-{body[4:]}"

    # 패턴 매칭 실패 — 숫자 그대로 반환
    return digits


def is_valid_korean_phone(phone: Optional[str]) -> bool:
    """
    정규화된 한국 전화번호 형식인지 검증합니다.

    >>> is_valid_korean_phone("02-1234-5678")
    True
    >>> is_valid_korean_phone("010-1234-5678")
    True
    >>> is_valid_korean_phone("invalid")
    False
    """
    if not phone:
        return False
    pattern = r"^(02|0\d{2})-\d{3,4}-\d{4}$"
    return bool(re.match(pattern, phone))


def strip_hyphens(phone: Optional[str]) -> Optional[str]:
    """국세청 API 등 하이픈 없는 형식이 필요할 때 사용합니다."""
    if not phone:
        return None
    return re.sub(r"[^\d]", "", phone)