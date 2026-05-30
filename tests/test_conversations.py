"""Tests for conversation router helpers (WhatsApp id normalization)."""

import pytest

from agent.conversations_router import _normalize_wa_id
from fastapi import HTTPException


class TestNormalizeWaId:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("923453241015", "923453241015"),
            ("+92 345 324 1015", "923453241015"),
            ("+1-202-555-0173", "12025550173"),
            ("  923001234567  ", "923001234567"),
        ],
    )
    def test_normalizes_valid_numbers(self, raw, expected):
        assert _normalize_wa_id(raw) == expected

    @pytest.mark.parametrize("bad", ["", "abc", "12345", "++", "12345678901234567890"])
    def test_rejects_invalid_numbers(self, bad):
        with pytest.raises(HTTPException) as exc:
            _normalize_wa_id(bad)
        assert exc.value.status_code == 400
