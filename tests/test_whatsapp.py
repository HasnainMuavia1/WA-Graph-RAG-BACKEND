"""Unit tests for the WhatsApp client's verification + signature logic."""

import hashlib
import hmac

import pytest

from integrations.whatsapp import WhatsAppClient


@pytest.fixture(autouse=True)
def _clear_whatsapp_env(monkeypatch):
    """Make tests hermetic — the real .env may inject WHATSAPP_* values that
    would otherwise override the empty constructor args we pass on purpose."""
    for var in (
        "WHATSAPP_ACCESS_TOKEN",
        "WHATSAPP_PHONE_NUMBER_ID",
        "WHATSAPP_APP_SECRET",
        "WHATSAPP_VERIFY_TOKEN",
        "WHATSAPP_API_VERSION",
    ):
        monkeypatch.delenv(var, raising=False)


def _client(**kw):
    defaults = dict(
        access_token="tok",
        phone_number_id="123",
        app_secret="secret",
        verify_token="verify-me",
    )
    defaults.update(kw)
    return WhatsAppClient(**defaults)


class TestVerifySubscription:
    def test_returns_challenge_on_match(self):
        c = _client()
        assert c.verify_subscription("subscribe", "verify-me", "CHAL") == "CHAL"

    def test_rejects_wrong_token(self):
        c = _client()
        assert c.verify_subscription("subscribe", "wrong", "CHAL") is None

    def test_rejects_wrong_mode(self):
        c = _client()
        assert c.verify_subscription("unsubscribe", "verify-me", "CHAL") is None


class TestVerifySignature:
    def test_accepts_valid_signature(self):
        c = _client()
        body = b'{"hello":"world"}'
        sig = hmac.new(b"secret", body, hashlib.sha256).hexdigest()
        assert c.verify_signature(body, f"sha256={sig}") is True

    def test_rejects_tampered_body(self):
        c = _client()
        body = b'{"hello":"world"}'
        sig = hmac.new(b"secret", body, hashlib.sha256).hexdigest()
        assert c.verify_signature(b'{"hello":"evil"}', f"sha256={sig}") is False

    def test_rejects_missing_header(self):
        c = _client()
        assert c.verify_signature(b"x", None) is False

    def test_skips_when_no_app_secret(self):
        c = _client(app_secret="")
        # Dev mode: no secret configured → verification is skipped (returns True).
        assert c.verify_signature(b"x", None) is True


class TestConfigured:
    def test_is_configured_true(self):
        assert _client().is_configured is True

    def test_is_configured_false_without_token(self):
        assert WhatsAppClient(access_token="", phone_number_id="").is_configured is False
