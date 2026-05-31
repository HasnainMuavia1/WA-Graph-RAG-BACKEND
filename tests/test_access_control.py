"""
Unit tests for access-control logic.
"""



from agent.access_control import (
    assign_access_level,
    can_access_document,
    get_user_access_filter,
)


class TestAssignAccessLevel:
    def test_private_bucket_returns_private(self):
        assert assign_access_level("private") == "private"

    def test_public_bucket_returns_public(self):
        assert assign_access_level("public") == "public"

    def test_unknown_bucket_type_returns_public(self):
        assert assign_access_level("other") == "public"


class TestCanAccessDocument:
    def test_public_doc_accessible_to_anyone(self):
        assert can_access_document(None, "public") is True
        assert can_access_document("random_user", "public") is True

    def test_private_doc_blocked_for_anonymous(self):
        assert can_access_document(None, "private") is False

    def test_private_doc_accessible_to_private_user(self):
        assert can_access_document("private_user", "private") is True

    def test_private_doc_blocked_for_unknown_user(self):
        assert can_access_document("stranger", "private") is False

    def test_admin_can_access_private_docs(self):
        assert can_access_document("admin_user", "private") is True

    def test_admin_can_access_public_docs(self):
        assert can_access_document("admin_user", "public") is True

    def test_unknown_user_cannot_access_private(self):
        assert can_access_document("hacker", "private") is False


class TestGetUserAccessFilter:
    def test_anonymous_gets_public_filter(self):
        assert get_user_access_filter(None) == "public"

    def test_admin_gets_no_filter(self):
        assert get_user_access_filter("admin_user") is None

    def test_private_user_gets_all_filter(self):
        assert get_user_access_filter("private_user") == "all"

    def test_regular_user_gets_public_filter(self):
        assert get_user_access_filter("random_joe") == "public"

    def test_empty_string_user_gets_public_filter(self):
        assert get_user_access_filter("") == "public"

    def test_custom_admin_env(self, monkeypatch):
        monkeypatch.setenv("ADMIN_USERS", "super_admin")
        assert get_user_access_filter("super_admin") is None

    def test_custom_private_user_env(self, monkeypatch):
        monkeypatch.setenv("PRIVATE_DOCUMENT_USERS", "vip_user")
        assert get_user_access_filter("vip_user") == "all"
