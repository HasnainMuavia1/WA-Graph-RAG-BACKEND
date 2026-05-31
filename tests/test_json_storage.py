"""
Unit tests for JSONStorage (file-based session/conversation persistence).
"""

import os
import json

import pytest

from agent.json_storage import JSONStorage


@pytest.fixture()
def storage(tmp_path):
    """Fresh JSONStorage backed by a temporary directory."""
    return JSONStorage(storage_dir=str(tmp_path / "conversations"))


class TestSessionLifecycle:
    def test_create_session_returns_uuid_string(self, storage):
        sid = storage.create_session()
        assert isinstance(sid, str)
        assert len(sid) == 36  # UUID4 format

    def test_get_session_after_create(self, storage):
        sid = storage.create_session(user_id="user_1")
        session = storage.get_session(sid)
        assert session is not None
        assert session["session_id"] == sid
        assert session["user_id"] == "user_1"

    def test_get_nonexistent_session_returns_none(self, storage):
        assert storage.get_session("no-such-id") is None

    def test_session_metadata_is_stored(self, storage):
        meta = {"project": "rag", "version": 2}
        sid = storage.create_session(metadata=meta)
        session = storage.get_session(sid)
        assert session["metadata"] == meta

    def test_anonymous_user_default(self, storage):
        sid = storage.create_session()
        session = storage.get_session(sid)
        assert session["user_id"] == "anonymous"

    def test_message_count_starts_at_zero(self, storage):
        sid = storage.create_session()
        session = storage.get_session(sid)
        assert session["message_count"] == 0

    def test_delete_session_removes_data(self, storage):
        sid = storage.create_session()
        assert storage.delete_session(sid) is True
        assert storage.get_session(sid) is None

    def test_delete_nonexistent_session(self, storage):
        result = storage.delete_session("ghost-session")
        assert result is True  # idempotent


class TestMessageOperations:
    def test_add_message_returns_id(self, storage):
        sid = storage.create_session()
        mid = storage.add_message(sid, "user", "Hello!")
        assert isinstance(mid, str) and len(mid) == 36

    def test_messages_are_retrievable(self, storage):
        sid = storage.create_session()
        storage.add_message(sid, "user", "Question")
        storage.add_message(sid, "assistant", "Answer")
        history = storage.get_conversation_history(sid)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_history_respects_limit(self, storage):
        sid = storage.create_session()
        for i in range(10):
            storage.add_message(sid, "user", f"Message {i}")
        history = storage.get_conversation_history(sid, limit=3)
        assert len(history) == 3

    def test_history_returns_last_n_messages(self, storage):
        sid = storage.create_session()
        for i in range(5):
            storage.add_message(sid, "user", f"msg-{i}")
        history = storage.get_conversation_history(sid, limit=2)
        assert history[-1]["content"] == "msg-4"
        assert history[-2]["content"] == "msg-3"

    def test_empty_history_for_new_session(self, storage):
        sid = storage.create_session()
        history = storage.get_conversation_history(sid)
        assert history == []

    def test_history_nonexistent_session(self, storage):
        history = storage.get_conversation_history("ghost")
        assert history == []

    def test_message_count_increments(self, storage):
        sid = storage.create_session()
        storage.add_message(sid, "user", "Hi")
        storage.add_message(sid, "assistant", "Hello")
        session = storage.get_session(sid)
        assert session["message_count"] == 2

    def test_message_metadata_stored(self, storage):
        sid = storage.create_session()
        meta = {"tool_calls": 3}
        storage.add_message(sid, "assistant", "response", metadata=meta)
        history = storage.get_conversation_history(sid)
        assert history[0]["metadata"] == meta


class TestListSessions:
    def test_list_all_sessions(self, storage):
        for _ in range(3):
            storage.create_session()
        sessions = storage.list_sessions()
        assert len(sessions) == 3

    def test_filter_by_user_id(self, storage):
        storage.create_session(user_id="alice")
        storage.create_session(user_id="bob")
        storage.create_session(user_id="alice")
        alice_sessions = storage.list_sessions(user_id="alice")
        assert len(alice_sessions) == 2
        assert all(s["user_id"] == "alice" for s in alice_sessions)

    def test_list_respects_limit(self, storage):
        for _ in range(10):
            storage.create_session()
        sessions = storage.list_sessions(limit=4)
        assert len(sessions) == 4


class TestExportConversation:
    def test_export_creates_file(self, storage, tmp_path):
        sid = storage.create_session(user_id="exporter")
        storage.add_message(sid, "user", "Hello")
        storage.add_message(sid, "assistant", "Hi there")

        out_file = str(tmp_path / "export.json")
        result_path = storage.export_conversation(sid, output_file=out_file)

        assert os.path.exists(result_path)
        with open(result_path) as fh:
            data = json.load(fh)
        assert "session" in data
        assert "conversation" in data
        assert len(data["conversation"]) == 2

    def test_export_default_filename(self, storage, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sid = storage.create_session()
        path = storage.export_conversation(sid)
        assert os.path.exists(path)
