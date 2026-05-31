"""
JSON-based storage for conversations and sessions.
Replaces PostgreSQL for conversation storage.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import uuid
import logging

logger = logging.getLogger(__name__)


class JSONStorage:
    """Manages JSON-based storage for conversations and sessions."""

    def __init__(self, storage_dir: str = "data/conversations"):
        """
        Initialize JSON storage.

        Args:
            storage_dir: Directory to store JSON files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_file = self.storage_dir / "sessions.json"
        self.conversations_dir = self.storage_dir / "conversations"
        self.conversations_dir.mkdir(exist_ok=True)

        # Initialize sessions file if not exists
        if not self.sessions_file.exists():
            self._save_json(self.sessions_file, {})

    def _save_json(self, filepath: Path, data: Any) -> None:
        """Save data to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _load_json(self, filepath: Path) -> Any:
        """Load data from JSON file."""
        if not filepath.exists():
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def create_session(
        self, user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new session.

        Args:
            user_id: Optional user identifier
            metadata: Optional session metadata

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        sessions = self._load_json(self.sessions_file) or {}

        sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id or "anonymous",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "message_count": 0,
        }

        self._save_json(self.sessions_file, sessions)

        # Create conversation file for this session
        conversation_file = self.conversations_dir / f"{session_id}.json"
        self._save_json(conversation_file, {"session_id": session_id, "messages": []})

        logger.info(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session data or None
        """
        sessions = self._load_json(self.sessions_file) or {}
        return sessions.get(session_id)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a message to a session.

        Args:
            session_id: Session identifier
            role: Message role (user/assistant)
            content: Message content
            metadata: Optional message metadata

        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        conversation_file = self.conversations_dir / f"{session_id}.json"

        conversation_data = self._load_json(conversation_file)
        if not conversation_data:
            conversation_data = {"session_id": session_id, "messages": []}

        message = {
            "message_id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

        conversation_data["messages"].append(message)
        self._save_json(conversation_file, conversation_data)

        # Update session
        sessions = self._load_json(self.sessions_file) or {}
        if session_id in sessions:
            sessions[session_id]["updated_at"] = datetime.now().isoformat()
            sessions[session_id]["message_count"] = len(conversation_data["messages"])
            self._save_json(self.sessions_file, sessions)

        logger.debug(f"Added message {message_id} to session {session_id}")
        return message_id

    def get_conversation_history(
        self, session_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return

        Returns:
            List of messages
        """
        conversation_file = self.conversations_dir / f"{session_id}.json"
        conversation_data = self._load_json(conversation_file)

        if not conversation_data:
            return []

        messages = conversation_data.get("messages", [])
        return messages[-limit:] if limit else messages

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its conversation history.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False otherwise
        """
        # Remove from sessions
        sessions = self._load_json(self.sessions_file) or {}
        if session_id in sessions:
            del sessions[session_id]
            self._save_json(self.sessions_file, sessions)

        # Remove conversation file
        conversation_file = self.conversations_dir / f"{session_id}.json"
        if conversation_file.exists():
            conversation_file.unlink()

        logger.info(f"Deleted session: {session_id}")
        return True

    def list_sessions(
        self, user_id: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        List sessions, optionally filtered by user.

        Args:
            user_id: Optional user filter
            limit: Maximum number of sessions

        Returns:
            List of sessions
        """
        sessions = self._load_json(self.sessions_file) or {}
        session_list = list(sessions.values())

        if user_id:
            session_list = [s for s in session_list if s.get("user_id") == user_id]

        # Sort by updated_at descending
        session_list.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        return session_list[:limit]

    def export_conversation(
        self, session_id: str, output_file: Optional[str] = None
    ) -> str:
        """
        Export conversation to a formatted JSON file.

        Args:
            session_id: Session identifier
            output_file: Optional output file path

        Returns:
            Path to exported file
        """
        if not output_file:
            output_file = f"conversation_export_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        session = self.get_session(session_id)
        conversation = self.get_conversation_history(session_id, limit=None)

        export_data = {
            "session": session,
            "conversation": conversation,
            "exported_at": datetime.now().isoformat(),
        }

        output_path = Path(output_file)
        self._save_json(output_path, export_data)

        logger.info(f"Exported conversation to: {output_path}")
        return str(output_path)


# Global storage instance
json_storage = JSONStorage()


def get_json_storage() -> JSONStorage:
    """Get the global JSON storage instance."""
    return json_storage
