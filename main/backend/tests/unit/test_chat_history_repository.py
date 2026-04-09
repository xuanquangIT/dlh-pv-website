import shutil
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.repositories.solar_ai_chat.history_repository import ChatHistoryRepository
from app.schemas.solar_ai_chat import ChatRole, ChatTopic, SourceMetadata


class ChatHistoryRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = Path(tempfile.mkdtemp(prefix="chat_history_unit_"))
        self.repo = ChatHistoryRepository(storage_dir=self._temp_dir)
        self.owner_user_id = "11111111-1111-1111-1111-111111111111"
        self.other_user_id = "22222222-2222-2222-2222-222222222222"

    def tearDown(self) -> None:
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def test_create_session_returns_summary(self) -> None:
        session = self.repo.create_session(
            role=ChatRole.DATA_ENGINEER,
            title="Test",
            owner_user_id=self.owner_user_id,
        )
        self.assertEqual(session.title, "Test")
        self.assertEqual(session.role, ChatRole.DATA_ENGINEER)
        self.assertEqual(session.message_count, 0)

    def test_list_sessions_returns_created_sessions(self) -> None:
        self.repo.create_session(role=ChatRole.ADMIN, title="A", owner_user_id=self.owner_user_id)
        self.repo.create_session(role=ChatRole.DATA_ENGINEER, title="B", owner_user_id=self.owner_user_id)
        sessions = self.repo.list_sessions(owner_user_id=self.owner_user_id)
        self.assertEqual(len(sessions), 2)

    def test_get_session_returns_detail_with_messages(self) -> None:
        session = self.repo.create_session(
            role=ChatRole.DATA_ENGINEER,
            title="Detail",
            owner_user_id=self.owner_user_id,
        )
        self.repo.add_message(session.session_id, sender="user", content="Hello")
        detail = self.repo.get_session(session_id=session.session_id, owner_user_id=self.owner_user_id)
        self.assertIsNotNone(detail)
        self.assertEqual(len(detail.messages), 1)
        self.assertEqual(detail.messages[0].sender, "user")

    def test_get_nonexistent_session_returns_none(self) -> None:
        self.assertIsNone(self.repo.get_session("nonexistent", owner_user_id=self.owner_user_id))

    def test_delete_session_removes_file(self) -> None:
        session = self.repo.create_session(
            role=ChatRole.DATA_ENGINEER,
            title="Delete",
            owner_user_id=self.owner_user_id,
        )
        self.assertTrue(self.repo.delete_session(session.session_id, owner_user_id=self.owner_user_id))
        self.assertIsNone(self.repo.get_session(session.session_id, owner_user_id=self.owner_user_id))

    def test_delete_nonexistent_returns_false(self) -> None:
        self.assertFalse(self.repo.delete_session("nonexistent", owner_user_id=self.owner_user_id))

    def test_add_message_with_topic_and_sources(self) -> None:
        session = self.repo.create_session(
            role=ChatRole.ADMIN,
            title="Sources",
            owner_user_id=self.owner_user_id,
        )
        sources = [SourceMetadata(layer="Gold", dataset="dim_facility")]
        msg = self.repo.add_message(
            session.session_id,
            sender="assistant",
            content="Answer",
            topic=ChatTopic.SYSTEM_OVERVIEW,
            sources=sources,
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.topic, ChatTopic.SYSTEM_OVERVIEW)
        self.assertEqual(len(msg.sources), 1)

    def test_get_recent_messages_limits_results(self) -> None:
        session = self.repo.create_session(
            role=ChatRole.DATA_ENGINEER,
            title="Limit",
            owner_user_id=self.owner_user_id,
        )
        for i in range(15):
            self.repo.add_message(session.session_id, sender="user", content=f"Msg {i}")
        recent = self.repo.get_recent_messages(session.session_id, limit=5)
        self.assertEqual(len(recent), 5)
        self.assertEqual(recent[0].content, "Msg 10")

    def test_fork_session_creates_copy(self) -> None:
        original = self.repo.create_session(
            role=ChatRole.ADMIN,
            title="Original",
            owner_user_id=self.owner_user_id,
        )
        self.repo.add_message(original.session_id, sender="user", content="Q1")
        self.repo.add_message(original.session_id, sender="assistant", content="A1")

        forked = self.repo.fork_session(
            source_session_id=original.session_id,
            new_title="Forked",
            new_role=ChatRole.DATA_ENGINEER,
            owner_user_id=self.owner_user_id,
        )
        self.assertIsNotNone(forked)
        self.assertEqual(forked.title, "Forked")
        self.assertEqual(forked.role, ChatRole.DATA_ENGINEER)
        self.assertEqual(forked.message_count, 2)

        forked_detail = self.repo.get_session(forked.session_id, owner_user_id=self.owner_user_id)
        self.assertEqual(len(forked_detail.messages), 2)
        self.assertNotEqual(
            forked_detail.messages[0].id,
            self.repo.get_session(original.session_id, owner_user_id=self.owner_user_id).messages[0].id,
        )

    def test_fork_nonexistent_returns_none(self) -> None:
        self.assertIsNone(
            self.repo.fork_session("fake", "title", owner_user_id=self.owner_user_id)
        )

    def test_session_id_sanitized(self) -> None:
        session = self.repo.create_session(
            role=ChatRole.DATA_ENGINEER,
            title="Sanitize",
            owner_user_id=self.owner_user_id,
        )
        path = self.repo._session_path(session.session_id)
        self.assertTrue(path.name.endswith(".json"))
        self.assertNotIn("..", str(path.name))

    def test_list_sessions_is_scoped_by_owner(self) -> None:
        self.repo.create_session(role=ChatRole.ADMIN, title="Mine", owner_user_id=self.owner_user_id)
        self.repo.create_session(role=ChatRole.ADMIN, title="Other", owner_user_id=self.other_user_id)

        sessions = self.repo.list_sessions(owner_user_id=self.owner_user_id)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].title, "Mine")

    def test_get_session_returns_none_for_other_owner(self) -> None:
        session = self.repo.create_session(
            role=ChatRole.ADMIN,
            title="Private",
            owner_user_id=self.owner_user_id,
        )

        session_detail = self.repo.get_session(
            session_id=session.session_id,
            owner_user_id=self.other_user_id,
        )
        self.assertIsNone(session_detail)


if __name__ == "__main__":
    unittest.main()
