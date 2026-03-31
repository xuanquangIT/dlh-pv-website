import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import create_app


class FrontendPagesIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(create_app())

    def test_home_page_renders_eight_module_cards(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

        html = response.text
        self.assertIn("One Home For 8 Working Modules", html)
        self.assertEqual(html.count('class="module-card"'), 8)

    def test_chatbot_test_page_renders_form(self) -> None:
        response = self.client.get("/solar-ai-chat/test")
        self.assertEqual(response.status_code, 200)

        html = response.text
        self.assertIn("Vietnamese Query Test Console", html)
        self.assertIn('id="chat-form"', html)
        self.assertIn('id="response-output"', html)


if __name__ == "__main__":
    unittest.main()
