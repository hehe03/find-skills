import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills_recommender.config import Settings
from skills_recommender.monitoring import FeedbackData, FeedbackStore, LangfuseTracer, MonitoringService


class ConfigTests(unittest.TestCase):
    def test_from_yaml_reads_nested_settings(self):
        test_root_base = PROJECT_ROOT / ".test_tmp"
        test_root_base.mkdir(exist_ok=True)
        temp_dir = test_root_base / f"config_{uuid.uuid4().hex}"
        temp_dir.mkdir()
        try:
            config_path = temp_dir / "config.yaml"
            config_path.write_text(
                """
app:
  name: "Custom Skill Recommender"
storage:
  json_settings:
    enabled: false
recommendation:
  recall:
    keyword:
      enabled: false
      top_k: 12
server:
  port: 9000
""".strip(),
                encoding="utf-8",
            )

            settings = Settings.from_yaml(str(config_path))

            self.assertEqual(settings.app.name, "Custom Skill Recommender")
            self.assertFalse(settings.storage.json_settings.enabled)
            self.assertFalse(settings.recommendation.recall.keyword.enabled)
            self.assertEqual(settings.recommendation.recall.keyword.top_k, 12)
            self.assertEqual(settings.server.port, 9000)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_from_env_overrides_provider_api_keys(self):
        fake_settings = Settings(
            llm={
                "default": "openai",
                "providers": {
                    "openai": {"api_key": None, "model": "gpt-4o"},
                    "alicloud": {"api_key": None, "model": "qwen"},
                },
            }
        )

        with mock.patch.object(Settings, "from_yaml", return_value=fake_settings):
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "open-key", "DASHSCOPE_API_KEY": "dash-key"}, clear=False):
                settings = Settings.from_env()

        self.assertEqual(settings.llm.providers["openai"].api_key, "open-key")
        self.assertEqual(settings.llm.providers["alicloud"].api_key, "dash-key")


class MonitoringTests(unittest.TestCase):
    def test_feedback_store_aggregates_ratings(self):
        store = FeedbackStore()
        store.add_feedback(FeedbackData(recommendation_id="r1", rating=5))
        store.add_feedback(FeedbackData(recommendation_id="r2", rating=1))

        stats = store.get_feedback_stats()

        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["avg_rating"], 3.0)
        self.assertEqual(stats["by_rating"][1], 1)
        self.assertEqual(stats["by_rating"][5], 1)

    def test_monitoring_service_records_feedback(self):
        tracer = LangfuseTracer(enabled=False)
        store = FeedbackStore()
        service = MonitoringService(tracer, store)

        service.record_feedback(FeedbackData(recommendation_id="r1", rating=4, selected_skill="skill_api"))

        self.assertEqual(len(store.get_all_feedbacks()), 1)
        self.assertEqual(store.get_all_feedbacks()[0].selected_skill, "skill_api")


if __name__ == "__main__":
    unittest.main()
