import importlib
import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class EntryPointTests(unittest.TestCase):
    def test_package_exports_main(self):
        package = importlib.import_module("skills_recommender")
        self.assertTrue(callable(package.main))

    def test_python_m_module_exists(self):
        spec = importlib.util.find_spec("skills_recommender.__main__")
        self.assertIsNotNone(spec)

    @unittest.skipUnless(importlib.util.find_spec("faiss"), "faiss is not installed")
    def test_sdk_module_can_be_imported_when_vector_dependency_exists(self):
        sdk_module = importlib.import_module("skills_recommender.sdk")
        self.assertTrue(hasattr(sdk_module, "create_recommender"))


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "fastapi is not installed")
class CatalogSyncTests(unittest.TestCase):
    def test_replace_catalog_keeps_engine_in_sync(self):
        from skills_recommender.api.app import SkillRecommenderApp

        class DummyStorage:
            def __init__(self):
                self.saved_skills = None
                self.saved_vector_metadata = None

            def save_skills(self, skills):
                self.saved_skills = skills

            def get_vector_index_path(self):
                return "./data/test-index"

            def save_vector_state(self, skills, dimension, model_path):
                return {"skills_count": len(skills), "dimension": dimension, "model_path": model_path}

            def is_vector_index_consistent(self, skills, dimension, model_path):
                return False

            def vector_index_exists(self):
                return True

            def is_vector_metadata_consistent(self, skills):
                return False

            def save_vector_metadata(self, metadata):
                self.saved_vector_metadata = metadata

        class DummyVectorStore:
            def __init__(self):
                self.reset_called = 0
                self.saved_path = None
                self.metadata = []

            def reset(self):
                self.reset_called += 1

            def save(self, path):
                self.saved_path = path

            def is_empty(self):
                return False

            def load(self, path):
                self.saved_path = path

        class DummyRecommendationEngine:
            def __init__(self):
                self.catalog = None
                self.build_index_called = 0

            def build_index(self):
                self.build_index_called += 1

        app = SkillRecommenderApp.__new__(SkillRecommenderApp)
        app.storage = DummyStorage()
        app.vector_store = DummyVectorStore()
        app.recommendation_engine = DummyRecommendationEngine()

        app._replace_catalog(
            [{"id": "skill_1", "name": "Skill 1", "description": "demo"}],
            log_prefix="[Test]",
        )

        self.assertEqual(app.catalog.count(), 1)
        self.assertIs(app.recommendation_engine.catalog, app.catalog)
        self.assertEqual(app.vector_store.reset_called, 1)
        self.assertEqual(app.recommendation_engine.build_index_called, 1)
        self.assertEqual(app.vector_store.saved_path, "./data/test-index")


if __name__ == "__main__":
    unittest.main()
