import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills_recommender.catalog import SkillSpec
from skills_recommender.llm import ChatResponse
from skills_recommender.storage import SkillStorage


SKILL_MD = """---
name: API Builder
description: Build API endpoints
---

# API Builder

Help create backend APIs quickly.
"""


class SkillStorageTests(unittest.TestCase):
    def setUp(self):
        self.test_root_base = PROJECT_ROOT / ".test_tmp"
        self.test_root_base.mkdir(exist_ok=True)
        self.root = self.test_root_base / f"storage_{uuid.uuid4().hex}"
        self.root.mkdir()
        self.hub_dir = self.root / "skills-hub"
        self.hub_dir.mkdir()
        (self.hub_dir / "api-builder").mkdir()
        (self.hub_dir / "api-builder" / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
        (self.hub_dir / "invalid-dir").mkdir()

        self.json_path = self.root / "data" / "skills.json"
        self.db_path = self.root / "data" / "skills.db"
        self.vector_path = self.root / "data" / "faiss_index"
        self.storage = SkillStorage(
            json_enabled=True,
            json_path=str(self.json_path),
            db_enabled=True,
            db_path=str(self.db_path),
            vector_enabled=True,
            vector_index_path=str(self.vector_path),
        )

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_get_hub_skills_reads_skill_markdown(self):
        skills = self.storage.get_hub_skills(str(self.hub_dir))

        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0]["id"], "hub_api-builder")
        self.assertEqual(skills[0]["name"], "API Builder")
        self.assertIn("api builder", [value.lower() for value in skills[0]["category"]])
        self.assertTrue(skills[0]["capabilities"])
        self.assertNotEqual(skills[0]["capabilities"], ["imported"])

    def test_import_skills_skips_existing_ids(self):
        result = self.storage.import_skills(
            skills_hub_path=str(self.hub_dir),
            existing_ids=["hub_api-builder"],
        )

        self.assertEqual(result["status"], "skipped")

    def test_update_all_skills_returns_updated_items(self):
        result = self.storage.update_all_skills(str(self.hub_dir))

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["updated_count"], 1)
        self.assertEqual(result["updated_skills"][0]["id"], "hub_api-builder")

    def test_save_vector_state_and_consistency_check(self):
        skills = [
            SkillSpec(
                id="skill_api",
                name="API Builder",
                description="Build API endpoints",
                enhanced_description="Create production-ready API endpoints",
            )
        ]

        index_file = Path(f"{self.vector_path}.index")
        meta_file = Path(f"{self.vector_path}.meta")
        index_file.parent.mkdir(parents=True, exist_ok=True)
        index_file.write_text("placeholder", encoding="utf-8")
        meta_file.write_text("placeholder", encoding="utf-8")

        self.storage.save_vector_state(skills, dimension=768, model_path="model-a")

        self.assertTrue(self.storage.is_vector_index_consistent(skills, dimension=768, model_path="model-a"))
        self.assertFalse(self.storage.is_vector_index_consistent(skills, dimension=1024, model_path="model-a"))

    def test_save_vector_metadata_and_check_consistency(self):
        skills = [
            SkillSpec(
                id="skill_api",
                name="API Builder",
                description="Build API endpoints",
                enhanced_description="Create production-ready API endpoints",
                usage_stats={"times_recommended": 3, "times_selected": 1},
            )
        ]

        index_file = Path(f"{self.vector_path}.index")
        index_file.parent.mkdir(parents=True, exist_ok=True)
        index_file.write_text("placeholder", encoding="utf-8")

        self.storage.save_vector_metadata(skills)

        self.assertTrue(self.storage.is_vector_metadata_consistent(skills))
        loaded = self.storage.load_vector_metadata()
        self.assertEqual(loaded[0]["usage_stats"]["times_recommended"], 3)

    def test_save_feedback_and_get_feedback_stats(self):
        from skills_recommender.monitoring import FeedbackData

        self.storage.save_feedback(FeedbackData(recommendation_id="rec-1", rating=5))
        self.storage.save_feedback(FeedbackData(recommendation_id="rec-2", rating=3))

        stats = self.storage.get_feedback_stats()

        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["avg_rating"], 4.0)
        self.assertEqual(stats["by_rating"][5], 1)

    def test_save_and_load_skills_json_first_then_database(self):
        skills = [
            SkillSpec(
                id="skill_api",
                name="API Builder",
                description="Build API endpoints",
                category=["code_generation"],
            )
        ]
        self.storage.save_skills(skills)

        loaded = self.storage.load_skills()

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["id"], "skill_api")

        data = json.loads(self.json_path.read_text(encoding="utf-8"))
        self.assertEqual(data["skills"][0]["name"], "API Builder")

    def test_load_skills_falls_back_to_database_when_json_is_invalid(self):
        skills = [SkillSpec(id="skill_db", name="DB Skill", description="from db")]
        self.storage.save_to_database(skills)
        self.json_path.write_text("{invalid json", encoding="utf-8")

        loaded = self.storage.load_skills()

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["id"], "skill_db")

    def test_query_skills_from_database_filters_by_keyword(self):
        self.storage.save_to_database(
            [
                SkillSpec(
                    id="skill_api",
                    name="API Builder",
                    description="Build API endpoints",
                    enhanced_description="Create backend interfaces quickly",
                ),
                SkillSpec(id="skill_image", name="Image Helper", description="Edit pictures"),
            ]
        )

        results = self.storage.query_skills_from_database(keyword="backend")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "skill_api")

    def test_init_database_migrates_legacy_skills_table(self):
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE skills (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT,
                capabilities TEXT,
                version TEXT,
                dependencies TEXT,
                author TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        conn.commit()
        conn.close()

        self.storage.save_to_database(
            [
                SkillSpec(
                    id="skill_api",
                    name="API Builder",
                    description="Build API endpoints",
                    enhanced_description="Create backend APIs efficiently",
                )
            ]
        )

        conn = sqlite3.connect(self.db_path)
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(skills)").fetchall()
        }
        row = conn.execute(
            "SELECT enhanced_description, input_schema, output_schema, usage_stats FROM skills WHERE id = ?",
            ("skill_api",),
        ).fetchone()
        conn.close()

        self.assertIn("enhanced_description", columns)
        self.assertIn("input_schema", columns)
        self.assertIn("output_schema", columns)
        self.assertIn("usage_stats", columns)
        self.assertEqual(row[0], "Create backend APIs efficiently")

    def test_vector_index_exists_checks_index_file(self):
        index_file = Path(f"{self.vector_path}.index")
        index_file.parent.mkdir(parents=True, exist_ok=True)
        index_file.write_text("placeholder", encoding="utf-8")

        self.assertTrue(self.storage.vector_index_exists())

    def test_enrich_skill_with_llm_translates_name_and_description_only(self):
        class DummyLLM:
            def chat(self, messages, **kwargs):
                return ChatResponse(content="API Builder: Build API endpoints", model="dummy")

        skill_data = {
            "id": "hub_api-builder",
            "name": "接口构建器",
            "description": "用于快速创建后端接口",
            "enhanced_description": None,
            "category": ["开发"],
            "capabilities": ["接口"],
            "raw_content": "ignored",
        }

        enriched = self.storage.enrich_skill_with_llm(skill_data, DummyLLM())

        self.assertEqual(enriched["enhanced_description"], "API Builder: Build API endpoints")
        self.assertEqual(enriched["category"], ["开发"])
        self.assertEqual(enriched["capabilities"], ["接口"])
        self.assertNotIn("raw_content", enriched)


if __name__ == "__main__":
    unittest.main()
