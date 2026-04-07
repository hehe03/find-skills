import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills_recommender.catalog import SkillCatalog, SkillSpec


class SkillCatalogTests(unittest.TestCase):
    def test_add_get_and_count_skills(self):
        catalog = SkillCatalog()
        skill = SkillSpec(id="skill_api", name="API", description="Build APIs")

        catalog.add_skill(skill)

        self.assertEqual(catalog.count(), 1)
        self.assertEqual(catalog.get_skill("skill_api").name, "API")

    def test_search_by_category_is_case_insensitive(self):
        catalog = SkillCatalog()
        catalog.add_skill(
            SkillSpec(
                id="skill_image",
                name="Image",
                description="Process images",
                category=["Image_Processing"],
            )
        )

        results = catalog.search_by_category("image_processing")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "skill_image")

    def test_update_usage_tracks_recommended_and_selected(self):
        catalog = SkillCatalog()
        catalog.add_skill(SkillSpec(id="skill_data", name="Data", description="Analyze data"))

        catalog.update_usage("skill_data", recommended=True)
        catalog.update_usage("skill_data", selected=True)

        usage_stats = catalog.get_skill("skill_data").usage_stats
        self.assertEqual(usage_stats["times_recommended"], 1)
        self.assertEqual(usage_stats["times_selected"], 1)

    def test_round_trip_from_dict(self):
        catalog = SkillCatalog()
        catalog.add_skill(
            SkillSpec(
                id="skill_writer",
                name="Writer",
                description="Write content",
                category=["content_generation"],
                capabilities=["drafting"],
            )
        )

        restored = SkillCatalog.from_dict(catalog.to_dict())

        self.assertEqual(restored.count(), 1)
        self.assertEqual(restored.get_skill("skill_writer").capabilities, ["drafting"])


if __name__ == "__main__":
    unittest.main()
