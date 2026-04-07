import sys
import types
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "faiss" not in sys.modules:
    fake_faiss = types.SimpleNamespace(
        IndexFlatIP=object,
        IndexFlatL2=object,
        normalize_L2=lambda *_args, **_kwargs: None,
        write_index=lambda *_args, **_kwargs: None,
        read_index=lambda *_args, **_kwargs: None,
    )
    sys.modules["faiss"] = fake_faiss

from skills_recommender.catalog import SkillCatalog, SkillSpec
from skills_recommender.llm import ChatResponse
from skills_recommender.recommendation import RecommendationEngine


class DummyEmbeddingModel:
    def encode_query(self, query):
        return np.array([0.2, 0.8], dtype=np.float32)

    def encode(self, texts):
        return np.array([[float(index), float(index) + 0.5] for index, _ in enumerate(texts)], dtype=np.float32)


class DummyVectorStore:
    def __init__(self, search_results=None):
        self.search_results = search_results or []
        self.added_embeddings = None
        self.added_metadata = None

    def search(self, query_embedding, top_k=10, score_threshold=None):
        results = []
        for item in self.search_results[:top_k]:
            if score_threshold is None or item["score"] >= score_threshold:
                results.append(item)
        return results

    def add_vectors(self, embeddings, metadata):
        self.added_embeddings = embeddings
        self.added_metadata = metadata


class DummyLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, messages, **kwargs):
        content = self.responses.pop(0)
        return ChatResponse(content=content, model="dummy")


class RecommendationEngineTests(unittest.TestCase):
    def _build_catalog(self):
        catalog = SkillCatalog()
        catalog.add_skill(
            SkillSpec(
                id="skill_api",
                name="API Builder",
                description="Build backend APIs",
                enhanced_description="Create backend APIs efficiently",
                category=["api_development", "code_generation"],
                capabilities=["api", "backend", "service"],
                input_schema={"transport": "http", "format": "json"},
                output_schema={"type": "openapi"},
            )
        )
        catalog.add_skill(
            SkillSpec(
                id="skill_research",
                name="Research Analyst",
                description="Research markets and compare competitors",
                category=["research", "analysis"],
                capabilities=["research", "competitor", "analysis"],
            )
        )
        catalog.add_skill(
            SkillSpec(
                id="skill_image",
                name="Image Helper",
                description="Edit images",
                category=["image_processing"],
                capabilities=["image", "photo"],
                input_schema={"file_type": ["png", "jpg"]},
            )
        )
        catalog.add_skill(
            SkillSpec(
                id="skill_audio",
                name="Audio Transcriber",
                description="Transcribe interview audio into searchable text",
                category=["audio_transcription"],
                capabilities=["audio", "transcribe", "subtitle"],
                input_schema={"media_type": "audio", "source": "interview"},
                output_schema={"format": "subtitle"},
            )
        )
        return catalog

    def test_recommend_filters_by_category_and_updates_usage(self):
        catalog = self._build_catalog()
        vector_store = DummyVectorStore(
            search_results=[
                {"score": 0.91, "metadata": catalog.get_skill("skill_api").model_dump()},
                {"score": 0.89, "metadata": catalog.get_skill("skill_image").model_dump()},
            ]
        )
        engine = RecommendationEngine(
            catalog=catalog,
            vector_store=vector_store,
            embedding_model=DummyEmbeddingModel(),
            llm=None,
            final_top_k=5,
            similarity_threshold=0.5,
            enable_llm_rerank=False,
            keyword_recall_enabled=False,
            intent_recall_enabled=False,
        )

        result = engine.recommend("build api", filters={"category": "code_generation"})

        self.assertEqual(len(result["recommendations"]), 1)
        self.assertEqual(result["recommendations"][0]["skill_id"], "skill_api")
        self.assertEqual(result["recommendations"][0]["description"], "Create backend APIs efficiently")
        self.assertIn("score", result["recommendations"][0])
        self.assertIn("score_band", result["recommendations"][0])
        self.assertEqual(catalog.get_skill("skill_api").usage_stats["times_recommended"], 1)

    def test_recommend_uses_llm_understanding_when_available(self):
        catalog = self._build_catalog()
        vector_store = DummyVectorStore(
            search_results=[
                {"score": 0.91, "metadata": catalog.get_skill("skill_api").model_dump()},
            ]
        )
        llm = DummyLLM(
            [
                '{"intent":"api_development","task_types":["api_development","code_generation"],'
                '"entities":["api"],"keywords":["backend","api"],"constraints":["json"],"rewritten_query":"build backend api"}'
            ]
        )
        engine = RecommendationEngine(
            catalog=catalog,
            vector_store=vector_store,
            embedding_model=DummyEmbeddingModel(),
            llm=llm,
            final_top_k=2,
            enable_llm_rerank=False,
            keyword_recall_enabled=False,
            intent_recall_enabled=False,
        )

        result = engine.recommend("build api")

        self.assertEqual(result["query_understanding"]["understanding_method"], "llm")
        self.assertEqual(result["query_understanding"]["intent"], "api_development")
        self.assertIn("json", result["query_understanding"]["constraints"])

    def test_recommend_falls_back_to_dynamic_heuristic_understanding_without_llm(self):
        catalog = self._build_catalog()
        vector_store = DummyVectorStore(search_results=[])
        engine = RecommendationEngine(
            catalog=catalog,
            vector_store=vector_store,
            embedding_model=DummyEmbeddingModel(),
            llm=None,
            final_top_k=2,
            vector_recall_enabled=False,
            keyword_recall_enabled=True,
            intent_recall_enabled=True,
        )

        result = engine.recommend("research and compare competitors")

        self.assertEqual(result["query_understanding"]["understanding_method"], "heuristic")
        self.assertIn("research", result["query_understanding"]["task_types"])
        self.assertEqual(result["recommendations"][0]["skill_id"], "skill_research")

    def test_dynamic_skill_terms_drive_heuristic_understanding(self):
        catalog = self._build_catalog()
        vector_store = DummyVectorStore(search_results=[])
        engine = RecommendationEngine(
            catalog=catalog,
            vector_store=vector_store,
            embedding_model=DummyEmbeddingModel(),
            llm=None,
            final_top_k=2,
            vector_recall_enabled=False,
            keyword_recall_enabled=True,
            intent_recall_enabled=True,
        )

        result = engine.recommend("transcribe audio interview")

        self.assertEqual(result["recommendations"][0]["skill_id"], "skill_audio")
        self.assertIn("audio_transcription", result["query_understanding"]["task_types"])
        self.assertEqual(result["query_understanding"]["intent"], "audio_transcription")

    def test_constraints_are_derived_from_skill_metadata(self):
        catalog = self._build_catalog()
        vector_store = DummyVectorStore(search_results=[])
        engine = RecommendationEngine(
            catalog=catalog,
            vector_store=vector_store,
            embedding_model=DummyEmbeddingModel(),
            llm=None,
            final_top_k=2,
            vector_recall_enabled=False,
            keyword_recall_enabled=True,
            intent_recall_enabled=True,
        )

        result = engine.recommend("generate json api and openapi spec")

        constraints = result["query_understanding"]["constraints"]
        self.assertIn("json", constraints)
        self.assertIn("openapi", constraints)
        self.assertEqual(result["recommendations"][0]["skill_id"], "skill_api")

    def test_cache_refreshes_after_catalog_changes(self):
        catalog = self._build_catalog()
        vector_store = DummyVectorStore(search_results=[])
        engine = RecommendationEngine(
            catalog=catalog,
            vector_store=vector_store,
            embedding_model=DummyEmbeddingModel(),
            llm=None,
            final_top_k=2,
            vector_recall_enabled=False,
            keyword_recall_enabled=True,
            intent_recall_enabled=True,
        )

        initial = engine.recommend("spreadsheet reconciliation")
        self.assertEqual(initial["recommendations"], [])

        catalog.add_skill(
            SkillSpec(
                id="skill_sheet",
                name="Spreadsheet Reconciler",
                description="Reconcile spreadsheet rows and validate totals",
                category=["spreadsheet_reconciliation"],
                capabilities=["spreadsheet", "reconciliation", "totals"],
                input_schema={"file_type": "xlsx"},
            )
        )

        refreshed = engine.recommend("spreadsheet reconciliation")

        self.assertEqual(refreshed["recommendations"][0]["skill_id"], "skill_sheet")
        self.assertIn("spreadsheet_reconciliation", refreshed["query_understanding"]["task_types"])

    def test_keyword_and_intent_recall_can_rank_without_vector_results(self):
        catalog = self._build_catalog()
        vector_store = DummyVectorStore(search_results=[])
        engine = RecommendationEngine(
            catalog=catalog,
            vector_store=vector_store,
            embedding_model=DummyEmbeddingModel(),
            llm=None,
            final_top_k=2,
            vector_recall_enabled=False,
            keyword_recall_enabled=True,
            intent_recall_enabled=True,
        )

        result = engine.recommend("build backend api service")

        self.assertEqual(result["recommendations"][0]["skill_id"], "skill_api")
        self.assertIn("keyword=", result["recommendations"][0]["match_reason"])
        self.assertTrue(result["recommendations"][0]["matched_terms"])

    def test_keyword_score_uses_only_name_and_description(self):
        catalog = SkillCatalog()
        catalog.add_skill(
            SkillSpec(
                id="skill_ops",
                name="Operations Helper",
                description="Coordinate workflows and document decisions",
                category=["product_strategy"],
                capabilities=["competitive_analysis"],
            )
        )
        vector_store = DummyVectorStore(search_results=[])
        engine = RecommendationEngine(
            catalog=catalog,
            vector_store=vector_store,
            embedding_model=DummyEmbeddingModel(),
            llm=None,
            vector_recall_enabled=False,
            keyword_recall_enabled=True,
            intent_recall_enabled=False,
        )

        result = engine.recommend("competitive analysis")

        self.assertEqual(result["recommendations"], [])

    def test_keyword_score_requires_same_language_description(self):
        catalog = SkillCatalog()
        catalog.add_skill(
            SkillSpec(
                id="skill_cn",
                name="接口构建器",
                description="用于快速创建后端接口",
            )
        )
        engine = RecommendationEngine(
            catalog=catalog,
            vector_store=DummyVectorStore(search_results=[]),
            embedding_model=DummyEmbeddingModel(),
            llm=None,
            vector_recall_enabled=False,
            keyword_recall_enabled=True,
            intent_recall_enabled=False,
        )

        result = engine.recommend("build backend api")

        self.assertEqual(result["recommendations"], [])

    def test_keyword_score_can_use_translated_enhanced_description(self):
        catalog = SkillCatalog()
        catalog.add_skill(
            SkillSpec(
                id="skill_cn",
                name="接口构建器",
                description="用于快速创建后端接口",
                enhanced_description="API Builder: Build backend APIs quickly",
            )
        )
        engine = RecommendationEngine(
            catalog=catalog,
            vector_store=DummyVectorStore(search_results=[]),
            embedding_model=DummyEmbeddingModel(),
            llm=None,
            vector_recall_enabled=False,
            keyword_recall_enabled=True,
            intent_recall_enabled=False,
        )

        result = engine.recommend("build backend api")

        self.assertEqual(result["recommendations"][0]["skill_id"], "skill_cn")
        self.assertGreater(result["recommendations"][0]["score_breakdown"]["keyword_score"], 0.0)

    def test_intent_recall_is_disabled_for_cross_language_without_llm(self):
        catalog = SkillCatalog()
        catalog.add_skill(
            SkillSpec(
                id="skill_cn",
                name="接口构建器",
                description="用于快速创建后端接口",
                category=["api_development"],
                capabilities=["backend", "api"],
            )
        )
        engine = RecommendationEngine(
            catalog=catalog,
            vector_store=DummyVectorStore(search_results=[]),
            embedding_model=DummyEmbeddingModel(),
            llm=None,
            vector_recall_enabled=False,
            keyword_recall_enabled=False,
            intent_recall_enabled=True,
            intent_score_threshold=0.0,
        )

        result = engine.recommend("build backend api")

        self.assertEqual(result["recommendations"], [])

    def test_intent_recall_can_bridge_cross_language_with_llm(self):
        catalog = SkillCatalog()
        catalog.add_skill(
            SkillSpec(
                id="skill_cn",
                name="接口构建器",
                description="用于快速创建后端接口",
                category=["api_development"],
                capabilities=["backend", "api"],
            )
        )
        llm = DummyLLM(
            [
                '{"intent":"api_development","task_types":["api_development"],'
                '"entities":["api"],"keywords":["backend","api"],"constraints":[],"rewritten_query":"build backend api"}'
            ]
        )
        engine = RecommendationEngine(
            catalog=catalog,
            vector_store=DummyVectorStore(search_results=[]),
            embedding_model=DummyEmbeddingModel(),
            llm=llm,
            vector_recall_enabled=False,
            keyword_recall_enabled=False,
            intent_recall_enabled=True,
            intent_score_threshold=0.0,
        )

        result = engine.recommend("构建 backend api")

        self.assertEqual(result["recommendations"][0]["skill_id"], "skill_cn")
        self.assertGreater(result["recommendations"][0]["score_breakdown"]["intent_score"], 0.0)

    def test_recommend_uses_llm_rerank_when_enabled(self):
        catalog = self._build_catalog()
        vector_store = DummyVectorStore(
            search_results=[
                {"score": 0.91, "metadata": catalog.get_skill("skill_api").model_dump()},
                {"score": 0.95, "metadata": catalog.get_skill("skill_image").model_dump()},
            ]
        )
        llm = DummyLLM(
            [
                '{"intent":"api_development","task_types":["api_development"],"entities":["api"],"keywords":["api"],"constraints":["json"],"rewritten_query":"build api"}',
                "skill_api,skill_image",
            ]
        )
        engine = RecommendationEngine(
            catalog=catalog,
            vector_store=vector_store,
            embedding_model=DummyEmbeddingModel(),
            llm=llm,
            final_top_k=2,
            similarity_threshold=0.5,
            enable_llm_rerank=True,
            keyword_recall_enabled=False,
            intent_recall_enabled=False,
        )

        result = engine.recommend("build api")

        self.assertEqual(result["recommendations"][0]["skill_id"], "skill_api")
        self.assertEqual(result["recommendations"][1]["skill_id"], "skill_image")

    def test_usage_stats_do_not_affect_ranking(self):
        catalog = self._build_catalog()
        popular_skill = catalog.get_skill("skill_research")
        popular_skill.usage_stats["times_recommended"] = 8
        popular_skill.usage_stats["times_selected"] = 4

        vector_store = DummyVectorStore(
            search_results=[
                {"score": 0.55, "metadata": catalog.get_skill("skill_api").model_dump()},
                {"score": 0.50, "metadata": popular_skill.model_dump()},
            ]
        )
        engine = RecommendationEngine(
            catalog=catalog,
            vector_store=vector_store,
            embedding_model=DummyEmbeddingModel(),
            llm=None,
            final_top_k=2,
            similarity_threshold=0.5,
            enable_llm_rerank=False,
            vector_recall_enabled=True,
            keyword_recall_enabled=False,
            intent_recall_enabled=False,
            semantic_weight=1.0,
        )

        result = engine.recommend("general task")

        self.assertEqual(result["recommendations"][0]["skill_id"], "skill_api")

    def test_build_index_pushes_embeddings_and_metadata_to_vector_store(self):
        catalog = self._build_catalog()
        vector_store = DummyVectorStore()
        engine = RecommendationEngine(
            catalog=catalog,
            vector_store=vector_store,
            embedding_model=DummyEmbeddingModel(),
            llm=None,
            enable_llm_rerank=False,
        )

        engine.build_index()

        self.assertEqual(vector_store.added_embeddings.shape, (4, 2))
        self.assertEqual(len(vector_store.added_metadata), 4)
        self.assertEqual(vector_store.added_metadata[0]["id"], "skill_api")


if __name__ == "__main__":
    unittest.main()
