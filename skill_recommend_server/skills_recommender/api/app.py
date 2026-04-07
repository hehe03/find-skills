from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..catalog import SkillCatalog, SkillSpec
from ..config import settings
from ..embedding import EmbeddingModel
from ..llm import LLMFactory
from ..logger import logger
from ..monitoring import FeedbackData, FeedbackStore, LangfuseTracer, MonitoringService
from ..recommendation import RecommendationEngine
from ..storage import SkillStorage
from ..vector_store import FAISSVectorStore


class RecommendRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None


class RecommendResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    query_understanding: Dict[str, Any]


class FeedbackRequest(BaseModel):
    recommendation_id: str
    user_id: Optional[str] = None
    rating: int
    comment: Optional[str] = None
    selected_skill: Optional[str] = None


class AddSkillRequest(BaseModel):
    id: str
    name: str
    description: str
    enhanced_description: Optional[str] = None
    category: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    version: str = "1.0.0"
    dependencies: List[str] = Field(default_factory=list)
    author: Optional[str] = None


class SkillRecommenderApp:
    def __init__(self, update_skills: bool = False, use_llm: Optional[bool] = None):
        self.storage = SkillStorage(
            json_enabled=settings.storage.json_settings.enabled,
            json_path=settings.storage.json_settings.file_path,
            db_enabled=settings.storage.database.enabled,
            db_path=settings.storage.database.db_path,
            vector_enabled=settings.storage.vector.enabled,
            vector_index_path=settings.storage.vector.vector_index_path,
        )

        self.catalog = SkillCatalog()
        self._load_skills_from_storage()

        self.embedding_model = EmbeddingModel(
            model_path=settings.embedding.model_path,
            device=settings.embedding.device,
        )
        self.vector_store = FAISSVectorStore(
            dimension=settings.vector_store.dimension,
            index_path=settings.storage.vector.vector_index_path,
            metric=settings.vector_store.metric,
        )

        self.llm = None
        if settings.llm.default in settings.llm.providers:
            provider_config = settings.llm.providers[settings.llm.default]
            self.llm = LLMFactory.create(
                settings.llm.default,
                api_key=provider_config.api_key,
                base_url=provider_config.base_url,
                model=provider_config.model,
                temperature=provider_config.temperature,
            )

        self.use_llm_enrichment = (
            use_llm if use_llm is not None else getattr(settings.app, "use_llm_enrichment", False)
        )

        self.recommendation_engine = RecommendationEngine(
            catalog=self.catalog,
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            llm=self.llm,
            vector_top_k=settings.recommendation.recall.vector.top_k,
            final_top_k=settings.recommendation.final_top_k,
            similarity_threshold=settings.recommendation.recall.vector.score_threshold,
            enable_llm_rerank=settings.recommendation.rerank.enable_llm_rerank,
            enable_graph_expand=settings.recommendation.enable_graph_expand,
            task_understanding_enabled=settings.recommendation.task_understanding.enabled,
            task_understanding_use_llm_when_available=settings.recommendation.task_understanding.use_llm_when_available,
            task_understanding_llm_temperature=settings.recommendation.task_understanding.llm_temperature,
            task_understanding_max_keywords=settings.recommendation.task_understanding.max_keywords,
            vector_recall_enabled=settings.recommendation.recall.vector.enabled,
            keyword_recall_enabled=settings.recommendation.recall.keyword.enabled,
            keyword_top_k=settings.recommendation.recall.keyword.top_k,
            keyword_score_threshold=settings.recommendation.recall.keyword.score_threshold,
            intent_recall_enabled=settings.recommendation.recall.intent.enabled,
            intent_top_k=settings.recommendation.recall.intent.top_k,
            intent_score_threshold=settings.recommendation.recall.intent.score_threshold,
            semantic_weight=settings.recommendation.scoring.semantic_weight,
            keyword_weight=settings.recommendation.scoring.keyword_weight,
            intent_weight=settings.recommendation.scoring.intent_weight,
            usage_weight=settings.recommendation.scoring.usage_weight,
            llm_rerank_top_n=settings.recommendation.rerank.llm_top_n,
        )

        self._ensure_index_ready()
        if update_skills:
            self._auto_import_from_skills_hub()

        self.feedback_store = FeedbackStore()
        self.monitoring = MonitoringService(
            LangfuseTracer(
                enabled=settings.langfuse.enabled,
                public_key=settings.langfuse.public_key,
                secret_key=settings.langfuse.secret_key,
                host=settings.langfuse.host,
            ),
            self.feedback_store,
        )

        logger.info("[SkillRecommenderApp] Initialized with %s skills", len(self.catalog.get_all_skills()))

    def _load_skills_from_storage(self):
        for skill_data in self.storage.load_skills():
            try:
                self.catalog.add_skill(SkillSpec(**skill_data))
            except Exception as exc:
                logger.warning("[Storage] Failed to load skill: %s", exc)

    def _rebuild_index(self) -> None:
        self.vector_store.reset()
        self.recommendation_engine.build_index()
        self.vector_store.save(self.storage.get_vector_index_path())
        self.storage.save_vector_state(
            self.catalog.get_all_skills(),
            dimension=settings.vector_store.dimension,
            model_path=settings.embedding.model_path,
        )

    def _sync_storage_consistency(self, rebuild_vector: bool = False) -> None:
        skills = self.catalog.get_all_skills()
        self.storage.save_skills(skills)
        if not skills:
            return

        if rebuild_vector or not self.storage.is_vector_index_consistent(
            skills,
            dimension=settings.vector_store.dimension,
            model_path=settings.embedding.model_path,
        ):
            self._rebuild_index()
            return

        if self.vector_store.is_empty() and self.storage.vector_index_exists():
            self.vector_store.load(self.storage.get_vector_index_path())

        if not self.storage.is_vector_metadata_consistent(skills):
            metadata = [skill.model_dump() for skill in skills]
            self.vector_store.metadata = metadata
            self.storage.save_vector_metadata(metadata)

    def _ensure_index_ready(self) -> None:
        skills = self.catalog.get_all_skills()
        if not skills:
            return

        if self.storage.is_vector_index_consistent(
            skills,
            dimension=settings.vector_store.dimension,
            model_path=settings.embedding.model_path,
        ):
            self.vector_store.load(settings.storage.vector.vector_index_path)
            if not self.storage.is_vector_metadata_consistent(skills):
                self.vector_store.metadata = [skill.model_dump() for skill in skills]
                self.storage.save_vector_metadata(self.vector_store.metadata)
            return

        logger.info("[SkillRecommenderApp] Vector index missing or stale, rebuilding")
        self._rebuild_index()

    def _replace_catalog(self, skills_data: List[Dict[str, Any]], log_prefix: str) -> None:
        new_catalog = SkillCatalog()
        for skill_data in skills_data:
            try:
                new_catalog.add_skill(SkillSpec(**skill_data))
            except Exception as exc:
                logger.warning("%s Failed to add skill: %s", log_prefix, exc)

        self.catalog = new_catalog
        self.recommendation_engine.catalog = self.catalog
        self._sync_storage_consistency(rebuild_vector=True)

    def _auto_import_from_skills_hub(self):
        result = self.storage.update_all_skills()
        if result["status"] == "success" and result.get("updated_skills"):
            if self.use_llm_enrichment and self.llm:
                for index, skill_data in enumerate(result["updated_skills"]):
                    result["updated_skills"][index] = self.storage.enrich_skill_with_llm(skill_data, self.llm)

            self._replace_catalog(result["updated_skills"], log_prefix="[AutoImport]")
            logger.info("[AutoImport] Imported %s skills from skills-hub", len(result["updated_skills"]))

    def add_skill(self, skill_data: AddSkillRequest) -> Dict[str, Any]:
        skill = SkillSpec(**skill_data.model_dump())
        self.catalog.add_skill(skill)
        self._sync_storage_consistency(rebuild_vector=True)
        return {"status": "success", "skill_id": skill.id}

    def recommend(self, request: RecommendRequest) -> Dict[str, Any]:
        self.monitoring.record_query_received(request.query, request.user_id)
        result = self.recommendation_engine.recommend(query=request.query, filters=request.filters)
        self.monitoring.record_vector_search_done(len(result.get("recommendations", [])))
        self.monitoring.record_recommendation_generated(result.get("recommendations", []))
        return result

    def submit_feedback(self, feedback: FeedbackRequest) -> Dict[str, Any]:
        feedback_data = FeedbackData(**feedback.model_dump())
        self.storage.save_feedback(feedback_data)
        self.monitoring.record_feedback(feedback_data)

        if feedback.selected_skill:
            self.catalog.update_usage(feedback.selected_skill, selected=True)
            self._sync_storage_consistency(rebuild_vector=False)

        return {"status": "success"}

    def get_skills(self) -> List[Dict[str, Any]]:
        return [skill.model_dump() for skill in self.catalog.get_all_skills()]

    def get_feedback_stats(self) -> Dict[str, Any]:
        return self.storage.get_feedback_stats()


def create_app(update_skills: bool = False, use_llm: Optional[bool] = None) -> FastAPI:
    app = FastAPI(
        title="Skill Recommender API",
        description="Agent Skills 推荐系统 API",
        version="1.0.0",
    )

    recommender = SkillRecommenderApp(update_skills=update_skills, use_llm=use_llm)

    @app.post("/recommend", response_model=RecommendResponse)
    async def recommend(request: RecommendRequest):
        return recommender.recommend(request)

    @app.post("/feedback")
    async def submit_feedback(feedback: FeedbackRequest):
        return recommender.submit_feedback(feedback)

    @app.post("/skills")
    async def add_skill(skill: AddSkillRequest):
        return recommender.add_skill(skill)

    @app.get("/skills")
    async def get_skills():
        return recommender.get_skills()

    @app.get("/stats/feedback")
    async def get_feedback_stats():
        return recommender.get_feedback_stats()

    @app.post("/skills/update-all")
    async def update_all_skills():
        result = recommender.storage.update_all_skills()
        if result["status"] == "success" and result.get("updated_skills"):
            if recommender.use_llm_enrichment and recommender.llm:
                for index, skill_data in enumerate(result["updated_skills"]):
                    result["updated_skills"][index] = recommender.storage.enrich_skill_with_llm(
                        skill_data,
                        recommender.llm,
                    )

            recommender._replace_catalog(result["updated_skills"], log_prefix="[API]")

        return result

    @app.post("/skills/import")
    async def import_skills(skill_id: Optional[str] = None):
        existing_ids = [skill.id for skill in recommender.catalog.get_all_skills()]
        result = recommender.storage.import_skills(skill_id=skill_id, existing_ids=existing_ids)

        if result["status"] == "success" and result.get("imported_skills"):
            if recommender.use_llm_enrichment and recommender.llm:
                for index, skill_data in enumerate(result["imported_skills"]):
                    result["imported_skills"][index] = recommender.storage.enrich_skill_with_llm(
                        skill_data,
                        recommender.llm,
                    )

            for skill_data in result["imported_skills"]:
                try:
                    recommender.catalog.add_skill(SkillSpec(**skill_data))
                except Exception as exc:
                    logger.warning("[API] Failed to add imported skill: %s", exc)

            recommender._sync_storage_consistency(rebuild_vector=True)

        return result

    return app
