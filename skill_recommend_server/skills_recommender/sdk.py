from typing import Any, Dict, Optional

from .app_main import build_engine, build_storage, load_catalog
from .catalog import SkillSpec
from .config import settings
from .logger import logger


class SkillsRecommender:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._init_components()

    def _init_components(self):
        from .embedding import EmbeddingModel
        from .vector_store import FAISSVectorStore

        self.storage = build_storage()
        self.catalog = load_catalog(self.storage)
        self.vector_store = FAISSVectorStore(
            dimension=settings.vector_store.dimension,
            index_path=settings.storage.vector.vector_index_path,
            metric=settings.vector_store.metric,
        )

        if self.storage.vector_index_exists():
            self.vector_store.load(settings.storage.vector.vector_index_path)

        self.embedding_model = EmbeddingModel(
            model_path=settings.embedding.model_path,
            device=settings.embedding.device,
        )
        self.engine = build_engine(
            catalog=self.catalog,
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
        )
        from .app_main import ensure_vector_index_ready

        ensure_vector_index_ready(
            self.storage,
            self.catalog,
            self.vector_store,
            self.embedding_model,
            self.engine,
        )

    def add_skill(self, skill: SkillSpec):
        self.catalog.add_skill(skill)

    def add_skills(self, skills: list[dict]):
        for skill_data in skills:
            skill = SkillSpec(**skill_data)
            self.catalog.add_skill(skill)

    def build_index(self):
        self.vector_store.reset()
        self.engine.build_index()
        self.vector_store.save(settings.storage.vector.vector_index_path)
        self.storage.save_vector_state(
            self.catalog.get_all_skills(),
            dimension=settings.vector_store.dimension,
            model_path=settings.embedding.model_path,
        )

    def load_index(self):
        self.vector_store.load(settings.storage.vector.vector_index_path)

    def recommend(self, query: str, filters: Optional[Dict[str, Any]] = None) -> dict:
        return self.engine.recommend(query, filters)

    def get_all_skills(self) -> list:
        return self.catalog.get_all_skills()


def create_recommender() -> SkillsRecommender:
    return SkillsRecommender()


if __name__ == "__main__":
    recommender = create_recommender()
    logger.info("Loaded %s skills", recommender.catalog.count())

    if recommender.catalog.count() == 0:
        logger.warning("No skills loaded. Please run the API import flow first.")
    else:
        result = recommender.recommend("帮我生成一个API接口")
        print(result)
