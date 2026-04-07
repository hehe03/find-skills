import argparse
import json
from pathlib import Path
from typing import Optional

from .catalog import SkillCatalog, SkillSpec
from .config import settings
from .llm import LLMFactory
from .logger import logger
from .storage import SkillStorage


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def ensure_runtime_environment() -> None:
    DATA_DIR.mkdir(exist_ok=True)


def build_llm():
    if settings.llm.default not in settings.llm.providers:
        return None

    provider_config = settings.llm.providers[settings.llm.default]
    if not provider_config.api_key and settings.llm.default not in {"ollama"}:
        logger.warning(
            "[CLI] LLM provider '%s' has no API key configured; continuing without LLM",
            settings.llm.default,
        )
        return None

    return LLMFactory.create(
        settings.llm.default,
        api_key=provider_config.api_key,
        base_url=provider_config.base_url,
        model=provider_config.model,
        temperature=provider_config.temperature,
    )


def build_storage() -> SkillStorage:
    ensure_runtime_environment()
    return SkillStorage(
        json_enabled=settings.storage.json_settings.enabled,
        json_path=settings.storage.json_settings.file_path,
        db_enabled=settings.storage.database.enabled,
        db_path=settings.storage.database.db_path,
        vector_enabled=settings.storage.vector.enabled,
        vector_index_path=settings.storage.vector.vector_index_path,
    )


def ensure_vector_index_ready(storage: SkillStorage, catalog: SkillCatalog, vector_store, embedding_model, engine) -> None:
    skills = catalog.get_all_skills()
    if not skills:
        return

    if storage.is_vector_index_consistent(
        skills,
        dimension=settings.vector_store.dimension,
        model_path=settings.embedding.model_path,
    ):
        vector_store.load(settings.storage.vector.vector_index_path)
        if not storage.is_vector_metadata_consistent(skills):
            vector_store.metadata = [skill.model_dump() for skill in skills]
            storage.save_vector_metadata(vector_store.metadata)
        return

    logger.info("[CLI] Vector index missing or stale, rebuilding")
    vector_store.reset()
    engine.build_index()
    vector_store.save(settings.storage.vector.vector_index_path)
    storage.save_vector_state(
        skills,
        dimension=settings.vector_store.dimension,
        model_path=settings.embedding.model_path,
    )


def load_catalog(storage: SkillStorage) -> SkillCatalog:
    catalog = SkillCatalog()
    for skill_data in storage.load_skills():
        try:
            catalog.add_skill(SkillSpec(**skill_data))
        except Exception as exc:
            logger.warning("[CLI] Failed to load skill: %s", exc)
    return catalog


def build_engine(
    catalog: SkillCatalog,
    vector_store,
    embedding_model,
):
    from .recommendation import RecommendationEngine

    return RecommendationEngine(
        catalog=catalog,
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm=build_llm(),
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


def run_api(reload: bool = False, update_skills: bool = False, use_llm: Optional[bool] = None):
    from .api import create_app
    import uvicorn

    ensure_runtime_environment()
    app = create_app(update_skills=update_skills, use_llm=use_llm)
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        reload=reload,
    )


def run_cli(query: str = None, json_output: bool = False, init: bool = False):
    from .embedding import EmbeddingModel
    from .vector_store import FAISSVectorStore

    storage = build_storage()
    catalog = load_catalog(storage)
    vector_store = FAISSVectorStore(
        dimension=settings.vector_store.dimension,
        index_path=settings.storage.vector.vector_index_path,
        metric=settings.vector_store.metric,
    )

    if storage.vector_index_exists():
        vector_store.load(settings.storage.vector.vector_index_path)

    embedding_model = EmbeddingModel(
        model_path=settings.embedding.model_path,
        device=settings.embedding.device,
    )
    engine = build_engine(catalog, vector_store, embedding_model)
    ensure_vector_index_ready(storage, catalog, vector_store, embedding_model, engine)

    if init:
        vector_store.reset()
        engine.build_index()
        vector_store.save(settings.storage.vector.vector_index_path)
        storage.save_vector_state(
            catalog.get_all_skills(),
            dimension=settings.vector_store.dimension,
            model_path=settings.embedding.model_path,
        )
        logger.info("Skills initialized successfully")
        return

    if not query:
        raise SystemExit('Please provide a query, for example: python -m skills_recommender "帮我写一个 API"')

    if catalog.count() == 0:
        raise SystemExit("No skills were loaded. Please import skills-hub data first.")

    result = engine.recommend(query)

    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"\n找到 {len(result['recommendations'])} 个推荐:\n")
    for rec in result["recommendations"]:
        print(f"  - {rec['name']} (ID: {rec['skill_id']})")
        print(f"  - score: {rec['score']:.2f} ({rec['score_band']})")
        print(f"  - description: {rec['description']}")
        print(f"  - match_reason: {rec['match_reason']}")
        if rec.get("matched_terms"):
            print(f"  - matched_terms: {', '.join(rec['matched_terms'])}")


def main():
    ensure_runtime_environment()
    parser = argparse.ArgumentParser(description="Find Skills - skill recommendation service")
    parser.add_argument("query", type=str, nargs="?", help="Natural-language task description")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument("--init", action="store_true", help="Rebuild vector index from current skills")
    parser.add_argument("--api", action="store_true", help="Start API service")
    parser.add_argument("--reload", action="store_true", help="Enable auto reload")
    parser.add_argument("--update-skills", action="store_true", help="Refresh skills from skills-hub on startup")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM enrichment when importing skills")
    args = parser.parse_args()

    if args.api:
        use_llm = args.use_llm if args.use_llm else None
        run_api(reload=args.reload, update_skills=args.update_skills, use_llm=use_llm)
        return

    run_cli(args.query, args.json, args.init)
