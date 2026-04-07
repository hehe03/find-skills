import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import yaml
from dotenv import load_dotenv


class AppSettings(BaseModel):
    name: str = "Skill Recommender"
    version: str = "1.0.0"
    debug: bool = False
    use_llm_enrichment: bool = False


class StorageJSONSettings(BaseModel):
    enabled: bool = True
    file_path: str = "./data/skills.json"


class StorageDatabaseSettings(BaseModel):
    enabled: bool = True
    db_type: str = "sqlite"
    db_path: str = "./data/skills.db"


class StorageVectorSettings(BaseModel):
    enabled: bool = True
    vector_index_path: str = "./data/faiss_index"


class StorageSettings(BaseModel):
    json_settings: StorageJSONSettings = Field(default_factory=StorageJSONSettings)
    database: StorageDatabaseSettings = Field(default_factory=StorageDatabaseSettings)
    vector: StorageVectorSettings = Field(default_factory=StorageVectorSettings)


class VectorStoreSettings(BaseModel):
    type: str = "faiss"
    dimension: int = 768
    metric: str = "ip"


class EmbeddingSettings(BaseModel):
    provider: str = "GTE"
    model_path: str = "./models/GTE-Multilingual-Base"
    device: str = "cpu"
    batch_size: int = 32


class LLMProviderSettings(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4o"
    temperature: float = 0.7


class LLMSettings(BaseModel):
    default: str = "openai"
    providers: Dict[str, LLMProviderSettings] = Field(default_factory=dict)


class LangfuseSettings(BaseModel):
    enabled: bool = False
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    host: str = "https://cloud.langfuse.com"


class DatabaseSettings(BaseModel):
    type: str = "sqlite"
    path: str = "./data/skills.db"


class RecommendationSettings(BaseModel):
    class TaskUnderstandingSettings(BaseModel):
        enabled: bool = True
        use_llm_when_available: bool = True
        llm_temperature: float = 0.1
        max_keywords: int = 8

    class RecallMethodSettings(BaseModel):
        enabled: bool = True
        top_k: int = 20
        score_threshold: float = 0.1

    class RecallSettings(BaseModel):
        vector: "RecommendationSettings.RecallMethodSettings" = Field(
            default_factory=lambda: RecommendationSettings.RecallMethodSettings(top_k=50, score_threshold=0.5)
        )
        keyword: "RecommendationSettings.RecallMethodSettings" = Field(
            default_factory=lambda: RecommendationSettings.RecallMethodSettings(top_k=20, score_threshold=0.15)
        )
        intent: "RecommendationSettings.RecallMethodSettings" = Field(
            default_factory=lambda: RecommendationSettings.RecallMethodSettings(top_k=20, score_threshold=0.2)
        )

    class ScoringSettings(BaseModel):
        semantic_weight: float = 0.5
        keyword_weight: float = 0.2
        intent_weight: float = 0.2
        usage_weight: float = 0.1

    class RerankSettings(BaseModel):
        enable_llm_rerank: bool = True
        llm_top_n: int = 10

    final_top_k: int = 5
    enable_graph_expand: bool = True
    task_understanding: TaskUnderstandingSettings = Field(default_factory=TaskUnderstandingSettings)
    recall: RecallSettings = Field(default_factory=RecallSettings)
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)

    @property
    def vector_top_k(self) -> int:
        return self.recall.vector.top_k

    @property
    def similarity_threshold(self) -> float:
        return self.recall.vector.score_threshold

    @property
    def enable_llm_rerank(self) -> bool:
        return self.rerank.enable_llm_rerank


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])


class Settings(BaseModel):
    app: AppSettings = Field(default_factory=AppSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    recommendation: RecommendationSettings = Field(default_factory=RecommendationSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Settings":
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        
        with open(config_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        
        return cls(**config_data)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        config = cls.from_yaml()
        
        if config.llm.providers.get("openai"):
            config.llm.providers["openai"].api_key = os.getenv("OPENAI_API_KEY", config.llm.providers["openai"].api_key)
        
        if config.llm.providers.get("anthropic"):
            config.llm.providers["anthropic"].api_key = os.getenv("ANTHROPIC_API_KEY", config.llm.providers["anthropic"].api_key)
        
        if config.llm.providers.get("zhipu"):
            config.llm.providers["zhipu"].api_key = os.getenv("ZHIPU_API_KEY", config.llm.providers["zhipu"].api_key)
        
        if config.llm.providers.get("alicloud"):
            config.llm.providers["alicloud"].api_key = os.getenv("DASHSCOPE_API_KEY", config.llm.providers["alicloud"].api_key)
        
        if config.langfuse.enabled:
            config.langfuse.public_key = os.getenv("LANGFUSE_PUBLIC_KEY", config.langfuse.public_key)
            config.langfuse.secret_key = os.getenv("LANGFUSE_SECRET_KEY", config.langfuse.secret_key)
        
        return config


settings = Settings.from_env()
