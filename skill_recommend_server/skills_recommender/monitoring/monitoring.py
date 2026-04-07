import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FeedbackData(BaseModel):
    recommendation_id: str
    user_id: Optional[str] = None
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None
    selected_skill: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class FeedbackStore:
    def __init__(self):
        self.feedbacks: List[FeedbackData] = []

    def add_feedback(self, feedback: FeedbackData) -> None:
        self.feedbacks.append(feedback)

    def get_all_feedbacks(self) -> List[FeedbackData]:
        return self.feedbacks

    def get_negative_feedback(self) -> List[FeedbackData]:
        return [f for f in self.feedbacks if f.rating <= 2]

    def get_feedback_stats(self) -> Dict[str, Any]:
        if not self.feedbacks:
            return {"total": 0, "avg_rating": 0, "by_rating": {}}
        
        ratings = [f.rating for f in self.feedbacks]
        by_rating = {i: ratings.count(i) for i in range(1, 6)}
        
        return {
            "total": len(self.feedbacks),
            "avg_rating": sum(ratings) / len(ratings),
            "by_rating": by_rating,
        }


class LangfuseTracer:
    def __init__(
        self,
        enabled: bool = False,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = "https://cloud.langfuse.com",
    ):
        self.enabled = enabled
        self.client = None
        
        if enabled and public_key and secret_key:
            try:
                from langfuse import Langfuse
                self.client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
            except ImportError:
                import warnings
                warnings.warn("langfuse package not installed. Run: pip install langfuse")

    def trace(self, name: str):
        if not self.enabled or self.client is None:
            return NullTrace()
        return LangfuseTraceWrapper(self.client.trace(name=name))

    def is_enabled(self) -> bool:
        return self.enabled and self.client is not None


class NullTrace:
    def event(self, name: str, metadata: Dict[str, Any] = None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class LangfuseTraceWrapper:
    def __init__(self, trace):
        self.trace = trace

    def event(self, name: str, metadata: Dict[str, Any] = None):
        self.trace.event(name=name, metadata=metadata)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.trace.end()


class MonitoringService:
    def __init__(self, tracer: LangfuseTracer, feedback_store: FeedbackStore):
        self.tracer = tracer
        self.feedback_store = feedback_store

    def record_query_received(self, query: str, user_id: Optional[str] = None):
        self.tracer.trace("skill_recommendation").event(
            "query_received",
            {"query": query, "user_id": user_id}
        )

    def record_vector_search_done(self, results_count: int):
        self.tracer.trace("skill_recommendation").event(
            "vector_search_done",
            {"results_count": results_count}
        )

    def record_llm_rerank_done(self, results_count: int):
        self.tracer.trace("skill_recommendation").event(
            "llm_rerank_done",
            {"results_count": results_count}
        )

    def record_recommendation_generated(self, recommendations: List[Dict]):
        self.tracer.trace("skill_recommendation").event(
            "recommendation_generated",
            {"count": len(recommendations)}
        )

    def record_feedback(self, feedback: FeedbackData):
        self.feedback_store.add_feedback(feedback)
        self.tracer.trace("skill_recommendation").event(
            "feedback_received",
            feedback.model_dump()
        )
