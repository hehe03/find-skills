import hashlib
import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..catalog import SkillCatalog, SkillSpec
from ..embedding import EmbeddingModel
from ..llm import LLMAdapter, Message
from ..logger import logger
from ..vector_store import FAISSVectorStore


class Recommendation(BaseModel):
    skill_id: str
    name: str
    description: str
    score: float
    score_band: str
    match_reason: str
    matched_terms: List[str] = Field(default_factory=list)
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    alternatives: List[str] = Field(default_factory=list)


class QueryUnderstanding(BaseModel):
    intent: str = "general_assistance"
    task_types: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    rewritten_query: str
    understanding_method: str = "heuristic"
    enabled_recalls: List[str] = Field(default_factory=list)
    query_terms: List[str] = Field(default_factory=list)
    query_language: str = "en"


class RecommendationEngine:
    def __init__(
        self,
        catalog: SkillCatalog,
        vector_store: FAISSVectorStore,
        embedding_model: EmbeddingModel,
        llm: Optional[LLMAdapter] = None,
        vector_top_k: int = 50,
        final_top_k: int = 5,
        similarity_threshold: float = 0.5,
        enable_llm_rerank: bool = True,
        enable_graph_expand: bool = True,
        task_understanding_enabled: bool = True,
        task_understanding_use_llm_when_available: bool = True,
        task_understanding_llm_temperature: float = 0.1,
        task_understanding_max_keywords: int = 8,
        vector_recall_enabled: bool = True,
        keyword_recall_enabled: bool = True,
        keyword_top_k: int = 20,
        keyword_score_threshold: float = 0.15,
        intent_recall_enabled: bool = True,
        intent_top_k: int = 20,
        intent_score_threshold: float = 0.2,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.2,
        intent_weight: float = 0.1,
        usage_weight: float = 0.0,
        llm_rerank_top_n: int = 10,
    ):
        self.catalog = catalog
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.llm = llm
        self.vector_top_k = vector_top_k
        self.final_top_k = final_top_k
        self.similarity_threshold = similarity_threshold
        self.enable_llm_rerank = enable_llm_rerank and llm is not None
        self.enable_graph_expand = enable_graph_expand
        self.task_understanding_enabled = task_understanding_enabled
        self.task_understanding_use_llm_when_available = task_understanding_use_llm_when_available
        self.task_understanding_llm_temperature = task_understanding_llm_temperature
        self.task_understanding_max_keywords = task_understanding_max_keywords
        self.vector_recall_enabled = vector_recall_enabled
        self.keyword_recall_enabled = keyword_recall_enabled
        self.keyword_top_k = keyword_top_k
        self.keyword_score_threshold = keyword_score_threshold
        self.intent_recall_enabled = intent_recall_enabled
        self.intent_top_k = intent_top_k
        self.intent_score_threshold = intent_score_threshold
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.intent_weight = intent_weight
        self.llm_rerank_top_n = llm_rerank_top_n
        self._cache_signature: Optional[str] = None
        self._lexicon_cache: Optional[Dict[str, Any]] = None
        self._skill_terms_cache: Dict[str, Dict[str, Any]] = {}

    def recommend(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        understanding = self._understand_query(query)
        candidate_map: Dict[str, Dict[str, Any]] = {}

        if self.vector_recall_enabled:
            for result in self._vector_recall(understanding.rewritten_query, filters):
                self._merge_candidate(candidate_map, result, "vector_score")
            understanding.enabled_recalls.append("vector")

        if self.keyword_recall_enabled:
            for result in self._keyword_recall(understanding, filters):
                self._merge_candidate(candidate_map, result, "keyword_score")
            understanding.enabled_recalls.append("keyword")

        if self.intent_recall_enabled:
            for result in self._intent_recall(understanding, filters):
                self._merge_candidate(candidate_map, result, "intent_score")
            understanding.enabled_recalls.append("intent")

        ranked = self._rank_candidates(list(candidate_map.values()))
        if self.enable_llm_rerank and ranked:
            ranked = self._llm_rerank(query, understanding, ranked)

        logger.info(
            "[RecommendDebug] query='%s' lang=%s method=%s query_terms=%s keywords=%s task_types=%s constraints=%s",
            query,
            understanding.query_language,
            understanding.understanding_method,
            understanding.query_terms,
            understanding.keywords,
            understanding.task_types,
            understanding.constraints,
        )

        recommendations = []
        for candidate in ranked[: self.final_top_k]:
            metadata = candidate["metadata"]
            description = self._preferred_display_description(metadata, understanding.query_language)
            recommendation = Recommendation(
                skill_id=metadata.get("id", ""),
                name=metadata.get("name", ""),
                description=description,
                score=round(candidate["final_score"], 4),
                score_band=self._score_band(candidate["final_score"]),
                match_reason=self._build_match_reason(candidate),
                matched_terms=candidate.get("matched_terms", []),
                score_breakdown={
                    "vector_score": round(candidate["vector_score"], 4),
                    "keyword_score": round(candidate["keyword_score"], 4),
                    "intent_score": round(candidate["intent_score"], 4),
                },
                alternatives=[],
            )
            recommendations.append(recommendation.model_dump())
            self.catalog.update_usage(metadata.get("id"), recommended=True)
            logger.info(
                "[RecommendDebug] skill=%s matched_terms=%s breakdown=%s final_score=%.4f",
                metadata.get("id", ""),
                recommendation.matched_terms,
                recommendation.score_breakdown,
                candidate["final_score"],
            )

        return {
            "recommendations": recommendations,
            "query_understanding": {
                "original_query": query,
                **understanding.model_dump(),
                "results_count": len(recommendations),
            },
        }

    def _understand_query(self, query: str) -> QueryUnderstanding:
        query_language = self._detect_language(query)
        if not self.task_understanding_enabled:
            tokens = self._extract_keywords(query)
            return QueryUnderstanding(
                rewritten_query=query.strip(),
                entities=tokens,
                keywords=tokens,
                understanding_method="disabled",
                query_terms=tokens,
                query_language=query_language,
            )

        if self.task_understanding_use_llm_when_available and self.llm is not None:
            understanding = self._llm_understand_query(query, query_language)
            if understanding is not None:
                return understanding

        return self._heuristic_understand_query(query, query_language)

    def _llm_understand_query(self, query: str, query_language: str) -> Optional[QueryUnderstanding]:
        prompt = (
            "You are a task understanding assistant for skill recommendation.\n"
            "Infer the user's goal based on the current skill library.\n"
            "Cross-language understanding is allowed when the query and skill text use different languages.\n"
            "Use the provided skill categories, capabilities, and skill-specific constraints as the main vocabulary.\n"
            "Respond with compact JSON only.\n"
            'Schema: {"intent": str, "task_types": [str], "entities": [str], '
            '"keywords": [str], "constraints": [str], "rewritten_query": str}\n'
            f"Query language: {query_language}\n"
            f"Available skill vocabulary:\n{self._build_skill_vocabulary_summary()}"
        )
        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=query),
        ]

        try:
            response = self.llm.chat(messages, temperature=self.task_understanding_llm_temperature)
            payload = self._extract_json_object(response.content)
            if payload is None:
                return None

            data = json.loads(payload)
            normalized_query_terms = self._normalize_terms(
                data.get("keywords", []) + data.get("entities", []) + data.get("task_types", [])
            )
            return QueryUnderstanding(
                intent=data.get("intent") or "general_assistance",
                task_types=self._normalize_terms(data.get("task_types", [])),
                entities=self._normalize_terms(data.get("entities", [])),
                keywords=self._normalize_terms(data.get("keywords", []))[: self.task_understanding_max_keywords],
                constraints=self._normalize_terms(data.get("constraints", [])),
                rewritten_query=(data.get("rewritten_query") or query).strip(),
                understanding_method="llm",
                query_terms=normalized_query_terms[: self.task_understanding_max_keywords],
                query_language=query_language,
            )
        except Exception:
            return None

    def _heuristic_understand_query(self, query: str, query_language: str) -> QueryUnderstanding:
        keywords = self._extract_keywords(query)
        lexicon = self._get_skill_lexicon()
        skill_matches = self._match_skills_from_query(keywords, query_language)

        task_terms: List[str] = []
        entity_terms: List[str] = []
        constraint_terms: List[str] = []
        for skill, overlap in skill_matches:
            cached_terms = self._get_skill_terms(skill)
            task_terms.extend(cached_terms["task_terms"])
            entity_terms.extend(overlap)
            constraint_terms.extend(self._derive_constraints(skill, overlap, lexicon))

        task_types = self._rank_terms(task_terms)
        entities = self._rank_terms(entity_terms, fallback=keywords)
        constraints = self._rank_terms(constraint_terms)
        intent = task_types[0] if task_types else (entities[0] if entities else "general_assistance")

        return QueryUnderstanding(
            intent=intent,
            task_types=task_types[: self.task_understanding_max_keywords],
            entities=entities[: self.task_understanding_max_keywords],
            keywords=keywords[: self.task_understanding_max_keywords],
            constraints=constraints[: self.task_understanding_max_keywords],
            rewritten_query=query.strip(),
            understanding_method="heuristic",
            query_terms=keywords[: self.task_understanding_max_keywords],
            query_language=query_language,
        )

    def _vector_recall(self, query: str, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode_query(query)
        results = self.vector_store.search(
            query_embedding,
            top_k=self.vector_top_k,
            score_threshold=self.similarity_threshold,
        )
        return self._apply_filters(results, filters)

    def _keyword_recall(self, understanding: QueryUnderstanding, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        terms = self._normalize_terms(understanding.keywords + understanding.entities + understanding.task_types)
        if not terms:
            return []

        results = []
        for skill in self.catalog.get_all_skills():
            if not self._skill_matches_filters(skill, filters):
                continue

            score = self._keyword_match_score(skill, terms, understanding.query_language)
            if score <= 0 or score < self.keyword_score_threshold:
                continue

            results.append(
                {
                    "score": score,
                    "metadata": skill.model_dump(),
                    "matched_terms": self._get_keyword_matched_terms(skill, terms, understanding.query_language),
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[: self.keyword_top_k]

    def _intent_recall(self, understanding: QueryUnderstanding, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        intent_terms = self._normalize_terms([understanding.intent] + understanding.task_types + understanding.constraints)
        if not intent_terms:
            return []

        results = []
        for skill in self.catalog.get_all_skills():
            if not self._skill_matches_filters(skill, filters):
                continue

            score = self._intent_match_score(skill, intent_terms, understanding.query_language)
            if score <= 0 or score < self.intent_score_threshold:
                continue

            results.append(
                {
                    "score": score,
                    "metadata": skill.model_dump(),
                    "matched_terms": self._get_intent_matched_terms(skill, intent_terms, understanding.query_language),
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[: self.intent_top_k]

    def _merge_candidate(self, candidate_map: Dict[str, Dict[str, Any]], result: Dict[str, Any], score_key: str) -> None:
        metadata = result.get("metadata") or {}
        skill_id = metadata.get("id")
        if not skill_id:
            return

        candidate = candidate_map.setdefault(
            skill_id,
            {
                "metadata": metadata,
                "vector_score": 0.0,
                "keyword_score": 0.0,
                "intent_score": 0.0,
                "final_score": 0.0,
                "matched_terms": [],
            },
        )
        candidate["metadata"] = metadata
        candidate[score_key] = max(candidate.get(score_key, 0.0), float(result.get("score", 0.0)))
        candidate["matched_terms"] = self._dedupe_preserve_order(
            list(candidate.get("matched_terms", [])) + list(result.get("matched_terms", []))
        )

    def _rank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for candidate in candidates:
            candidate["final_score"] = (
                candidate["vector_score"] * self.semantic_weight
                + candidate["keyword_score"] * self.keyword_weight
                + candidate["intent_score"] * self.intent_weight
            )

        candidates.sort(key=lambda item: item["final_score"], reverse=True)
        return candidates

    def _filter_by_category(self, results: List[Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
        return [
            result
            for result in results
            if category.lower() in [value.lower() for value in result["metadata"].get("category", [])]
        ]

    def _apply_filters(self, results: List[Dict[str, Any]], filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not filters or not filters.get("category"):
            return results
        return self._filter_by_category(results, filters["category"])

    def _skill_matches_filters(self, skill: SkillSpec, filters: Optional[Dict[str, Any]]) -> bool:
        if not filters or not filters.get("category"):
            return True
        category = filters["category"].lower()
        return category in [value.lower() for value in skill.category]

    def _keyword_match_score(self, skill: SkillSpec, terms: List[str], query_language: str) -> float:
        if not terms:
            return 0.0

        cached_terms = self._get_skill_terms(skill)
        same_language_description_terms = cached_terms["description_terms_by_language"].get(query_language, set())
        if not same_language_description_terms:
            return 0.0

        terms_set = set(terms)
        same_language_name_terms = cached_terms["name_terms"] if cached_terms["name_language"] == query_language else set()

        score = 0.0
        score += 0.45 * (len(terms_set & same_language_name_terms) / len(terms_set))
        score += 0.55 * (len(terms_set & same_language_description_terms) / len(terms_set))
        return min(1.0, score)

    def _intent_match_score(self, skill: SkillSpec, intent_terms: List[str], query_language: str) -> float:
        if not intent_terms:
            return 0.0

        cached_terms = self._get_skill_terms(skill)
        same_language_description_terms = cached_terms["description_terms_by_language"].get(query_language, set())
        if self.llm is None and not same_language_description_terms:
            return 0.0

        intent_set = set(intent_terms)
        task_overlap = len(intent_set & cached_terms["task_term_set"]) / len(intent_set)
        constraint_overlap = len(intent_set & cached_terms["constraint_term_set"]) / len(intent_set)
        description_overlap = len(intent_set & same_language_description_terms) / len(intent_set)
        return min(1.0, task_overlap * 0.65 + constraint_overlap * 0.2 + description_overlap * 0.15)

    def _build_match_reason(self, candidate: Dict[str, Any]) -> str:
        reasons = []
        if candidate["vector_score"] > 0:
            reasons.append(f"semantic={candidate['vector_score']:.2f}")
        if candidate["keyword_score"] > 0:
            reasons.append(f"keyword={candidate['keyword_score']:.2f}")
        if candidate["intent_score"] > 0:
            reasons.append(f"intent={candidate['intent_score']:.2f}")
        if not reasons:
            reasons.append("matched by fallback ranking")
        return "; ".join(reasons)

    def _llm_rerank(
        self,
        query: str,
        understanding: QueryUnderstanding,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.llm or not candidates:
            return candidates[: self.final_top_k]

        shortlist = candidates[: self.llm_rerank_top_n]
        candidate_lines = []
        for candidate in shortlist:
            metadata = candidate["metadata"]
            description = self._preferred_display_description(metadata, understanding.query_language)
            candidate_lines.append(f"- {metadata.get('id')}: {metadata.get('name')} | {description}")

        prompt = (
            "You are reranking skills for a user task.\n"
            "Return only a comma-separated list of skill ids ordered from most relevant to least relevant.\n"
            f"User task: {query}\n"
            f"Query language: {understanding.query_language}\n"
            f"Intent: {understanding.intent}\n"
            f"Task types: {', '.join(understanding.task_types)}\n"
            f"Entities: {', '.join(understanding.entities)}\n"
            f"Constraints: {', '.join(understanding.constraints)}\n"
            "Candidates:\n"
            + "\n".join(candidate_lines)
        )

        try:
            response = self.llm.chat([Message(role="user", content=prompt)])
            ranked_ids = [skill_id.strip() for skill_id in response.content.split(",") if skill_id.strip()]
            candidate_by_id = {candidate["metadata"].get("id"): candidate for candidate in shortlist}

            reranked = [candidate_by_id[skill_id] for skill_id in ranked_ids if skill_id in candidate_by_id]
            for candidate in shortlist:
                if candidate["metadata"].get("id") not in ranked_ids:
                    reranked.append(candidate)
            reranked.extend(candidates[self.llm_rerank_top_n :])
            return reranked
        except Exception:
            return candidates

    def _extract_keywords(self, text: str) -> List[str]:
        tokens = self._tokenize(text)
        lexicon = self._get_skill_lexicon()
        keywords = []
        for token in tokens:
            if token.isdigit():
                continue
            if token in lexicon["task_terms"] or token in lexicon["constraint_terms"] or token in lexicon["description_terms"]:
                keywords.append(token)
                continue
            if len(token) > 2 or re.search(r"[\u4e00-\u9fff]", token):
                keywords.append(token)
        return self._dedupe_preserve_order(keywords)[: self.task_understanding_max_keywords]

    def _get_skill_lexicon(self) -> Dict[str, Any]:
        signature = self._catalog_signature()
        if self._lexicon_cache is None or self._cache_signature != signature:
            self._refresh_skill_cache(signature)
        return self._lexicon_cache or {}

    def _refresh_skill_cache(self, signature: str) -> None:
        categories: List[str] = []
        capabilities: List[str] = []
        constraint_terms: List[str] = []
        description_terms: List[str] = []
        task_terms: List[str] = []
        term_frequency: Counter = Counter()
        skill_terms_cache: Dict[str, Dict[str, Any]] = {}

        for skill in self.catalog.get_all_skills():
            skill_terms = self._build_skill_terms(skill)
            skill_terms_cache[skill.id] = skill_terms
            categories.extend(skill.category)
            capabilities.extend(skill.capabilities)
            constraint_terms.extend(skill_terms["raw_constraints"])
            description_terms.extend(skill_terms["all_description_terms"])
            task_terms.extend(skill_terms["task_terms"])
            term_frequency.update(skill_terms["task_terms"])
            term_frequency.update(skill_terms["all_description_terms"])
            term_frequency.update(skill_terms["constraint_terms"])

        self._cache_signature = signature
        self._skill_terms_cache = skill_terms_cache
        self._lexicon_cache = {
            "categories": self._dedupe_preserve_order(categories),
            "capabilities": self._dedupe_preserve_order(capabilities),
            "task_terms": set(self._normalize_terms(task_terms + categories + capabilities)),
            "constraint_terms": set(self._normalize_terms(constraint_terms)),
            "description_terms": set(self._normalize_terms(description_terms)),
            "term_frequency": term_frequency,
        }

    def _catalog_signature(self) -> str:
        serializable = []
        for skill in self.catalog.get_all_skills():
            serializable.append(
                {
                    "id": skill.id,
                    "name": skill.name,
                    "description": skill.description,
                    "enhanced_description": skill.enhanced_description,
                    "category": skill.category,
                    "capabilities": skill.capabilities,
                    "input_schema": skill.input_schema,
                    "output_schema": skill.output_schema,
                }
            )
        payload = json.dumps(serializable, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _build_skill_terms(self, skill: SkillSpec) -> Dict[str, Any]:
        name_language = self._detect_language(skill.name)
        name_terms = set(self._tokenize(skill.name))
        ordered_category_terms = self._normalize_terms(skill.category)
        ordered_capability_terms = self._normalize_terms(skill.capabilities)
        category_terms = set(ordered_category_terms)
        capability_terms = set(ordered_capability_terms)

        description_terms_by_language = {"zh": set(), "en": set()}
        for text in [skill.description, skill.enhanced_description]:
            if not text:
                continue
            language = self._detect_language(text)
            description_terms_by_language[language].update(self._tokenize(text))

        all_description_terms = set().union(*description_terms_by_language.values())
        schema_terms = set(
            self._tokenize(
                json.dumps(
                    {
                        "input_schema": skill.input_schema,
                        "output_schema": skill.output_schema,
                    },
                    ensure_ascii=False,
                )
            )
        )
        raw_constraints = self._extract_structured_constraints(skill)
        constraint_terms = set(self._normalize_terms(raw_constraints)) | schema_terms
        task_terms = self._dedupe_preserve_order(ordered_category_terms + ordered_capability_terms)
        task_term_set = set(task_terms)
        all_terms = task_term_set | name_terms | all_description_terms | constraint_terms
        return {
            "name_language": name_language,
            "name_terms": name_terms,
            "category_terms": category_terms,
            "capability_terms": capability_terms,
            "description_terms_by_language": description_terms_by_language,
            "all_description_terms": all_description_terms,
            "constraint_terms": constraint_terms,
            "task_terms": task_terms,
            "task_term_set": task_term_set,
            "constraint_term_set": constraint_terms,
            "all_terms": all_terms,
            "raw_constraints": raw_constraints,
        }

    def _get_skill_terms(self, skill: SkillSpec) -> Dict[str, Any]:
        self._get_skill_lexicon()
        return self._skill_terms_cache.get(skill.id) or self._build_skill_terms(skill)

    def _get_keyword_matched_terms(self, skill: SkillSpec, terms: List[str], query_language: str) -> List[str]:
        cached_terms = self._get_skill_terms(skill)
        same_language_description_terms = cached_terms["description_terms_by_language"].get(query_language, set())
        if not same_language_description_terms:
            return []

        combined = set(same_language_description_terms)
        if cached_terms["name_language"] == query_language:
            combined |= cached_terms["name_terms"]
        return [term for term in terms if term in combined]

    def _get_intent_matched_terms(self, skill: SkillSpec, terms: List[str], query_language: str) -> List[str]:
        cached_terms = self._get_skill_terms(skill)
        same_language_description_terms = cached_terms["description_terms_by_language"].get(query_language, set())
        combined = cached_terms["task_term_set"] | cached_terms["constraint_term_set"] | same_language_description_terms
        return [term for term in terms if term in combined]

    def _extract_structured_constraints(self, skill: SkillSpec) -> List[str]:
        values = []
        for container in (skill.input_schema, skill.output_schema):
            values.extend(self._flatten_constraint_values(container))

        structured_terms = [
            term
            for term in self._normalize_terms(values)
            if len(term) > 2 or re.search(r"[\u4e00-\u9fff]", term)
        ]

        capability_terms = self._normalize_terms(skill.capabilities)
        category_terms = set(self._normalize_terms(skill.category))
        specific_capabilities = [term for term in capability_terms if term not in category_terms and len(term) > 2]

        return self._dedupe_preserve_order(structured_terms + specific_capabilities)

    def _flatten_constraint_values(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, dict):
            values: List[str] = []
            for key, nested in value.items():
                values.append(str(key))
                values.extend(self._flatten_constraint_values(nested))
            return values
        if isinstance(value, list):
            values: List[str] = []
            for item in value:
                values.extend(self._flatten_constraint_values(item))
            return values
        return [str(value)]

    def _match_skills_from_query(self, keywords: List[str], query_language: str) -> List[tuple[SkillSpec, List[str]]]:
        if not keywords:
            return []

        results = []
        for skill in self.catalog.get_all_skills():
            cached_terms = self._get_skill_terms(skill)
            overlap = [term for term in keywords if term in cached_terms["all_terms"]]
            if self.llm is None and not cached_terms["description_terms_by_language"].get(query_language):
                overlap = [
                    term
                    for term in overlap
                    if term in cached_terms["task_term_set"] or term in cached_terms["constraint_term_set"]
                ]
            if overlap:
                results.append((skill, overlap))

        results.sort(key=lambda item: len(item[1]), reverse=True)
        return results

    def _derive_constraints(self, skill: SkillSpec, overlap: List[str], lexicon: Dict[str, Any]) -> List[str]:
        cached_terms = self._get_skill_terms(skill)
        term_frequency: Counter = lexicon["term_frequency"]
        constraints = []
        for term in overlap:
            if term in cached_terms["category_terms"]:
                continue
            if term in cached_terms["constraint_terms"]:
                constraints.append(term)
                continue
            if term in cached_terms["capability_terms"] and term_frequency.get(term, 0) <= 2:
                constraints.append(term)
        return self._dedupe_preserve_order(constraints)

    def _rank_terms(self, terms: List[str], fallback: Optional[List[str]] = None) -> List[str]:
        ranked = [term for term, _count in Counter(terms).most_common()]
        if ranked:
            return ranked
        return fallback or []

    def _build_skill_vocabulary_summary(self) -> str:
        lexicon = self._get_skill_lexicon()
        categories = ", ".join(lexicon["categories"][:20]) or "none"
        capabilities = ", ".join(lexicon["capabilities"][:30]) or "none"
        constraints = ", ".join(sorted(lexicon["constraint_terms"])[:30]) or "none"
        return f"Categories: {categories}\nCapabilities: {capabilities}\nConstraints: {constraints}"

    def _preferred_display_description(self, metadata: Dict[str, Any], query_language: str) -> str:
        description = metadata.get("description") or ""
        enhanced_description = metadata.get("enhanced_description") or ""
        if enhanced_description and self._detect_language(enhanced_description) == query_language:
            return enhanced_description
        if description and self._detect_language(description) == query_language:
            return description
        return enhanced_description or description

    def _score_band(self, score: float) -> str:
        if score >= 0.65:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"

    def _detect_language(self, text: str) -> str:
        if re.search(r"[\u4e00-\u9fff]", text or ""):
            return "zh"
        return "en"

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", (text or "").lower()) if token]

    def _normalize_terms(self, values: List[Any]) -> List[str]:
        normalized: List[str] = []
        for value in values:
            if not value:
                continue
            normalized.extend(self._tokenize(str(value)))
        return self._dedupe_preserve_order(normalized)

    def _dedupe_preserve_order(self, values: List[str]) -> List[str]:
        seen = set()
        ordered = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _extract_json_object(self, text: str) -> Optional[str]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

    def build_index(self) -> None:
        skills = self.catalog.get_all_skills()
        if not skills:
            return

        texts = [skill.get_text_for_embedding() for skill in skills]
        embeddings = self.embedding_model.encode(texts)
        metadata = [skill.model_dump() for skill in skills]
        self.vector_store.add_vectors(embeddings, metadata)
