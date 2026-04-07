import hashlib
import json
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence

import yaml

from ..catalog import SkillSpec
from ..logger import logger
from ..monitoring import FeedbackData


class SkillStorage:
    """Skill 存储管理器，使用 SQLite 作为真源，JSON 作为可读缓存。"""

    def __init__(
        self,
        json_enabled: bool = True,
        json_path: str = "./data/skills.json",
        db_enabled: bool = True,
        db_path: str = "./data/skills.db",
        vector_enabled: bool = True,
        vector_index_path: str = "./data/faiss_index",
    ):
        self.json_enabled = json_enabled
        self.json_path = json_path
        self.db_enabled = db_enabled
        self.db_path = db_path
        self.vector_enabled = vector_enabled
        self.vector_index_path = vector_index_path
        self.vector_state_path = f"{vector_index_path}.state.json"

        self._ensure_directories()

    def _ensure_directories(self):
        for path in [self.json_path, self.db_path, self.vector_index_path, self.vector_state_path]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    # ==================== skills-hub ====================

    def _find_all_skill_md(self, dir_path: Path) -> List[Path]:
        skill_md_path = dir_path / "SKILL.md"
        if skill_md_path.exists():
            return [skill_md_path]

        results = []
        for sub_dir in dir_path.iterdir():
            if sub_dir.is_dir():
                results.extend(self._find_all_skill_md(sub_dir))
        return results

    def _scan_hub_directory(self, hub_path: Path) -> tuple[List[Dict[str, Any]], List[str]]:
        skills = []
        invalid_dirs = []

        for item in hub_path.iterdir():
            if not item.is_dir():
                continue

            skill_md_paths = self._find_all_skill_md(item)
            if skill_md_paths:
                for skill_md_path in skill_md_paths:
                    try:
                        skill_id = skill_md_path.parent.name
                        skill_data = self._parse_skill_md(skill_id, skill_md_path)
                        if skill_data:
                            skills.append(skill_data)
                    except Exception as exc:
                        logger.warning("[Storage] Failed to parse %s: %s", skill_md_path.parent.name, exc)
            else:
                invalid_dirs.append(item.name)

        return skills, invalid_dirs

    def get_hub_skills(self, skills_hub_path: str = "./skills-hub") -> List[Dict[str, Any]]:
        hub_path = Path(skills_hub_path)
        if not hub_path.exists():
            logger.warning("[Storage] Skills hub path not found: %s", skills_hub_path)
            return []

        skills, invalid_dirs = self._scan_hub_directory(hub_path)
        if invalid_dirs:
            logger.warning("[Storage] 无法注册的目录（未找到SKILL.md）: %s", ", ".join(invalid_dirs))
        return skills

    def import_skills(
        self,
        skills_hub_path: str = "./skills-hub",
        skill_id: Optional[str] = None,
        existing_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        hub_path = Path(skills_hub_path)
        if not hub_path.exists():
            return {"status": "error", "message": f"Skills hub path not found: {skills_hub_path}"}

        existing_ids = set(existing_ids or [])
        imported_skills: List[Dict[str, Any]] = []
        invalid_dirs: List[str] = []

        if skill_id:
            skill_dir = hub_path / skill_id
            skill_md_paths = self._find_all_skill_md(skill_dir) if skill_dir.exists() else []
            if not skill_dir.exists():
                return {"status": "error", "message": f"Skill folder not found: {skill_id}"}
            if not skill_md_paths:
                return {"status": "error", "message": f"SKILL.md not found in: {skill_id}"}

            for skill_md_path in skill_md_paths:
                try:
                    parsed_skill_id = skill_md_path.parent.name
                    skill_data = self._parse_skill_md(parsed_skill_id, skill_md_path)
                    if skill_data and skill_data["id"] not in existing_ids:
                        imported_skills.append(skill_data)
                except Exception as exc:
                    logger.warning("[Storage] Failed to import %s: %s", skill_md_path.parent.name, exc)
        else:
            skills, invalid_dirs = self._scan_hub_directory(hub_path)
            imported_skills = [skill_data for skill_data in skills if skill_data["id"] not in existing_ids]

        if not imported_skills and skill_id:
            return {"status": "skipped", "message": f"Skill already registered: {skill_id}"}
        if not imported_skills:
            return {"status": "skipped", "message": "所有技能已注册"}

        result = {
            "status": "success",
            "imported_count": len(imported_skills),
            "imported_skills": imported_skills,
            "invalid_dirs": invalid_dirs,
            "message": f"成功导入 {len(imported_skills)} 个技能",
        }
        if invalid_dirs:
            result["message"] += f"，{len(invalid_dirs)} 个目录无法注册"
        return result

    def update_all_skills(self, skills_hub_path: str = "./skills-hub") -> Dict[str, Any]:
        hub_path = Path(skills_hub_path)
        if not hub_path.exists():
            return {"status": "error", "message": f"Skills hub path not found: {skills_hub_path}"}

        updated_skills, invalid_dirs = self._scan_hub_directory(hub_path)
        logger.info("[Storage] 更新了 %s 个技能", len(updated_skills))
        for skill_data in updated_skills[:5]:
            logger.info("  - %s", skill_data["name"])
        if len(updated_skills) > 5:
            logger.info("  ... 还有 %s 个", len(updated_skills) - 5)
        if invalid_dirs:
            logger.warning("[Storage] 无法注册的目录（未找到SKILL.md）: %s", ", ".join(invalid_dirs))

        return {
            "status": "success",
            "updated_count": len(updated_skills),
            "updated_skills": updated_skills,
            "invalid_dirs": invalid_dirs,
            "message": f"成功更新 {len(updated_skills)} 个技能",
        }

    def _parse_skill_md(self, skill_id: str, skill_md_path: Path) -> Optional[Dict[str, Any]]:
        content = skill_md_path.read_text(encoding="utf-8")
        if not content.startswith("---"):
            return None

        parts = content.split("---", 2)
        if len(parts) < 3:
            return None

        frontmatter = parts[1].strip()
        try:
            metadata = yaml.safe_load(frontmatter) or {}
        except Exception:
            metadata = {}

        name = metadata.get("name", skill_id)
        description = metadata.get("description", "")
        if len(description) > 500:
            description = f"{description[:500]}..."

        body = parts[2].strip()
        for line in body.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("="):
                if len(line) > len(description):
                    description = line
                break

        category, capabilities = self._derive_skill_taxonomy(
            skill_id=skill_id,
            name=name,
            description=description,
            metadata=metadata,
            body=body,
        )

        return {
            "id": f"hub_{skill_id}",
            "name": name,
            "description": description,
            "enhanced_description": None,
            "category": category,
            "capabilities": capabilities,
            "input_schema": {},
            "output_schema": {},
            "version": "1.0.0",
            "dependencies": [],
            "author": None,
            "source_path": str(skill_md_path.parent),
            "raw_content": content,
        }

    def _derive_skill_taxonomy(
        self,
        skill_id: str,
        name: str,
        description: str,
        metadata: Dict[str, Any],
        body: str,
    ) -> tuple[List[str], List[str]]:
        explicit_categories = metadata.get("category") or metadata.get("categories") or []
        explicit_capabilities = metadata.get("capabilities") or []
        if isinstance(explicit_categories, str):
            explicit_categories = [explicit_categories]
        if isinstance(explicit_capabilities, str):
            explicit_capabilities = [explicit_capabilities]

        keyword_phrases = self._extract_keyword_phrases(body, metadata)
        summary_phrases = self._extract_summary_phrases(skill_id, name)
        description_terms = self._token_candidates(description)
        body_phrases = self._extract_body_capabilities(body)

        categories = self._dedupe_text_list(
            explicit_categories + summary_phrases[:2] + keyword_phrases[:2]
        )
        capabilities = self._dedupe_text_list(
            explicit_capabilities + keyword_phrases + body_phrases + description_terms[:4] + summary_phrases
        )

        if not categories:
            categories = [name.replace("-", " ").strip()]
        if not capabilities:
            capabilities = self._token_candidates(f"{name} {description}")[:6]

        return categories[:3], capabilities[:8]

    def _extract_keyword_phrases(self, body: str, metadata: Dict[str, Any]) -> List[str]:
        phrases: List[str] = []
        metadata_keywords = metadata.get("keywords") or []
        if isinstance(metadata_keywords, str):
            metadata_keywords = [item.strip() for item in metadata_keywords.split(",") if item.strip()]
        phrases.extend(metadata_keywords)

        keyword_match = re.search(r"\*\*Keywords\*\*:\s*(.+)", body, flags=re.IGNORECASE)
        if keyword_match:
            phrases.extend([item.strip() for item in keyword_match.group(1).split(",") if item.strip()])
        return self._dedupe_text_list(phrases)

    def _extract_summary_phrases(self, skill_id: str, name: str) -> List[str]:
        return self._dedupe_text_list([skill_id.replace("-", " "), name.replace("-", " ")])

    def _extract_body_capabilities(self, body: str) -> List[str]:
        phrases: List[str] = []
        in_target_section = False
        for raw_line in body.splitlines():
            line = raw_line.strip()
            lower_line = line.lower()
            if lower_line.startswith("## when to use") or lower_line.startswith("## what this skill does"):
                in_target_section = True
                continue
            if in_target_section and line.startswith("## "):
                in_target_section = False
            if not in_target_section:
                continue
            if line.startswith("- "):
                phrases.append(line[2:].strip().rstrip("."))
            elif re.match(r"^\d+\.\s+", line):
                phrases.append(re.sub(r"^\d+\.\s+", "", line).strip().rstrip("."))
        return self._dedupe_text_list(phrases)

    def _token_candidates(self, text: str) -> List[str]:
        stopwords = {
            "a", "an", "and", "are", "for", "from", "help", "into", "not", "or", "that", "the", "this",
            "to", "use", "using", "when", "with", "your",
        }
        tokens = []
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower()):
            normalized = token.replace("_", " ").replace("-", " ").strip()
            if normalized in stopwords:
                continue
            if len(normalized) <= 2:
                continue
            tokens.append(normalized)
        return self._dedupe_text_list(tokens)

    def _dedupe_text_list(self, values: List[str]) -> List[str]:
        seen = set()
        result = []
        for value in values:
            cleaned = " ".join(str(value).strip().split())
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            result.append(cleaned)
        return result

    def _detect_language(self, text: str) -> str:
        if re.search(r"[\u4e00-\u9fff]", text or ""):
            return "zh"
        return "en"

    def enrich_skill_with_llm(self, skill_data: Dict[str, Any], llm_adapter) -> Dict[str, Any]:
        if not llm_adapter:
            return skill_data

        skill_name = skill_data.get("name", "")
        skill_desc = skill_data.get("description", "")
        raw_text = f"{skill_name}: {skill_desc}".strip(": ")
        source_language = self._detect_language(raw_text)
        target_language = "English" if source_language == "zh" else "Chinese"
        prompt = (
            "Translate the following skill summary into the target language.\n"
            "Keep the output as a single concise line in the format 'Name: Description'.\n"
            "Return only the translated line with no markdown or explanation.\n"
            f"Source language: {source_language}\n"
            f"Target language: {target_language}\n"
            f"Input: {raw_text}"
        )

        try:
            from ..llm import Message

            response = llm_adapter.chat([Message(role="user", content=prompt)])
            translated_text = response.content.strip().strip("`").strip()
            skill_data["enhanced_description"] = translated_text or None
            logger.info("[Storage] LLM translated skill summary: %s", skill_name)
        except Exception as exc:
            logger.warning("[Storage] LLM translation failed for %s: %s", skill_name, exc)
        finally:
            skill_data.pop("raw_content", None)

        return skill_data

    # ==================== JSON ====================

    def save_to_json(self, skills: List[SkillSpec]) -> None:
        if not self.json_enabled:
            return

        data = {
            "updated_at": datetime.now().isoformat(),
            "skills": [self._serialize_skill(skill) for skill in skills],
        }
        with open(self.json_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        logger.info("[Storage] Saved %s skills to JSON: %s", len(skills), self.json_path)

    def load_from_json(self) -> List[Dict[str, Any]]:
        if not self.json_enabled or not Path(self.json_path).exists():
            return []

        try:
            with open(self.json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("[Storage] Failed to load JSON: %s, will skip and try database", exc)
            return []

        logger.info("[Storage] Loaded %s skills from JSON", len(data.get("skills", [])))
        return data.get("skills", [])

    # ==================== Database ====================

    def _init_database(self) -> None:
        if not self.db_enabled:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                enhanced_description TEXT,
                category TEXT,
                capabilities TEXT,
                input_schema TEXT,
                output_schema TEXT,
                version TEXT,
                dependencies TEXT,
                author TEXT,
                created_at TEXT,
                updated_at TEXT,
                usage_stats TEXT
            )
            """
        )
        self._migrate_skills_table(cursor)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recommendation_id TEXT,
                user_id TEXT,
                rating INTEGER,
                comment TEXT,
                selected_skill TEXT,
                timestamp TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def _migrate_skills_table(self, cursor: sqlite3.Cursor) -> None:
        columns = {
            row[1]
            for row in cursor.execute("PRAGMA table_info(skills)").fetchall()
        }
        required_columns = {
            "enhanced_description": "TEXT",
            "input_schema": "TEXT",
            "output_schema": "TEXT",
            "usage_stats": "TEXT",
        }
        for column_name, column_type in required_columns.items():
            if column_name in columns:
                continue
            cursor.execute(f"ALTER TABLE skills ADD COLUMN {column_name} {column_type}")

    def save_to_database(self, skills: List[SkillSpec]) -> None:
        if not self.db_enabled:
            return

        self._init_database()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for skill in skills:
            cursor.execute(
                """
                INSERT OR REPLACE INTO skills
                (id, name, description, enhanced_description, category, capabilities, input_schema,
                 output_schema, version, dependencies, author, created_at, updated_at, usage_stats)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    skill.id,
                    skill.name,
                    skill.description,
                    skill.enhanced_description,
                    json.dumps(skill.category),
                    json.dumps(skill.capabilities),
                    json.dumps(skill.input_schema),
                    json.dumps(skill.output_schema),
                    skill.version,
                    json.dumps(skill.dependencies),
                    skill.author,
                    skill.created_at.isoformat(),
                    skill.updated_at.isoformat(),
                    json.dumps(skill.usage_stats),
                ),
            )
        conn.commit()
        conn.close()
        logger.info("[Storage] Saved %s skills to database: %s", len(skills), self.db_path)

    def load_from_database(self) -> List[Dict[str, Any]]:
        if not self.db_enabled or not Path(self.db_path).exists():
            return []

        self._init_database()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM skills").fetchall()
        conn.close()

        skills = []
        for row in rows:
            skill = dict(row)
            skill["category"] = json.loads(skill["category"]) if skill["category"] else []
            skill["capabilities"] = json.loads(skill["capabilities"]) if skill["capabilities"] else []
            skill["input_schema"] = json.loads(skill["input_schema"]) if skill["input_schema"] else {}
            skill["output_schema"] = json.loads(skill["output_schema"]) if skill["output_schema"] else {}
            skill["dependencies"] = json.loads(skill["dependencies"]) if skill["dependencies"] else []
            skill["usage_stats"] = json.loads(skill["usage_stats"]) if skill["usage_stats"] else {}
            skills.append(skill)

        logger.info("[Storage] Loaded %s skills from database", len(skills))
        return skills

    def query_skills_from_database(self, category: Optional[str] = None, keyword: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.db_enabled or not Path(self.db_path).exists():
            return []

        self._init_database()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM skills WHERE 1=1"
        params: List[str] = []
        if category:
            query += " AND category LIKE ?"
            params.append(f"%{category}%")
        if keyword:
            query += " AND (name LIKE ? OR description LIKE ? OR enhanced_description LIKE ?)"
            params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])

        rows = cursor.execute(query, params).fetchall()
        conn.close()

        skills = []
        for row in rows:
            skill = dict(row)
            skill["category"] = json.loads(skill["category"]) if skill["category"] else []
            skill["capabilities"] = json.loads(skill["capabilities"]) if skill["capabilities"] else []
            skills.append(skill)
        return skills

    def save_feedback(self, feedback: FeedbackData) -> None:
        if not self.db_enabled:
            return

        self._init_database()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO feedback (recommendation_id, user_id, rating, comment, selected_skill, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                feedback.recommendation_id,
                feedback.user_id,
                feedback.rating,
                feedback.comment,
                feedback.selected_skill,
                feedback.timestamp.isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    def get_feedback_stats(self) -> Dict[str, Any]:
        if not self.db_enabled or not Path(self.db_path).exists():
            return {"total": 0, "avg_rating": 0, "by_rating": {}}

        self._init_database()
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT rating FROM feedback").fetchall()
        conn.close()
        if not rows:
            return {"total": 0, "avg_rating": 0, "by_rating": {}}

        ratings = [row[0] for row in rows]
        return {
            "total": len(ratings),
            "avg_rating": sum(ratings) / len(ratings),
            "by_rating": {i: ratings.count(i) for i in range(1, 6)},
        }

    # ==================== Vector Index State ====================

    def get_vector_index_path(self) -> str:
        return self.vector_index_path

    def vector_index_exists(self) -> bool:
        return Path(f"{self.vector_index_path}.index").exists()

    def _skill_to_state_dict(self, skill: SkillSpec | Dict[str, Any]) -> Dict[str, Any]:
        data = skill.model_dump() if isinstance(skill, SkillSpec) else dict(skill)
        return {
            "id": data.get("id"),
            "description": data.get("description"),
            "enhanced_description": data.get("enhanced_description"),
            "category": data.get("category", []),
            "capabilities": data.get("capabilities", []),
            "updated_at": str(data.get("updated_at")),
        }

    def build_vector_state(self, skills: Sequence[SkillSpec | Dict[str, Any]], dimension: int, model_path: str) -> Dict[str, Any]:
        serializable = [self._skill_to_state_dict(skill) for skill in skills]
        fingerprint = hashlib.sha256(
            json.dumps(serializable, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "fingerprint": fingerprint,
            "skills_count": len(serializable),
            "dimension": dimension,
            "model_path": model_path,
            "updated_at": datetime.now().isoformat(),
        }

    def save_vector_state(self, skills: Sequence[SkillSpec | Dict[str, Any]], dimension: int, model_path: str) -> Dict[str, Any]:
        state = self.build_vector_state(skills, dimension, model_path)
        with open(self.vector_state_path, "w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)
        return state

    def load_vector_state(self) -> Optional[Dict[str, Any]]:
        path = Path(self.vector_state_path)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[Storage] Failed to load vector state: %s", exc)
            return None

    def is_vector_index_consistent(self, skills: Sequence[SkillSpec | Dict[str, Any]], dimension: int, model_path: str) -> bool:
        if not self.vector_enabled:
            return False

        index_path = Path(f"{self.vector_index_path}.index")
        meta_path = Path(f"{self.vector_index_path}.meta")
        if not index_path.exists() or not meta_path.exists():
            return False

        current_state = self.build_vector_state(skills, dimension, model_path)
        saved_state = self.load_vector_state()
        if not saved_state:
            return False

        return (
            saved_state.get("fingerprint") == current_state["fingerprint"]
            and saved_state.get("skills_count") == current_state["skills_count"]
            and saved_state.get("dimension") == current_state["dimension"]
            and saved_state.get("model_path") == current_state["model_path"]
        )

    def _vector_meta_path(self) -> Path:
        return Path(f"{self.vector_index_path}.meta")

    def load_vector_metadata(self) -> List[Dict[str, Any]]:
        meta_path = self._vector_meta_path()
        if not meta_path.exists():
            return []
        try:
            with open(meta_path, "rb") as file:
                return pickle.load(file)
        except (OSError, pickle.PickleError, EOFError) as exc:
            logger.warning("[Storage] Failed to load vector metadata: %s", exc)
            return []

    def save_vector_metadata(self, skills: Sequence[SkillSpec | Dict[str, Any]]) -> None:
        if not self.vector_enabled or not self.vector_index_exists():
            return
        meta_path = self._vector_meta_path()
        with open(meta_path, "wb") as file:
            pickle.dump([self._serialize_skill_payload(skill) for skill in skills], file)

    def is_vector_metadata_consistent(self, skills: Sequence[SkillSpec | Dict[str, Any]]) -> bool:
        if not self.vector_enabled or not self.vector_index_exists():
            return False
        stored = self.load_vector_metadata()
        expected = [self._serialize_skill_payload(skill) for skill in skills]
        return stored == expected

    # ==================== Unified Interface ====================

    def load_skills(self) -> List[Dict[str, Any]]:
        # SQLite is the source of truth; JSON is fallback cache/export.
        skills = self.load_from_database()
        if skills:
            return skills
        return self.load_from_json()

    def save_skills(self, skills: List[SkillSpec]) -> None:
        if self.db_enabled:
            self.save_to_database(skills)
        if self.json_enabled:
            self.save_to_json(skills)
        logger.info("[Storage] Skills saved with SQLite as source of truth and JSON as cache")

    def _serialize_skill(self, skill: SkillSpec) -> Dict[str, Any]:
        data = self._serialize_skill_payload(skill)
        if data.get("created_at"):
            data["created_at"] = skill.created_at.isoformat()
        if data.get("updated_at"):
            data["updated_at"] = skill.updated_at.isoformat()
        return data

    def _serialize_skill_payload(self, skill: SkillSpec | Dict[str, Any]) -> Dict[str, Any]:
        return skill.model_dump() if isinstance(skill, SkillSpec) else dict(skill)
