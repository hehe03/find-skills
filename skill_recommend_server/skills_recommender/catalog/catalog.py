from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SkillSpec(BaseModel):
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
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    usage_stats: Dict[str, int] = Field(default_factory=lambda: {"times_recommended": 0, "times_selected": 0})

    def get_text_for_embedding(self) -> str:
        description_text = self.enhanced_description or self.description
        parts = [
            self.name,
            description_text,
            f"categories: {', '.join(self.category)}",
            f"capabilities: {', '.join(self.capabilities)}",
        ]
        return " | ".join(parts)


class SkillCatalog:
    def __init__(self):
        self.skills: Dict[str, SkillSpec] = {}

    def add_skill(self, skill: SkillSpec) -> None:
        self.skills[skill.id] = skill

    def get_skill(self, skill_id: str) -> Optional[SkillSpec]:
        return self.skills.get(skill_id)

    def get_all_skills(self) -> List[SkillSpec]:
        return list(self.skills.values())
    
    def count(self) -> int:
        return len(self.skills)

    def search_by_category(self, category: str) -> List[SkillSpec]:
        return [s for s in self.skills.values() if category.lower() in [c.lower() for c in s.category]]

    def update_usage(self, skill_id: str, recommended: bool = False, selected: bool = False) -> None:
        if skill_id in self.skills:
            if recommended:
                self.skills[skill_id].usage_stats["times_recommended"] += 1
            if selected:
                self.skills[skill_id].usage_stats["times_selected"] += 1
            self.skills[skill_id].updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {sid: skill.model_dump() for sid, skill in self.skills.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillCatalog":
        catalog = cls()
        for sid, sdata in data.items():
            skill = SkillSpec(**sdata)
            catalog.add_skill(skill)
        return catalog
