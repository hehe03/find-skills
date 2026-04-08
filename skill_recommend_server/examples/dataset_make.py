import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from skills_recommender.config import settings
from skills_recommender.llm import LLMFactory, Message


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKILLS_HUB = PROJECT_ROOT / "skills-hub"
DEFAULT_OUTPUT = PROJECT_ROOT / "examples" / "skill_recommend_dataset.json"
DEFAULT_MISSING = PROJECT_ROOT / "examples" / "missing_skill_md.json"


def build_llm(provider: Optional[str] = None):
    provider_name = provider or settings.llm.default
    if provider_name not in settings.llm.providers:
        raise ValueError(f"Unknown LLM provider: {provider_name}")

    provider_config = settings.llm.providers[provider_name]
    if not provider_config.api_key and provider_name not in {"ollama"}:
        raise ValueError(
            f"LLM provider '{provider_name}' has no API key configured. "
            "Please configure the key in environment variables or config.yaml."
        )

    return LLMFactory.create(
        provider_name,
        api_key=provider_config.api_key,
        base_url=provider_config.base_url,
        model=provider_config.model,
        temperature=provider_config.temperature,
    )


def find_all_skill_md(dir_path: Path) -> List[Path]:
    skill_md_path = dir_path / "SKILL.md"
    if skill_md_path.exists():
        return [skill_md_path]

    results: List[Path] = []
    for sub_dir in dir_path.iterdir():
        if sub_dir.is_dir():
            results.extend(find_all_skill_md(sub_dir))
    return results


def parse_skill_markdown(skill_md_path: Path) -> Optional[Dict[str, Any]]:
    content = skill_md_path.read_text(encoding="utf-8")
    if not content.startswith("---"):
        return None

    parts = content.split("---", 2)
    if len(parts) < 3:
        return None

    try:
        metadata = yaml.safe_load(parts[1].strip()) or {}
    except Exception:
        metadata = {}

    body = parts[2].strip()
    folder_name = skill_md_path.parent.name
    name = metadata.get("name", folder_name)
    description = metadata.get("description", "")

    if not description:
        for line in body.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("="):
                description = line
                break

    return {
        "skill_id": f"hub_{folder_name}",
        "folder_name": folder_name,
        "name": name,
        "description": description,
        "body": body[:4000],
        "source_path": str(skill_md_path),
    }


def scan_skills_hub(skills_hub_path: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    valid_skills: List[Dict[str, Any]] = []
    missing_skill_md: List[Dict[str, str]] = []

    for item in sorted(skills_hub_path.iterdir(), key=lambda path: path.name.lower()):
        if not item.is_dir():
            continue

        skill_md_paths = find_all_skill_md(item)
        if not skill_md_paths:
            missing_skill_md.append(
                {
                    "folder_name": item.name,
                    "path": str(item),
                    "reason": "SKILL.md not found",
                }
            )
            continue

        for skill_md_path in skill_md_paths:
            parsed = parse_skill_markdown(skill_md_path)
            if parsed:
                valid_skills.append(parsed)

    return valid_skills, missing_skill_md


def generate_examples_for_skill(llm, skill: Dict[str, Any], samples_per_skill: int) -> List[Dict[str, Any]]:
    prompt = f"""
You are creating an evaluation dataset for a skill recommendation system.

Given the following skill, generate {samples_per_skill} realistic user requests that should clearly match this skill.

Requirements:
1. Each sample must represent a realistic user need or task request, not a formal benchmark sentence.
2. Use Chinese as the main language. More than 80% of samples should be in Chinese.
3. Keep a small minority of samples in English or mixed Chinese-English when natural.
4. Vary scenarios, tone, and phrasing. Include short requests, long requests, direct commands, question style, and conversational style.
5. Simulate a small amount of real-world noise, such as occasional typos, omitted punctuation, colloquial wording, or slightly incomplete context.
6. The correct label for every sample is the provided skill_id.
7. Make the requests specific enough that this skill is a strong match.
8. Return JSON only.

JSON schema:
[
  {{
    "query": "user request",
    "label": "{skill["skill_id"]}",
    "reason": "short explanation",
    "sample_type": "single_skill"
  }}
]

skill_id: {skill["skill_id"]}
name: {skill["name"]}
description: {skill["description"]}
source_path: {skill["source_path"]}
skill_body:
{skill["body"]}
""".strip()

    response = llm.chat([Message(role="user", content=prompt)])
    content = response.content.strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0].strip()

    data = json.loads(content)
    results: List[Dict[str, Any]] = []
    for item in data:
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        results.append(
            {
                "query": query,
                "label": skill["skill_id"],
                "reason": str(item.get("reason", "")).strip(),
                "sample_type": str(item.get("sample_type", "single_skill")).strip() or "single_skill",
                "skill_name": skill["name"],
                "source_path": skill["source_path"],
            }
        )
    return results


def generate_multi_skill_examples(llm, skills: List[Dict[str, Any]], total_samples: int) -> List[Dict[str, Any]]:
    if total_samples <= 0 or not skills:
        return []

    skill_summaries = []
    for skill in skills[:80]:
        skill_summaries.append(
            f'- {skill["skill_id"]} | {skill["name"]} | {skill["description"]}'
        )

    prompt = f"""
You are creating a hard evaluation dataset for a skill recommendation system.

Generate {total_samples} realistic complex user requests that require combining multiple skills.

Requirements:
1. These are multi-skill requests, not single-skill requests.
2. Use Chinese as the main language. More than 80% of samples should be in Chinese.
3. Keep a small minority of samples in English or mixed Chinese-English when natural.
4. Scenarios must be diverse, realistic, and practical.
5. Vary phrasing styles: command, question, conversational, terse, detailed, partially specified.
6. Simulate a small amount of real-world noise, such as occasional typos, omitted punctuation, or colloquial expressions.
7. For each sample, provide 2-3 correct skill ids in execution order.
8. Return JSON only.

JSON schema:
[
  {{
    "query": "user request",
    "labels": ["skill_id_1", "skill_id_2"],
    "reason": "short explanation",
    "sample_type": "multi_skill"
  }}
]

Available skills:
{chr(10).join(skill_summaries)}
""".strip()

    response = llm.chat([Message(role="user", content=prompt)])
    content = response.content.strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0].strip()

    data = json.loads(content)
    results: List[Dict[str, Any]] = []
    known_skill_ids = {skill["skill_id"] for skill in skills}
    for item in data:
        query = str(item.get("query", "")).strip()
        labels = [str(label).strip() for label in item.get("labels", []) if str(label).strip() in known_skill_ids]
        if not query or len(labels) < 2:
            continue
        results.append(
            {
                "query": query,
                "labels": labels,
                "reason": str(item.get("reason", "")).strip(),
                "sample_type": "multi_skill",
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a labeled evaluation dataset for skill recommendation."
    )
    parser.add_argument(
        "--skills-hub",
        type=Path,
        default=DEFAULT_SKILLS_HUB,
        help="Path to the skills-hub directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to save generated dataset JSON.",
    )
    parser.add_argument(
        "--missing-output",
        type=Path,
        default=DEFAULT_MISSING,
        help="Path to save folders that do not contain SKILL.md.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Optional LLM provider override.",
    )
    parser.add_argument(
        "--samples-per-skill",
        type=int,
        default=3,
        choices=[3, 4, 5],
        help="Number of labeled examples to create per skill.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of skills to process.",
    )
    args = parser.parse_args()

    if not args.skills_hub.exists():
        raise SystemExit(f"skills-hub path not found: {args.skills_hub}")

    llm = build_llm(args.provider)
    skills, missing_skill_md = scan_skills_hub(args.skills_hub)
    if args.limit is not None:
        skills = skills[: args.limit]

    dataset: List[Dict[str, Any]] = []
    multi_skill_dataset: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for skill in skills:
        try:
            dataset.extend(generate_examples_for_skill(llm, skill, args.samples_per_skill))
            print(f"[OK] {skill['skill_id']} -> {skill['name']}")
        except Exception as exc:
            errors.append(
                {
                    "skill_id": skill["skill_id"],
                    "source_path": skill["source_path"],
                    "error": str(exc),
                }
            )
            print(f"[ERROR] {skill['skill_id']}: {exc}")

    multi_skill_target = max(1, round(len(dataset) * 0.1)) if dataset else 0
    try:
        multi_skill_dataset = generate_multi_skill_examples(llm, skills, multi_skill_target)
        if multi_skill_dataset:
            print(f"[OK] generated {len(multi_skill_dataset)} multi-skill samples")
    except Exception as exc:
        errors.append(
            {
                "skill_id": "multi_skill_generation",
                "source_path": str(args.skills_hub),
                "error": str(exc),
            }
        )
        print(f"[ERROR] multi-skill generation: {exc}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.missing_output.parent.mkdir(parents=True, exist_ok=True)

    output_payload = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "skills_processed": len(skills),
        "samples_per_skill": args.samples_per_skill,
        "multi_skill_samples": len(multi_skill_dataset),
        "total_samples": len(dataset),
        "dataset": dataset,
        "multi_skill_dataset": multi_skill_dataset,
        "errors": errors,
    }
    args.output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.missing_output.write_text(json.dumps(missing_skill_md, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDataset saved to: {args.output}")
    print(f"Missing SKILL.md records saved to: {args.missing_output}")
    print(f"Total samples: {len(dataset)}")
    print(f"Folders without SKILL.md: {len(missing_skill_md)}")
    print(f"Generation errors: {len(errors)}")


if __name__ == "__main__":
    main()
