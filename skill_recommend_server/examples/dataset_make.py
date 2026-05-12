import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
import zipfile
import yaml
import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(root_path)
if root_path not in sys.path:
    sys.path.append(root_path)

from skills_recommender.config import settings
from skills_recommender.llm import LLMFactory, Message, ChatResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKILLS_HUB = PROJECT_ROOT / "skills-hub"
DEFAULT_OUTPUT = PROJECT_ROOT / "examples" / "keywords_dataset.json"
DEFAULT_MISSING = PROJECT_ROOT / "examples" / "missing_skill_md.json"


class SkillGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        raw_response: str = "",
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.raw_response = raw_response
        self.original_error = original_error
        self.retry_history: List[Dict[str, str]] = []


RETRYABLE_GENERATION_STAGES = {
    "llm_empty_response",
    "json_parse",
    "json_schema",
}

def custom_generate(query):
    from aigc import UniAIGC

    llm = UniAIGC()
    response = llm.client_qwen3_32b(query)
    return response

class CustomLLM:
    def chat(self, messages):
        prompt = "\n\n".join(message.content for message in messages if message.content)
        response_text = custom_generate(prompt)
        return ChatResponse(content=response_text, model="custom")

def build_llm(provider: Optional[str] = None):
    provider_name = provider or settings.llm.default
    if provider_name == "custom":
        return CustomLLM()

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


def find_all_skill_md_in_zip(zip_path: Path) -> List[str]:
    with zipfile.ZipFile(zip_path) as archive:
        return [name for name in archive.namelist()
                if name.endswith("SKILL.md") and not name.endswith("/SKILL.md/")
                ]


def read_skill_md_from_zip(zip_path: Path, member_name: str) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as file:
            return file.read().decode("utf-8")


def parse_skill_markdown(skill_md_path: Path) -> Optional[Dict[str, Any]]:
    content = skill_md_path.read_text(encoding="utf-8")
    return parse_skill_markdown_content(
        content=content,
        folder_name=skill_md_path.parent.name,
        source_path=str(skill_md_path),
    )


def parse_skill_markdown_content(content, folder_name, source_path) -> Optional[Dict[str, Any]]:
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
        "source_path": source_path,
    }


def scan_skills_hub(skills_hub_path: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    valid_skills: List[Dict[str, Any]] = []
    missing_skill_md: List[Dict[str, str]] = []

    for item in sorted(skills_hub_path.iterdir(), key=lambda path: path.name.lower()):
        if item.is_dir():
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
            continue

        if item.is_file() and item.suffix.lower() == ".zip":
            skill_md_members = find_all_skill_md_in_zip(item)
            if not skill_md_members:
                missing_skill_md.append( {
                        "folder_name": item.stem,
                        "path": str(item),
                        "reason": "SKILL.md not found",
                    })
                continue

            for member_name in skill_md_members:
                folder_name = Path(member_name).parent.name or item.stem
                content = read_skill_md_from_zip(item, member_name)
                parsed = parse_skill_markdown_content(content, folder_name, f"{item}!/{member_name}")
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
3. the user only uses very few keywords, such as "PPT制作", "用户声音分析", "视频编辑", "代码审核".
4、query 必须不多于6个汉字或4个英文单词
4. Keep a small minority of samples in English or mixed Chinese-English when natural.
5. Simulate a small amount of real-world noise, such as occasional typos, omitted punctuation, colloquial wording, or slightly incomplete context.
6. The correct label for every sample is the skill_name.
7. Return skill description in chinese.
8. Return JSON only.


JSON schema:
[
  {{
    "query": "user request",
    "label": "{skill["name"]}",
    "reason": "short explanation",
    "sample_type": "single_skill"
    "description (chinese)": "翻译成中文的技能描述"
  }}
]

skill_id: {skill["skill_id"]}
name: {skill["name"]}
description: {skill["description"]}
source_path: {skill["source_path"]}
skill_body:
{skill["body"]}
""".strip()

    try:
        response = llm.chat([Message(role="user", content=prompt)])
    except Exception as exc:
        raise SkillGenerationError(
            f"LLM request failed: {exc}",
            stage="llm_request",
            original_error=exc,
        ) from exc

    raw_content = getattr(response, "content", "")
    if raw_content is None:
        raw_content = ""
    content = str(raw_content).strip()
    if not content:
        raise SkillGenerationError(
            "LLM returned empty content; JSON parsing was not attempted.",
            stage="llm_empty_response",
            raw_response=str(raw_content),
        )

    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SkillGenerationError(
            f"LLM response is not valid JSON: {exc}",
            stage="json_parse",
            raw_response=content,
            original_error=exc,
        ) from exc

    if not isinstance(data, list):
        raise SkillGenerationError(
            f"LLM JSON root must be a list, got {type(data).__name__}.",
            stage="json_schema",
            raw_response=content,
        )

    results: List[Dict[str, Any]] = []
    for item in data:
        query = str(item.get("query", "")).strip()
        description = str(item.get("description (chinese)", "")).strip()
        if not query:
            continue
        results.append(
            {
                "query": query,
                "label": skill["name"],
                "reason": str(item.get("reason", "")).strip(),
                "sample_type": str(item.get("sample_type", "single_skill")).strip() or "single_skill",
                "skill_name": skill["name"],
                "description": description,
                "source_path": skill["source_path"],
            }
        )
    return results


def generate_examples_with_retries(
    llm,
    skill: Dict[str, Any],
    samples_per_skill: int,
    llm_retries: int,
) -> List[Dict[str, Any]]:
    retry_history: List[Dict[str, str]] = []
    max_attempts = llm_retries + 1

    for attempt in range(1, max_attempts + 1):
        try:
            return generate_examples_for_skill(llm, skill, samples_per_skill)
        except SkillGenerationError as exc:
            retry_history.append(
                {
                    "attempt": str(attempt),
                    "stage": exc.stage,
                    "error": str(exc),
                    "raw_response_preview": str(exc.raw_response)[:300],
                }
            )
            if exc.stage not in RETRYABLE_GENERATION_STAGES or attempt >= max_attempts:
                exc.retry_history = retry_history
                raise
            print(
                f"[RETRY] {skill['skill_id']} attempt {attempt}/{max_attempts} "
                f"failed at {exc.stage}: {exc}. Retrying..."
            )

    raise RuntimeError("Retry loop exited unexpectedly")


def load_existing_output(output_path: Path) -> Dict[str, Any]:
    if not output_path.exists():
        return {}

    try:
        return json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read existing output file for resume: {output_path}: {exc}") from exc


def successful_source_paths(payload: Dict[str, Any]) -> set[str]:
    return {
        str(item.get("source_path", "")).strip()
        for item in payload.get("dataset", [])
        if str(item.get("source_path", "")).strip()
    }


def failed_source_paths(payload: Dict[str, Any]) -> set[str]:
    return {
        str(item.get("source_path", "")).strip()
        for item in payload.get("errors", [])
        if str(item.get("source_path", "")).strip()
    }


def save_outputs(
    output_path: Path,
    missing_output_path: Path,
    missing_skill_md: List[Dict[str, str]],
    skills_processed: int,
    samples_per_skill: int,
    dataset: List[Dict[str, Any]],
    multi_skill_dataset: List[Dict[str, Any]],
    errors: List[Dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_output_path.parent.mkdir(parents=True, exist_ok=True)

    output_payload = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "skills_processed": skills_processed,
        "samples_per_skill": samples_per_skill,
        "multi_skill_samples": len(multi_skill_dataset),
        "total_samples": len(dataset),
        "dataset": dataset,
        "multi_skill_dataset": multi_skill_dataset,
        "errors": errors,
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    missing_output_path.write_text(json.dumps(missing_skill_md, ensure_ascii=False, indent=2), encoding="utf-8")


def progress_line(index: int, total: int, skill: Dict[str, Any]) -> str:
    percent = (index / total * 100) if total else 100
    return f"[{index}/{total} {percent:5.1f}%] {skill['skill_id']} -> {skill['name']}"


def format_error_record(skill: Dict[str, Any], exc: Exception) -> Dict[str, str]:
    traceback_summary = traceback.extract_tb(exc.__traceback__)
    last_frame = traceback_summary[-1] if traceback_summary else None
    root_exc = exc
    while root_exc.__cause__ is not None:
        root_exc = root_exc.__cause__

    has_raw_response = hasattr(exc, "raw_response")
    raw_response = getattr(exc, "raw_response", "")
    raw_response_text = str(raw_response)
    record = {
        "skill_id": skill["skill_id"],
        "source_path": skill["source_path"],
        "stage": getattr(exc, "stage", "unknown"),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "root_error_type": type(root_exc).__name__,
        "root_error": str(root_exc),
        "code_location": (
            f"{last_frame.filename}:{last_frame.lineno} in {last_frame.name}"
            if last_frame else ""
        ),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }

    if has_raw_response:
        record["raw_response_length"] = str(len(raw_response_text))
        record["raw_response_empty"] = str(not bool(raw_response_text)).lower()
    if raw_response_text:
        record["raw_response_preview"] = raw_response_text[:1000]
    retry_history = getattr(exc, "retry_history", [])
    if retry_history:
        record["retry_attempts"] = str(len(retry_history))
        record["retry_history"] = json.dumps(retry_history, ensure_ascii=False)
    return record


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
        default='custom',
        help="Optional LLM provider override.",
    )
    parser.add_argument(
        "--samples-per-skill",
        type=int,
        default=2,
        help="Number of labeled examples to create per skill.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Optional limit on number of skills to process.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save partial results after every N processed skills. Use 0 to save only at the end.",
    )
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=0,
        help="Retry count for LLM output errors such as empty response or invalid JSON.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load existing output and skip skills that already have successful samples.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="When used with --resume, retry skills recorded in existing errors plus unprocessed skills.",
    )
    args = parser.parse_args()

    if not args.skills_hub.exists():
        raise SystemExit(f"skills-hub path not found: {args.skills_hub}")
    if args.save_every < 0:
        raise SystemExit("--save-every must be greater than or equal to 0")
    if args.llm_retries < 0:
        raise SystemExit("--llm-retries must be greater than or equal to 0")

    llm = build_llm(args.provider)
    skills, missing_skill_md = scan_skills_hub(args.skills_hub)
    if args.limit is not None:
        skills = skills[: args.limit]

    existing_output = load_existing_output(args.output) if args.resume else {}
    dataset: List[Dict[str, Any]] = list(existing_output.get("dataset", []))
    multi_skill_dataset: List[Dict[str, Any]] = list(existing_output.get("multi_skill_dataset", []))
    errors: List[Dict[str, str]] = list(existing_output.get("errors", []))

    successful_paths = successful_source_paths(existing_output)
    previous_failed_paths = failed_source_paths(existing_output)
    retry_paths = previous_failed_paths if args.retry_errors else set()

    if args.resume:
        before_error_count = len(errors)
        errors = [
            error for error in errors
            if str(error.get("source_path", "")).strip() not in retry_paths
        ]
        print(
            f"[RESUME] loaded {len(dataset)} samples, "
            f"{len(successful_paths)} successful skills, "
            f"{before_error_count} previous errors"
        )

    skills_to_process = []
    for skill in skills:
        source_path = skill["source_path"]
        if args.resume and source_path in successful_paths and source_path not in retry_paths:
            continue
        if args.resume and source_path in previous_failed_paths and source_path not in retry_paths:
            continue
        skills_to_process.append(skill)

    total_to_process = len(skills_to_process)
    print(f"[START] skills to process: {total_to_process} / scanned: {len(skills)}")

    for index, skill in enumerate(skills_to_process, start=1):
        try:
            print(f"{progress_line(index, total_to_process, skill)} [RUN]")
            dataset.extend(
                generate_examples_with_retries(
                    llm,
                    skill,
                    args.samples_per_skill,
                    args.llm_retries,
                )
            )
            print(f"{progress_line(index, total_to_process, skill)} [OK]")
        except Exception as exc:
            error_record = format_error_record(skill, exc)
            errors.append(error_record)
            print(
                f"{progress_line(index, total_to_process, skill)} "
                f"[ERROR] {error_record['stage']} {error_record['root_error_type']}: "
                f"{error_record['root_error']}"
            )

        if args.save_every and index % args.save_every == 0:
            save_outputs(
                args.output,
                args.missing_output,
                missing_skill_md,
                len(skills),
                args.samples_per_skill,
                dataset,
                multi_skill_dataset,
                errors,
            )
            print(f"[SAVE] checkpoint saved after {index}/{total_to_process} processed skills")

    save_outputs(
        args.output,
        args.missing_output,
        missing_skill_md,
        len(skills),
        args.samples_per_skill,
        dataset,
        multi_skill_dataset,
        errors,
    )

    print(f"\nDataset saved to: {args.output}")
    print(f"Missing SKILL.md records saved to: {args.missing_output}")
    print(f"Total samples: {len(dataset)}")
    print(f"Folders without SKILL.md: {len(missing_skill_md)}")
    print(f"Generation errors: {len(errors)}")


if __name__ == "__main__":
    main()
