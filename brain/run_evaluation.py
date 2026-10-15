import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

from brain.openrouter import OPENROUTER_URL, parse_response, send_request

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
PAPER_PATH = Path("evaluation/parsed/paper_01.md")
QUESTIONS_PATH = Path("evaluation/ground_truth/paper_01.md")
RUNS_DIR = Path("evaluation/runs")

EXPECTED_QUESTION_COUNT = 12
TIMEOUT_SECONDS = {"question": 120, "analysis": 300}
# The private analysis criteria in ground_truth/ are for grading only and must never be sent.
ANALYSIS_INSTRUCTION = (
    "Read the attached research paper carefully and provide a comprehensive analysis of it. "
    "Ground your analysis in the paper, distinguish the authors' claims from your own interpretations, "
    "and explicitly acknowledge information that is missing or cannot be verified from the provided material. "
    "Keep the analysis to a maximum of 1,500 words."
)
# Sampling, reasoning and max_tokens are left out so each model runs on its native defaults.
REQUEST_SETTINGS = {"stream": False}
MAX_CONSECUTIVE_TRANSIENT_FAILURES = 3
DEFAULT_RETRY_WAIT_SECONDS = 30
# A longer Retry-After usually means a daily limit, so stop and resume later instead of waiting.
MAX_RETRY_WAIT_SECONDS = 600


def read_required_text(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"File is empty: {path}")
    return text


def load_questions(questions_path: Path) -> list[str]:
    text = read_required_text(questions_path)
    questions = []
    # Only each block's **Question:** line is read, so expected answers never leave the file.
    for block in re.split(r"^### ", text, flags=re.MULTILINE)[1:]:
        number = int(block.split(".", 1)[0])
        if number != len(questions) + 1:
            raise ValueError(f"Question {number} is out of order in {questions_path}")

        question_lines = [line for line in block.splitlines() if line.startswith("**Question:**")]
        if len(question_lines) != 1:
            raise ValueError(f"Question {number} needs exactly one **Question:** line in {questions_path}")

        question = question_lines[0].removeprefix("**Question:**").strip()
        if not question:
            raise ValueError(f"Question {number} is empty in {questions_path}")
        questions.append(question)

    if len(questions) != EXPECTED_QUESTION_COUNT:
        raise ValueError(f"Expected {EXPECTED_QUESTION_COUNT} questions in {questions_path}, found {len(questions)}")
    return questions


def build_question_prompt(paper_markdown: str, question: str) -> str:
    return (
        "Below is the full text of a research paper, extracted from the PDF as Markdown.\n"
        "\n"
        "<paper>\n"
        f"{paper_markdown}\n"
        "</paper>\n"
        "\n"
        "Answer the following question about this paper.\n"
        "\n"
        "Base your answer on the provided paper. If the paper does not provide enough information, "
        "say so rather than guessing. Clearly label any interpretation that goes beyond what the paper states.\n"
        "\n"
        f"Question: {question}"
    )


def build_analysis_prompt(paper_markdown: str) -> str:
    return (
        "Below is the full text of a research paper, extracted from the PDF as Markdown.\n"
        "\n"
        "<paper>\n"
        f"{paper_markdown}\n"
        "</paper>\n"
        "\n"
        f"{ANALYSIS_INSTRUCTION}"
    )


def build_tasks(paper_markdown: str, questions: list[str]) -> list[dict]:
    tasks = []
    for number, question in enumerate(questions, start=1):
        tasks.append({
            "task_id": f"q{number:02d}",
            "task_type": "question",
            "question": question,
            "prompt": build_question_prompt(paper_markdown, question),
        })
    tasks.append({
        "task_id": "analysis",
        "task_type": "analysis",
        "question": None,
        "prompt": build_analysis_prompt(paper_markdown),
    })
    return tasks


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_identity(model_id: str, paper_path: Path, paper_markdown: str, tasks: list[dict]) -> dict:
    # Everything that changes what the model receives; any difference means a different experiment.
    return {
        "format_version": 1,
        "model": model_id,
        "paper": {"path": paper_path.as_posix(), "sha256": sha256_text(paper_markdown)},
        "request_settings": REQUEST_SETTINGS,
        "tasks": [
            {
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "question": task["question"],
                "prompt_sha256": sha256_text(task["prompt"]),
            }
            for task in tasks
        ],
    }


def make_experiment_id(identity: dict, paper_path: Path) -> str:
    digest = sha256_text(json.dumps(identity, sort_keys=True))[:12]
    model_slug = re.sub(r"[^A-Za-z0-9.-]", "_", identity["model"])
    return f"{paper_path.stem}__{model_slug}__{digest}"


def fetch_listed_model_info(model_id: str) -> dict:
    try:
        response = httpx.get(OPENROUTER_MODELS_URL, timeout=60)
        response.raise_for_status()
        models = response.json()["data"]
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}

    for model in models:
        if model.get("id") == model_id:
            return {
                "note": "Listed by OpenRouter; not confirmed as the settings applied to our requests.",
                "fetched_utc": datetime.now(timezone.utc).isoformat(),
                "default_parameters": model.get("default_parameters"),
                "reasoning": model.get("reasoning"),
                "top_provider": model.get("top_provider"),
                "supported_parameters": model.get("supported_parameters"),
            }
    return {"unavailable": f"{model_id} is not in OpenRouter's model list"}


def write_json(path: Path, data: dict) -> None:
    # Write then rename, so an interrupted run can't leave a half-written file behind.
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary_path, path)


def prepare_experiment(runs_dir: Path, experiment_id: str, identity: dict) -> Path:
    experiment_dir = runs_dir / experiment_id
    config_path = experiment_dir / "config.json"

    if config_path.exists():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if saved_config["identity"] != identity:
            raise ValueError(f"{config_path} does not match the current configuration; refusing to mix results")
        return experiment_dir

    (experiment_dir / "attempts").mkdir(parents=True)
    write_json(config_path, {
        "experiment_id": experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "timeout_seconds": TIMEOUT_SECONDS,
        "listed_model_defaults_unverified": fetch_listed_model_info(identity["model"]),
    })
    return experiment_dir


def load_attempts(attempts_dir: Path, task_id: str) -> list[dict]:
    attempts = []
    for path in attempts_dir.glob(f"{task_id}__attempt_*.json"):
        attempts.append(json.loads(path.read_text(encoding="utf-8")))
    return sorted(attempts, key=lambda attempt: attempt["attempt"])


def is_fatal_status(status_code: int, response_headers: dict | None) -> bool:
    if status_code in (408, 429):
        return False
    # OpenRouter documents a 402 with Retry-After as a temporary budget limit, not missing credits.
    if status_code == 402 and response_headers and "retry-after" in response_headers:
        return False
    return 400 <= status_code < 500


def classify_failure(request_result: dict, parsed: dict) -> str | None:
    if request_result["transport_error"] is not None:
        return "transient"

    status_code = request_result["http_status"]
    if status_code == 200 and parsed["error"] is not None:
        if parsed["error_code"] is None:
            return "transient"
        status_code = parsed["error_code"]

    if status_code != 200:
        if is_fatal_status(status_code, request_result["response_headers"]):
            return "fatal"
        return "transient"

    if parsed["finish_reason"] != "stop" or not parsed["answer"]:
        return "incomplete"
    return None


def describe_error(request_result: dict, parsed: dict, failure_kind: str | None) -> str | None:
    if failure_kind is None:
        return None
    if request_result["transport_error"] is not None:
        return request_result["transport_error"]
    if parsed["error"] is not None:
        return f"HTTP {request_result['http_status']}: {parsed['error']}"
    if request_result["http_status"] != 200:
        return f"HTTP {request_result['http_status']}"
    return f"Incomplete response: finish_reason={parsed['finish_reason']!r}, answer empty={not parsed['answer']}"


def run_attempt(api_key: str, experiment_id: str, model_id: str, task: dict, attempt_number: int) -> dict:
    request_body = {
        "model": model_id,
        "messages": [{"role": "user", "content": task["prompt"]}],
    }
    request_body.update(REQUEST_SETTINGS)
    timeout_seconds = TIMEOUT_SECONDS[task["task_type"]]

    request_result = send_request(api_key, request_body, timeout_seconds)
    parsed = parse_response(request_result["raw_response"])
    failure_kind = classify_failure(request_result, parsed)

    return {
        "experiment_id": experiment_id,
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "attempt": attempt_number,
        "status": "success" if failure_kind is None else "failed",
        "failure_kind": failure_kind,
        "error": describe_error(request_result, parsed, failure_kind),
        "timestamp_utc": request_result["sent_at"].isoformat(),
        "latency_seconds": round(request_result["latency_seconds"], 2),
        "request": {
            "url": OPENROUTER_URL,
            "timeout_seconds": timeout_seconds,
            "body": request_body,
        },
        "requested_model": model_id,
        "returned_model": parsed["returned_model"],
        "http_status": request_result["http_status"],
        "finish_reason": parsed["finish_reason"],
        "usage": parsed["usage"],
        "visible_tokens": parsed["visible_tokens"],
        "reported_by_response": {
            "provider": parsed["provider"],
            "system_fingerprint": parsed["system_fingerprint"],
            "service_tier": parsed["service_tier"],
            "native_finish_reason": parsed["native_finish_reason"],
        },
        "answer": parsed["answer"],
        "response_headers": request_result["response_headers"],
        "raw_response": request_result["raw_response"],
    }


def retry_wait_seconds(response_headers: dict | None) -> float:
    if response_headers and "retry-after" in response_headers:
        try:
            return float(response_headers["retry-after"])
        except ValueError:
            # Retry-After can also be an HTTP date; fall back to the default rather than parse dates.
            pass
    return DEFAULT_RETRY_WAIT_SECONDS


def summarize(experiment_dir: Path, tasks: list[dict], stopped_reason: str | None) -> dict:
    unfinished = []
    for task in tasks:
        attempts = load_attempts(experiment_dir / "attempts", task["task_id"])
        if not any(attempt["status"] == "success" for attempt in attempts):
            unfinished.append(task["task_id"])
    return {
        "experiment_dir": experiment_dir,
        "succeeded": len(tasks) - len(unfinished),
        "total": len(tasks),
        "unfinished": unfinished,
        "stopped_reason": stopped_reason,
    }


def run_experiment(
    api_key: str,
    model_id: str,
    paper_path: Path,
    questions_path: Path,
    runs_dir: Path,
) -> dict:
    paper_markdown = read_required_text(paper_path)
    questions = load_questions(questions_path)

    tasks = build_tasks(paper_markdown, questions)
    identity = build_identity(model_id, paper_path, paper_markdown, tasks)
    experiment_id = make_experiment_id(identity, paper_path)
    experiment_dir = prepare_experiment(runs_dir, experiment_id, identity)
    attempts_dir = experiment_dir / "attempts"
    print(f"Experiment: {experiment_id}")

    consecutive_transient_failures = 0
    for task in tasks:
        task_id = task["task_id"]
        previous_attempts = load_attempts(attempts_dir, task_id)
        if any(attempt["status"] == "success" for attempt in previous_attempts):
            print(f"{task_id}: already succeeded, skipping")
            continue

        attempt_number = max((attempt["attempt"] for attempt in previous_attempts), default=0) + 1
        while True:
            record = run_attempt(api_key, experiment_id, model_id, task, attempt_number)
            write_json(attempts_dir / f"{task_id}__attempt_{attempt_number:03d}.json", record)

            if record["status"] == "success":
                print(f"{task_id} attempt {attempt_number}: success ({record['latency_seconds']} s)")
                consecutive_transient_failures = 0
                break

            print(f"{task_id} attempt {attempt_number}: failed ({record['failure_kind']}) - {record['error']}")
            if record["failure_kind"] == "fatal":
                return summarize(experiment_dir, tasks, f"Fatal error on {task_id}: {record['error']}")
            if record["failure_kind"] == "incomplete":
                break

            consecutive_transient_failures += 1
            if consecutive_transient_failures >= MAX_CONSECUTIVE_TRANSIENT_FAILURES:
                return summarize(experiment_dir, tasks, f"{consecutive_transient_failures} consecutive transient failures")

            wait_seconds = retry_wait_seconds(record["response_headers"])
            if wait_seconds > MAX_RETRY_WAIT_SECONDS:
                return summarize(experiment_dir, tasks, f"Provider asked to wait {wait_seconds:.0f} s; resume later")
            print(f"Waiting {wait_seconds:.0f} s before retrying {task_id}")
            time.sleep(wait_seconds)
            attempt_number += 1

    return summarize(experiment_dir, tasks, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Paper 1 evaluation for one OpenRouter model.")
    parser.add_argument("model", help="OpenRouter model ID, e.g. nvidia/nemotron-3-super-120b-a12b:free")
    args = parser.parse_args()

    # Data paths are relative to brain/ and saved in experiment identities, so the launch folder must not matter.
    os.chdir(Path(__file__).resolve().parent)

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY is not set. Add it to .env in the project root.")

    summary = run_experiment(
        api_key,
        args.model,
        PAPER_PATH,
        QUESTIONS_PATH,
        RUNS_DIR,
    )

    print(f"\nSucceeded: {summary['succeeded']}/{summary['total']}")
    print(f"Results: {summary['experiment_dir']}")
    if summary["unfinished"]:
        print(f"Unfinished: {', '.join(summary['unfinished'])}")
    if summary["stopped_reason"]:
        print(f"Stopped: {summary['stopped_reason']}")
    if summary["unfinished"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
