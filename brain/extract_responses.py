import argparse
import json
from pathlib import Path

from brain.run_evaluation import load_attempts


def load_config(experiment_dir: Path) -> dict:
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"No config.json in {experiment_dir}; is this an experiment directory?")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_task_results(experiment_dir: Path, config: dict) -> list[dict]:
    results = []
    # Task order and question wording come from the run's own config, never from ground_truth/.
    for task in config["identity"]["tasks"]:
        attempts = load_attempts(experiment_dir / "attempts", task["task_id"])
        for attempt in attempts:
            if attempt["experiment_id"] != config["experiment_id"]:
                raise ValueError(f"{task['task_id']} attempt {attempt['attempt']} belongs to another experiment")

        successful_attempts = [attempt for attempt in attempts if attempt["status"] == "success"]
        results.append({
            "task": task,
            "attempt_count": len(attempts),
            "final_success": successful_attempts[-1] if successful_attempts else None,
            "last_attempt": attempts[-1] if attempts else None,
        })
    return results


def format_count(value: int | None) -> str:
    if value is None:
        return "not reported"
    return f"{value:,}"


def describe_tokens(attempt: dict) -> str:
    usage = attempt["usage"] or {}
    details = usage.get("completion_tokens_details") or {}
    return (
        f"prompt {format_count(usage.get('prompt_tokens'))}, "
        f"completion {format_count(usage.get('completion_tokens'))} "
        f"(reasoning {format_count(details.get('reasoning_tokens'))}, "
        f"visible {format_count(attempt['visible_tokens'])})"
    )


def describe_status(result: dict) -> str:
    if result["final_success"] is not None:
        return f"Succeeded on attempt {result['final_success']['attempt']} of {result['attempt_count']}"
    if result["last_attempt"] is None:
        return "Not attempted"
    last_attempt = result["last_attempt"]
    return (
        f"No successful answer after {result['attempt_count']} attempt(s). "
        f"Last attempt failed ({last_attempt['failure_kind']}): {last_attempt['error']}"
    )


def quote_answer(answer: str) -> str:
    # Quoting keeps the model's own headings from breaking the report's structure without changing its text.
    quoted_lines = []
    for line in answer.splitlines():
        if line:
            quoted_lines.append(f"> {line}")
        else:
            quoted_lines.append(">")
    return "\n".join(quoted_lines)


def render_report(config: dict, results: list[dict]) -> str:
    identity = config["identity"]
    succeeded = sum(1 for result in results if result["final_success"] is not None)

    lines = [
        f"# Evaluation report: {config['experiment_id']}",
        "",
        f"- Model requested: `{identity['model']}`",
        f"- Paper: `{identity['paper']['path']}` (sha256 `{identity['paper']['sha256'][:12]}`)",
        f"- Experiment created: {config['created_utc']}",
        f"- Tasks with a successful answer: {succeeded}/{len(results)}",
        "",
        "| Task | Status | Attempts | Latency (s) | Completion tokens |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        final_success = result["final_success"]
        if final_success is not None:
            usage = final_success["usage"] or {}
            lines.append(
                f"| {result['task']['task_id']} | success | {result['attempt_count']} | "
                f"{final_success['latency_seconds']} | {format_count(usage.get('completion_tokens'))} |"
            )
        else:
            status = "not attempted" if result["last_attempt"] is None else "failed"
            lines.append(f"| {result['task']['task_id']} | {status} | {result['attempt_count']} | - | - |")

    for result in results:
        task = result["task"]
        lines += ["", "---", ""]
        if task["task_type"] == "question":
            lines += [f"## {task['task_id']}", "", f"**Question:** {task['question']}", ""]
        else:
            lines += [f"## {task['task_id']}", "", "**Task:** Full-paper analysis", ""]

        lines.append(f"**Status:** {describe_status(result)}")

        final_success = result["final_success"]
        if final_success is None:
            continue

        lines += [
            "",
            f"- Returned model: `{final_success['returned_model']}`",
            f"- Latency: {final_success['latency_seconds']} s",
            f"- Tokens: {describe_tokens(final_success)}",
            f"- Finish reason: {final_success['finish_reason']}",
            "",
            "**Answer:**",
            "",
            quote_answer(final_success["answer"]),
        ]

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a Markdown report of an experiment's final answers.")
    parser.add_argument("experiment_dir", type=Path, help="An experiment folder under brain/evaluation/runs/")
    args = parser.parse_args()

    config = load_config(args.experiment_dir)
    results = load_task_results(args.experiment_dir, config)

    report_path = args.experiment_dir / "report.md"
    report_path.write_text(render_report(config, results), encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
