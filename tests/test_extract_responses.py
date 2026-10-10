import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The brain folder isn't a package, so make it importable for the tests.
sys.path.insert(0, str(PROJECT_ROOT / "brain"))

import extract_responses

EXPERIMENT_ID = "paper_01__test_model__abc123"
QUESTIONS = {
    "q01": "What does the paper report about X (in %)?",
    "q02": "Why does Y happen?",
    "q03": "How is Z divided?",
    "q04": "Who ran the experiment?",
}
ANALYSIS_ANSWER = "## Overview\n\nThe paper reviews agents.\n\n- Point one\n- Point two"


def make_attempt(task_id: str, attempt: int, status: str, answer: str | None, failure_kind: str | None = None) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "task_id": task_id,
        "task_type": "analysis" if task_id == "analysis" else "question",
        "attempt": attempt,
        "status": status,
        "failure_kind": failure_kind,
        "error": None if status == "success" else "Incomplete response: finish_reason='length'",
        "timestamp_utc": "2026-10-10T02:00:00+00:00",
        "latency_seconds": 4.46,
        "request": {"body": {"messages": [{"role": "user", "content": "SECRET PAPER TEXT"}]}},
        "requested_model": "test/model",
        "returned_model": "test/model",
        "http_status": 200,
        "finish_reason": "stop" if status == "success" else "length",
        "usage": {"prompt_tokens": 40011, "completion_tokens": 383,
                  "completion_tokens_details": {"reasoning_tokens": 344}},
        "visible_tokens": 39,
        "reported_by_response": {"provider": "TestProvider"},
        "answer": answer,
        "response_headers": {"x-secret-header": "SECRET HEADER VALUE"},
        "raw_response": json.dumps({"choices": [{"message": {"reasoning": "SECRET REASONING"}}]}),
    }


def write_experiment(folder: Path) -> Path:
    experiment_dir = folder / EXPERIMENT_ID
    attempts_dir = experiment_dir / "attempts"
    attempts_dir.mkdir(parents=True)

    tasks = [{"task_id": task_id, "task_type": "question", "question": question, "prompt_sha256": "x"}
             for task_id, question in QUESTIONS.items()]
    tasks.append({"task_id": "analysis", "task_type": "analysis", "question": None, "prompt_sha256": "x"})
    config = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": "2026-10-10T01:59:00+00:00",
        "identity": {"model": "test/model", "paper": {"path": "evaluation/parsed/paper_01.md", "sha256": "f" * 64},
                     "tasks": tasks},
    }
    (experiment_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    attempts = [
        make_attempt("q01", 1, "success", "Answer one."),
        make_attempt("q02", 1, "failed", "PARTIAL FIRST ANSWER", failure_kind="incomplete"),
        make_attempt("q02", 2, "success", "Answer two, second try."),
        make_attempt("q03", 1, "failed", None, failure_kind="incomplete"),
        make_attempt("analysis", 1, "success", ANALYSIS_ANSWER),
    ]
    for attempt in attempts:
        path = attempts_dir / f"{attempt['task_id']}__attempt_{attempt['attempt']:03d}.json"
        path.write_text(json.dumps(attempt), encoding="utf-8")
    return experiment_dir


class ExtractResponsesTests(unittest.TestCase):
    def setUp(self):
        temporary_folder = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_folder.cleanup)
        self.experiment_dir = write_experiment(Path(temporary_folder.name))

    def build_report(self) -> str:
        config = extract_responses.load_config(self.experiment_dir)
        results = extract_responses.load_task_results(self.experiment_dir, config)
        return extract_responses.render_report(config, results)

    def test_tasks_appear_in_original_order(self):
        report = self.build_report()
        positions = [report.index(f"## {task_id}\n") for task_id in ["q01", "q02", "q03", "q04", "analysis"]]
        self.assertEqual(positions, sorted(positions))

    def test_question_wording_is_preserved_exactly(self):
        report = self.build_report()
        for question in QUESTIONS.values():
            self.assertIn(f"**Question:** {question}", report)

    def test_final_successful_attempt_is_used(self):
        report = self.build_report()
        self.assertIn("> Answer two, second try.", report)
        self.assertIn("Succeeded on attempt 2 of 2", report)
        self.assertNotIn("PARTIAL FIRST ANSWER", report)

    def test_failed_and_missing_tasks_are_reported_clearly(self):
        report = self.build_report()
        self.assertIn("No successful answer after 1 attempt(s). Last attempt failed (incomplete)", report)
        self.assertIn("**Status:** Not attempted", report)
        self.assertIn("| q03 | failed | 1 | - | - |", report)
        self.assertIn("| q04 | not attempted | 0 | - | - |", report)
        self.assertIn("Tasks with a successful answer: 3/5", report)

    def test_metadata_is_included(self):
        report = self.build_report()
        self.assertIn("Model requested: `test/model`", report)
        self.assertIn("- Latency: 4.46 s", report)
        self.assertIn("- Tokens: prompt 40,011, completion 383 (reasoning 344, visible 39)", report)
        self.assertIn("- Finish reason: stop", report)

    def test_answer_is_quoted_without_changing_its_text(self):
        report = self.build_report()
        self.assertIn("> ## Overview\n>\n> The paper reviews agents.\n>\n> - Point one\n> - Point two", report)

    def test_private_and_raw_data_are_excluded(self):
        report = self.build_report()
        for secret in ["SECRET PAPER TEXT", "SECRET REASONING", "SECRET HEADER VALUE", "x-secret-header"]:
            self.assertNotIn(secret, report)

    def test_ground_truth_files_are_never_read(self):
        original_read_text = Path.read_text
        files_read = []

        def recording_read_text(path, *args, **kwargs):
            files_read.append(Path(path).resolve())
            return original_read_text(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", recording_read_text):
            self.build_report()
        self.assertGreater(len(files_read), 0)
        self.assertFalse(any("ground_truth" in path.parts for path in files_read))

    def test_rejects_folder_without_config(self):
        with self.assertRaises(ValueError):
            extract_responses.load_config(self.experiment_dir / "attempts")

    def test_rejects_attempt_from_another_experiment(self):
        stray = make_attempt("q04", 1, "success", "Stray answer.")
        stray["experiment_id"] = "some_other_experiment"
        (self.experiment_dir / "attempts" / "q04__attempt_001.json").write_text(json.dumps(stray), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.build_report()

    def test_main_writes_report_into_experiment_folder(self):
        with mock.patch.object(sys, "argv", ["extract_responses.py", str(self.experiment_dir)]), \
                contextlib.redirect_stdout(io.StringIO()):
            extract_responses.main()
        report_path = self.experiment_dir / "report.md"
        self.assertTrue(report_path.exists())
        self.assertIn("# Evaluation report:", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
