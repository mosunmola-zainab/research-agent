import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The brain folder isn't a package, so make it importable for the tests.
sys.path.insert(0, str(PROJECT_ROOT / "brain"))

import run_evaluation

API_KEY = "sk-test-key-must-not-be-saved"
MODEL = "test/model:free"
REAL_PAPER = PROJECT_ROOT / "brain/evaluation/parsed/paper_01.md"
REAL_QUESTIONS = PROJECT_ROOT / "brain/evaluation/ground_truth/paper_01.md"
PRIVATE_ANALYSIS_CRITERIA = PROJECT_ROOT / "brain/evaluation/ground_truth/full_paper_analysis.md"
EXACT_ANALYSIS_INSTRUCTION = (
    "Read the attached research paper carefully and provide a comprehensive analysis of it. Ground your analysis "
    "in the paper, distinguish the authors' claims from your own interpretations, and explicitly acknowledge "
    "information that is missing or cannot be verified from the provided material. "
    "Keep the analysis to a maximum of 1,500 words."
)
FAKE_REQUEST = httpx.Request("POST", run_evaluation.OPENROUTER_URL)


def success_response(answer: str = "An answer.", finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(200, request=FAKE_REQUEST, json={
        "model": MODEL,
        "provider": "TestProvider",
        "choices": [{"finish_reason": finish_reason, "native_finish_reason": finish_reason,
                     "message": {"content": answer}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30,
                  "completion_tokens_details": {"reasoning_tokens": 20}},
    })


def error_response(status_code: int, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status_code, request=FAKE_REQUEST, headers=headers,
                          json={"error": {"code": status_code, "message": "test error"}})


def fake_post(responses: list):
    """Return queued responses (or raise queued exceptions), then succeed for every later call."""
    def post(*args, **kwargs):
        if responses:
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return success_response()
    return mock.Mock(side_effect=post)


def private_criteria_lines() -> list[str]:
    text = PRIVATE_ANALYSIS_CRITERIA.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def write_fixture_files(folder: Path, paper_text: str = "Fixture paper text.") -> tuple[Path, Path]:
    paper_path = folder / "paper_01.md"
    paper_path.write_text(paper_text, encoding="utf-8")

    blocks = []
    for number in range(1, 13):
        blocks.append(
            f"### {number}. Topic {number}\n\n"
            f"**Question:** Fixture question {number}?\n\n"
            f"**Expected:** SECRET EXPECTED ANSWER {number}.\n"
        )
    questions_path = folder / "questions.md"
    questions_path.write_text("# Ground Truth\n\n" + "\n".join(blocks), encoding="utf-8")
    return paper_path, questions_path


class QuestionAndPromptTests(unittest.TestCase):
    def test_loads_twelve_questions_from_ground_truth(self):
        questions = run_evaluation.load_questions(REAL_QUESTIONS)
        self.assertEqual(len(questions), 12)
        self.assertEqual(
            questions[10],
            "What does the paper see as the main priorities for Agentic AI in the short, medium, and long term?",
        )

    @unittest.skipUnless(REAL_PAPER.exists(), "parsed paper not available")
    def test_prompts_never_contain_expected_answers(self):
        paper = REAL_PAPER.read_text(encoding="utf-8")
        tasks = run_evaluation.build_tasks(paper, run_evaluation.load_questions(REAL_QUESTIONS))
        self.assertEqual(len(tasks), 13)

        ground_truth = REAL_QUESTIONS.read_text(encoding="utf-8")
        expected_lines = []
        for block in ground_truth.split("**Expected:**")[1:]:
            expected_text = block.split("\n### ")[0]
            expected_lines += [line.strip() for line in expected_text.splitlines() if line.strip()]
        self.assertGreater(len(expected_lines), 0)

        for task in tasks:
            self.assertNotIn("**Expected:**", task["prompt"])
            for line in expected_lines:
                self.assertNotIn(line, task["prompt"], f"{task['task_id']} contains expected answer text")

    @unittest.skipUnless(REAL_PAPER.exists(), "parsed paper not available")
    def test_question_prompt_matches_approved_template(self):
        paper = REAL_PAPER.read_text(encoding="utf-8")
        question = "What does the paper see as the main priorities for Agentic AI in the short, medium, and long term?"
        template = (
            "Below is the full text of a research paper, extracted from the PDF as Markdown.\n\n"
            "<paper>\n{PAPER}\n</paper>\n\n"
            "Answer the following question about this paper.\n\n"
            "Base your answer on the provided paper. If the paper does not provide enough information, "
            "say so rather than guessing. Clearly label any interpretation that goes beyond what the paper states.\n\n"
            f"Question: {question}"
        )
        self.assertEqual(run_evaluation.build_question_prompt(paper, question), template.replace("{PAPER}", paper))

    def test_analysis_prompt_is_the_paper_plus_the_exact_instruction(self):
        prompt = run_evaluation.build_analysis_prompt("PAPER")
        self.assertEqual(
            prompt,
            "Below is the full text of a research paper, extracted from the PDF as Markdown.\n\n"
            "<paper>\nPAPER\n</paper>\n\n"
            f"{EXACT_ANALYSIS_INSTRUCTION}",
        )

    @unittest.skipUnless(REAL_PAPER.exists(), "parsed paper not available")
    def test_prompts_never_contain_private_analysis_criteria(self):
        paper = REAL_PAPER.read_text(encoding="utf-8")
        tasks = run_evaluation.build_tasks(paper, run_evaluation.load_questions(REAL_QUESTIONS))
        criteria_lines = private_criteria_lines()
        self.assertGreater(len(criteria_lines), 0)

        for task in tasks:
            for line in criteria_lines:
                self.assertNotIn(line, task["prompt"], f"{task['task_id']} contains private analysis criteria")

    def test_rejects_question_file_with_missing_question(self):
        with tempfile.TemporaryDirectory() as folder:
            _, questions_path = write_fixture_files(Path(folder))
            text = questions_path.read_text(encoding="utf-8")
            questions_path.write_text(text.split("### 12.")[0], encoding="utf-8")
            with self.assertRaises(ValueError):
                run_evaluation.load_questions(questions_path)


class RunExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary_folder = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_folder.cleanup)
        self.folder = Path(temporary_folder.name)
        self.paper_path, self.questions_path = write_fixture_files(self.folder)
        self.runs_dir = self.folder / "runs"

        listed_models = httpx.Response(200, request=FAKE_REQUEST, json={"data": [
            {"id": MODEL, "default_parameters": {"temperature": 1}, "reasoning": {"default_enabled": True}}
        ]})
        get_patcher = mock.patch.object(run_evaluation.httpx, "get", return_value=listed_models)
        self.mock_get = get_patcher.start()
        self.addCleanup(get_patcher.stop)

        sleep_patcher = mock.patch.object(run_evaluation.time, "sleep")
        self.mock_sleep = sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    def run_with(self, responses: list, model: str = MODEL, paper_path: Path | None = None) -> tuple[dict, mock.Mock]:
        post = fake_post(responses)
        with mock.patch.object(run_evaluation.httpx, "post", post), contextlib.redirect_stdout(io.StringIO()):
            summary = run_evaluation.run_experiment(
                API_KEY, model, paper_path or self.paper_path, self.questions_path, self.runs_dir
            )
        return summary, post

    def attempts(self, summary: dict, task_id: str) -> list[dict]:
        return run_evaluation.load_attempts(summary["experiment_dir"] / "attempts", task_id)

    def test_all_tasks_succeed_and_are_saved(self):
        summary, post = self.run_with([])

        self.assertEqual(summary["succeeded"], 13)
        self.assertEqual(summary["unfinished"], [])
        self.assertIsNone(summary["stopped_reason"])
        self.assertEqual(post.call_count, 13)

        record = self.attempts(summary, "q01")[0]
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["requested_model"], MODEL)
        self.assertEqual(record["returned_model"], MODEL)
        self.assertEqual(record["http_status"], 200)
        self.assertEqual(record["finish_reason"], "stop")
        self.assertEqual(record["usage"]["completion_tokens_details"]["reasoning_tokens"], 20)
        self.assertEqual(record["visible_tokens"], 10)
        self.assertEqual(record["reported_by_response"]["provider"], "TestProvider")
        self.assertIsNotNone(record["timestamp_utc"])
        self.assertIsNotNone(record["raw_response"])

    def test_request_body_leaves_inference_settings_to_the_model(self):
        _, post = self.run_with([])
        body = post.call_args_list[0].kwargs["json"]
        self.assertEqual(set(body), {"model", "messages", "stream"})
        self.assertFalse(body["stream"])

    def test_credentials_are_never_saved(self):
        summary, _ = self.run_with([])
        for path in summary["experiment_dir"].rglob("*.json"):
            self.assertNotIn(API_KEY, path.read_text(encoding="utf-8"), path.name)

    def test_expected_answers_are_never_sent(self):
        _, post = self.run_with([])
        for call in post.call_args_list:
            self.assertNotIn("SECRET EXPECTED ANSWER", json.dumps(call.kwargs["json"]))

    def test_private_analysis_criteria_file_is_never_read(self):
        original_read_text = Path.read_text
        files_read = []

        def recording_read_text(path, *args, **kwargs):
            files_read.append(Path(path).resolve())
            return original_read_text(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", recording_read_text):
            self.run_with([])
        self.assertGreater(len(files_read), 0)
        self.assertNotIn(PRIVATE_ANALYSIS_CRITERIA.resolve(), files_read)

    def test_private_analysis_criteria_never_enter_a_request(self):
        _, post = self.run_with([])
        criteria_lines = private_criteria_lines()
        for call in post.call_args_list:
            sent_prompt = call.kwargs["json"]["messages"][0]["content"]
            for line in criteria_lines:
                self.assertNotIn(line, sent_prompt)

    def test_analysis_request_sends_the_paper_and_exact_instruction(self):
        _, post = self.run_with([])
        analysis_prompt = post.call_args_list[-1].kwargs["json"]["messages"][0]["content"]
        self.assertIn("<paper>\nFixture paper text.\n</paper>", analysis_prompt)
        self.assertTrue(analysis_prompt.endswith(EXACT_ANALYSIS_INSTRUCTION))

    def test_timeouts_per_task_type(self):
        _, post = self.run_with([])
        timeouts = [call.kwargs["timeout"] for call in post.call_args_list]
        self.assertEqual(timeouts, [120] * 12 + [300])

    def test_config_records_identity_and_unverified_listed_defaults(self):
        summary, _ = self.run_with([])
        config = json.loads((summary["experiment_dir"] / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["identity"]["model"], MODEL)
        self.assertEqual(len(config["identity"]["tasks"]), 13)
        listed = config["listed_model_defaults_unverified"]
        self.assertEqual(listed["default_parameters"], {"temperature": 1})
        self.assertIn("not confirmed", listed["note"])

    def test_listed_defaults_fetch_failure_is_recorded(self):
        self.mock_get.side_effect = httpx.ConnectError("offline")
        summary, _ = self.run_with([])
        config = json.loads((summary["experiment_dir"] / "config.json").read_text(encoding="utf-8"))
        self.assertIn("ConnectError", config["listed_model_defaults_unverified"]["unavailable"])
        self.assertEqual(summary["succeeded"], 13)

    def test_authentication_error_stops_immediately(self):
        summary, post = self.run_with([error_response(401)])

        self.assertEqual(post.call_count, 1)
        self.assertIn("Fatal error on q01", summary["stopped_reason"])
        record = self.attempts(summary, "q01")[0]
        self.assertEqual(record["failure_kind"], "fatal")
        self.assertEqual(record["http_status"], 401)

    def test_invalid_request_stops_immediately(self):
        summary, post = self.run_with([error_response(400)])
        self.assertEqual(post.call_count, 1)
        self.assertIn("Fatal", summary["stopped_reason"])

    def test_transient_failure_is_retried_and_earlier_attempt_kept(self):
        summary, _ = self.run_with([error_response(502)])

        self.assertEqual(summary["succeeded"], 13)
        attempts = self.attempts(summary, "q01")
        self.assertEqual([attempt["status"] for attempt in attempts], ["failed", "success"])
        self.assertEqual(attempts[0]["failure_kind"], "transient")
        self.mock_sleep.assert_called_once_with(run_evaluation.DEFAULT_RETRY_WAIT_SECONDS)

    def test_rate_limit_waits_for_retry_after(self):
        self.run_with([error_response(429, headers={"Retry-After": "7"})])
        self.mock_sleep.assert_called_once_with(7.0)

    def test_long_retry_after_stops_instead_of_waiting(self):
        summary, post = self.run_with([error_response(429, headers={"Retry-After": "5000"})])
        self.assertEqual(post.call_count, 1)
        self.assertIn("resume later", summary["stopped_reason"])
        self.mock_sleep.assert_not_called()

    def test_three_consecutive_transient_failures_stop_the_run(self):
        summary, post = self.run_with([error_response(503), error_response(503), error_response(503)])
        self.assertEqual(post.call_count, 3)
        self.assertIn("3 consecutive transient failures", summary["stopped_reason"])
        self.assertEqual(len(self.attempts(summary, "q01")), 3)

    def test_success_resets_consecutive_failure_count(self):
        responses = [error_response(503), error_response(503), success_response(),
                     error_response(503), error_response(503)]
        summary, _ = self.run_with(responses)
        self.assertIsNone(summary["stopped_reason"])
        self.assertEqual(summary["succeeded"], 13)

    def test_timeout_is_recorded_as_transient(self):
        summary, _ = self.run_with([httpx.ReadTimeout("timed out")])
        failed = self.attempts(summary, "q01")[0]
        self.assertEqual(failed["failure_kind"], "transient")
        self.assertIn("ReadTimeout", failed["error"])
        self.assertIsNone(failed["http_status"])
        self.assertEqual(summary["succeeded"], 13)

    def test_error_body_inside_200_is_a_failure(self):
        provider_failure = httpx.Response(200, request=FAKE_REQUEST, json={"error": {"code": 502, "message": "dropped"}})
        summary, _ = self.run_with([provider_failure])
        failed = self.attempts(summary, "q01")[0]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failure_kind"], "transient")
        self.assertEqual(failed["http_status"], 200)

    def test_incomplete_response_fails_the_task_and_moves_on(self):
        summary, post = self.run_with([success_response(answer="", finish_reason="length")])

        self.assertEqual(post.call_count, 13)
        self.assertEqual(summary["unfinished"], ["q01"])
        failed = self.attempts(summary, "q01")[0]
        self.assertEqual(failed["failure_kind"], "incomplete")
        self.mock_sleep.assert_not_called()

    def test_resume_skips_successes_and_retries_failed_task(self):
        first_summary, _ = self.run_with([success_response()] * 4 + [error_response(401)])
        self.assertEqual(first_summary["unfinished"][0], "q05")

        second_summary, post = self.run_with([])

        self.assertEqual(second_summary["experiment_dir"], first_summary["experiment_dir"])
        self.assertEqual(post.call_count, 9)
        self.assertEqual(second_summary["succeeded"], 13)
        self.assertEqual(len(self.attempts(second_summary, "q01")), 1)
        q05_attempts = self.attempts(second_summary, "q05")
        self.assertEqual([attempt["attempt"] for attempt in q05_attempts], [1, 2])
        self.assertEqual([attempt["status"] for attempt in q05_attempts], ["failed", "success"])

    def test_resume_ignores_half_written_temporary_file(self):
        first_summary, _ = self.run_with([success_response()] * 4 + [error_response(401)])
        attempts_dir = first_summary["experiment_dir"] / "attempts"
        (attempts_dir / "q05__attempt_002.json.tmp").write_text('{"partial', encoding="utf-8")

        second_summary, _ = self.run_with([])
        self.assertEqual(second_summary["succeeded"], 13)

    def test_completed_experiment_makes_no_new_requests(self):
        self.run_with([])
        summary, post = self.run_with([])
        self.assertEqual(post.call_count, 0)
        self.assertEqual(summary["succeeded"], 13)

    def test_different_model_gets_a_separate_experiment(self):
        first_summary, _ = self.run_with([])
        second_summary, post = self.run_with([], model="other/model:free")

        self.assertNotEqual(first_summary["experiment_dir"], second_summary["experiment_dir"])
        self.assertEqual(post.call_count, 13)

    def test_changed_paper_gets_a_separate_experiment(self):
        first_summary, _ = self.run_with([])
        changed_folder = self.folder / "changed"
        changed_folder.mkdir()
        changed_paper, _ = write_fixture_files(changed_folder, paper_text="A different paper text.")

        second_summary, _ = self.run_with([], paper_path=changed_paper)
        self.assertNotEqual(first_summary["experiment_dir"], second_summary["experiment_dir"])

    def test_mismatched_saved_config_is_refused(self):
        summary, _ = self.run_with([])
        config_path = summary["experiment_dir"] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["identity"]["request_settings"] = {"stream": False, "temperature": 0}
        config_path.write_text(json.dumps(config), encoding="utf-8")

        with self.assertRaises(ValueError):
            self.run_with([])


if __name__ == "__main__":
    unittest.main()
