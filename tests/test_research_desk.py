import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import httpx

from brain.openrouter import NEMOTRON_FREE_MODEL, OPENROUTER_URL
from research_desk import ResearchDeskError, discovery, llm, openalex, storage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 10, 14)
OPENROUTER_REQUEST = httpx.Request("POST", OPENROUTER_URL)
OPENALEX_REQUEST = httpx.Request("GET", openalex.OPENALEX_WORKS_URL)
QUERIES_JSON = '{"recent_query": "agentic AI, LLM agents", "foundational_query": "autonomous agents"}'


def openrouter_response(content: str, model: str = NEMOTRON_FREE_MODEL, cost: float = 0,
                        finish_reason: str = "stop", status_code: int = 200) -> httpx.Response:
    if status_code != 200:
        return httpx.Response(status_code, request=OPENROUTER_REQUEST,
                              json={"error": {"code": status_code, "message": "test error"}})
    return httpx.Response(200, request=OPENROUTER_REQUEST, json={
        "model": model,
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
        "usage": {"prompt_tokens": 500, "completion_tokens": 80, "cost": cost},
    })


def make_work(number: int, title: str, publication_date: str, cited_by_count: int, doi: str | None = None) -> dict:
    return {
        "id": f"https://openalex.org/W{number}",
        "doi": doi if doi is not None else f"https://doi.org/10.1000/test.{number}",
        "display_name": title,
        "publication_date": publication_date,
        "authorships": [{"author": {"display_name": "Ada Lovelace"}}, {"author": {"display_name": "Alan Turing"}}],
        "primary_location": {"landing_page_url": f"https://example.org/{number}",
                             "source": {"display_name": "Journal of Agents"}},
        "cited_by_count": cited_by_count,
        "abstract_inverted_index": {"Agents": [0], "plan": [1], "and": [2], "act.": [3]},
    }


RECENT_WORKS = [make_work(1, "Recent Agent Planning", "2026-05-01", 4),
                make_work(2, "Recent Tool Use", "2026-07-15", 1)]
FOUNDATIONAL_WORKS = [make_work(3, "ReAct: Reasoning and Acting", "2022-10-06", 2500),
                      make_work(4, "Generative Agents", "2023-04-07", 2000)]


def selection_json(*paper_ids: str) -> str:
    return json.dumps({"selections": [
        {"paper_id": paper_id,
         "description": f"Paper {paper_id} studies how language-model agents plan and act.",
         "why_worth_reading": f"Paper {paper_id} gives a concrete method that later agent work builds on."}
        for paper_id in paper_ids
    ]})


def fake_openalex_get(recent: list[dict], foundational: list[dict]):
    def get(url, params, timeout):
        works = recent if params["sort"] == "relevance_score:desc" else foundational
        return httpx.Response(200, request=OPENALEX_REQUEST, json={"meta": {}, "results": works})
    return mock.Mock(side_effect=get)


class OpenAlexTests(unittest.TestCase):
    def test_rebuilds_abstract_from_inverted_index(self):
        index = {"act.": [3], "Agents": [0], "and": [2], "plan": [1]}
        self.assertEqual(openalex.reconstruct_abstract_text(index), "Agents plan and act.")
        self.assertIsNone(openalex.reconstruct_abstract_text(None))

    def test_candidate_uses_openalex_metadata(self):
        candidate = openalex.format_paper_for_selection(make_work(7, "A Title", "2024-01-02", 9), "recent")
        self.assertEqual(candidate["openalex_id"], "W7")
        self.assertEqual(candidate["authors"], ["Ada Lovelace", "Alan Turing"])
        self.assertEqual(candidate["source_url"], "https://doi.org/10.1000/test.7")
        self.assertEqual(candidate["venue"], "Journal of Agents")

    def test_source_url_falls_back_to_landing_page(self):
        work = make_work(8, "No DOI", "2024-01-02", 0)
        work["doi"] = None
        self.assertEqual(openalex.format_paper_for_selection(work, "recent")["source_url"], "https://example.org/8")

    def test_queries_are_cleaned_of_filter_syntax(self):
        self.assertEqual(openalex.clean_search_query('agentic AI, "LLM": agents | tools'), "agentic AI LLM agents tools")

    def test_search_sends_cleaned_title_and_abstract_filter(self):
        get = fake_openalex_get(RECENT_WORKS, FOUNDATIONAL_WORKS)
        with mock.patch.object(openalex.httpx, "get", get):
            openalex.find_recent_and_foundational_papers("agentic AI, agents", "autonomous agents", date(2025, 10, 14))
        recent_params = get.call_args_list[0].kwargs["params"]
        self.assertIn("title_and_abstract.search:agentic AI agents", recent_params["filter"])
        self.assertIn("from_publication_date:2025-10-14", recent_params["filter"])

    def test_overlapping_searches_and_shared_dois_are_deduplicated(self):
        same_doi_other_id = make_work(99, "Duplicate record", "2022-10-06", 5, doi="https://doi.org/10.1000/test.3")
        get = fake_openalex_get(RECENT_WORKS + [FOUNDATIONAL_WORKS[0]], FOUNDATIONAL_WORKS + [same_doi_other_id])
        with mock.patch.object(openalex.httpx, "get", get):
            candidates = openalex.find_recent_and_foundational_papers("a", "b", date(2025, 10, 14))
        self.assertEqual([candidate["openalex_id"] for candidate in candidates], ["W1", "W2", "W3", "W4"])


class LlmTests(unittest.TestCase):
    def call_with(self, response):
        with mock.patch.object(httpx, "post", return_value=response) as post:
            result = llm.call_model("test-key", "prompt")
        return result, post

    def test_request_uses_free_model_and_native_settings(self):
        _, post = self.call_with(openrouter_response("hello"))
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "nvidia/nemotron-3-super-120b-a12b:free")
        self.assertEqual(set(body), {"model", "messages", "stream"})

    def test_refuses_a_non_free_model_without_calling_the_api(self):
        with mock.patch.object(llm, "NEMOTRON_FREE_MODEL", "nvidia/nemotron-3-super-120b-a12b"), \
                mock.patch.object(httpx, "post") as post:
            with self.assertRaises(ResearchDeskError):
                llm.call_model("test-key", "prompt")
        post.assert_not_called()

    def test_rejects_a_substituted_model(self):
        with self.assertRaises(ResearchDeskError):
            self.call_with(openrouter_response("hello", model="some/paid-model"))

    def test_rejects_a_reported_cost(self):
        with self.assertRaises(ResearchDeskError):
            self.call_with(openrouter_response("hello", cost=0.002))

    def test_rejects_an_incomplete_response(self):
        with self.assertRaises(ResearchDeskError):
            self.call_with(openrouter_response("partial", finish_reason="length"))

    def test_parses_json_inside_code_fences_and_prose(self):
        text = 'Here you go:\n```json\n{"recent_query": "x y"}\n```'
        self.assertEqual(llm.parse_json_object(text), {"recent_query": "x y"})

    def test_rejects_text_without_json(self):
        with self.assertRaises(ResearchDeskError):
            llm.parse_json_object("I could not decide.")


class DiscoveryWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary_folder = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_folder.cleanup)
        self.db_path = Path(temporary_folder.name) / "test.db"
        self.connection = storage.connect(self.db_path)
        self.addCleanup(self.connection.close)

    def run_discovery(self, model_responses: list, recent=RECENT_WORKS, foundational=FOUNDATIONAL_WORKS):
        post = mock.Mock(side_effect=model_responses)
        get = fake_openalex_get(recent, foundational)
        with mock.patch.object(httpx, "post", post), mock.patch.object(httpx, "get", get):
            result = discovery.run_discovery(self.connection, "test-key", today=TODAY)
        return result, post, get

    def runs(self) -> list[sqlite3.Row]:
        return self.connection.execute("SELECT * FROM runs ORDER BY id").fetchall()

    def count(self, table: str) -> int:
        return self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def test_full_run_recommends_recent_and_foundational_papers(self):
        result, post, _ = self.run_discovery([
            openrouter_response(QUERIES_JSON),
            openrouter_response(selection_json("W1", "W3", "W4")),
        ])

        self.assertEqual(result["status"], "completed")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(result["recent_query"], "agentic AI LLM agents")
        self.assertEqual(result["candidate_count"], 4)

        first, second, third = result["recommendations"]
        self.assertEqual((first["title"], first["category"]), ("Recent Agent Planning", "recent"))
        self.assertEqual((second["title"], second["category"]), ("ReAct: Reasoning and Acting", "foundational"))
        self.assertEqual(second["authors"], ["Ada Lovelace", "Alan Turing"])
        self.assertEqual(second["publication_date"], "2022-10-06")
        self.assertEqual(second["source_url"], "https://doi.org/10.1000/test.3")
        self.assertIn("plan and act", second["description"])
        self.assertIn("later agent work", second["why_worth_reading"])

        run = self.runs()[0]
        self.assertEqual((run["status"], run["model"], run["candidate_count"]), ("completed", NEMOTRON_FREE_MODEL, 4))
        self.assertEqual(self.count("papers"), 3)
        self.assertEqual(self.count("recommendations"), 3)

    def test_selection_prompt_contains_candidates_and_rules(self):
        _, post, _ = self.run_discovery([
            openrouter_response(QUERIES_JSON), openrouter_response(selection_json("W1")),
        ])
        prompt = post.call_args_list[1].kwargs["json"]["messages"][0]["content"]
        self.assertIn("[ID W3] (foundational) ReAct: Reasoning and Acting", prompt)
        self.assertIn("Abstract: Agents plan and act.", prompt)
        self.assertIn("Do not call a paper trending", prompt)

    def test_openrouter_error_records_failed_run(self):
        with self.assertRaises(ResearchDeskError):
            self.run_discovery([openrouter_response("", status_code=429)])
        run = self.runs()[0]
        self.assertEqual(run["status"], "failed")
        self.assertIn("429", run["error"])
        self.assertIsNotNone(run["finished_utc"])
        self.assertEqual(self.count("papers"), 0)

    def test_openrouter_timeout_records_failed_run(self):
        with self.assertRaises(ResearchDeskError):
            self.run_discovery([httpx.ReadTimeout("timed out")])
        self.assertIn("ReadTimeout", self.runs()[0]["error"])

    def test_openalex_error_records_failed_run(self):
        get = mock.Mock(return_value=httpx.Response(503, request=OPENALEX_REQUEST, text="unavailable"))
        post = mock.Mock(side_effect=[openrouter_response(QUERIES_JSON)])
        with mock.patch.object(httpx, "post", post), mock.patch.object(httpx, "get", get):
            with self.assertRaises(ResearchDeskError):
                discovery.run_discovery(self.connection, "test-key", today=TODAY)
        self.assertEqual(self.runs()[0]["status"], "failed")
        self.assertIn("OpenAlex returned HTTP 503", self.runs()[0]["error"])

    def test_openalex_timeout_records_failed_run(self):
        get = mock.Mock(side_effect=httpx.ConnectTimeout("no route"))
        with mock.patch.object(httpx, "post", mock.Mock(side_effect=[openrouter_response(QUERIES_JSON)])), \
                mock.patch.object(httpx, "get", get):
            with self.assertRaises(ResearchDeskError):
                discovery.run_discovery(self.connection, "test-key", today=TODAY)
        self.assertIn("ConnectTimeout", self.runs()[0]["error"])

    def test_malformed_model_output_fails_the_run(self):
        with self.assertRaises(ResearchDeskError):
            self.run_discovery([openrouter_response('{"recent_query": "only one"}')])
        self.assertEqual(self.runs()[0]["status"], "failed")

    def test_invented_paper_is_rejected_and_nothing_saved(self):
        with self.assertRaises(ResearchDeskError):
            self.run_discovery([openrouter_response(QUERIES_JSON), openrouter_response(selection_json("W1", "W999"))])
        self.assertIn("W999", self.runs()[0]["error"])
        self.assertEqual(self.count("papers"), 0)

    def test_more_than_three_selections_are_rejected(self):
        with self.assertRaises(ResearchDeskError):
            self.run_discovery([openrouter_response(QUERIES_JSON),
                                openrouter_response(selection_json("W1", "W2", "W3", "W4"))])

    def test_second_run_skips_papers_already_recommended(self):
        self.run_discovery([openrouter_response(QUERIES_JSON), openrouter_response(selection_json("W1", "W3"))])
        result, post, _ = self.run_discovery([
            openrouter_response(QUERIES_JSON), openrouter_response(selection_json("W2")),
        ])

        prompt = post.call_args_list[1].kwargs["json"]["messages"][0]["content"]
        self.assertNotIn("[ID W1]", prompt)
        self.assertNotIn("[ID W3]", prompt)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(self.count("papers"), 3)

    def test_no_new_candidates_ends_without_a_selection_call(self):
        self.run_discovery([openrouter_response(QUERIES_JSON), openrouter_response(selection_json("W1", "W2", "W3"))])
        self.run_discovery([openrouter_response(QUERIES_JSON), openrouter_response(selection_json("W4"))])

        result, post, _ = self.run_discovery([openrouter_response(QUERIES_JSON)])
        self.assertEqual(result["status"], "no_candidates")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(self.runs()[-1]["status"], "no_candidates")

    def test_paper_sharing_a_doi_with_an_earlier_recommendation_is_not_offered_again(self):
        self.run_discovery([openrouter_response(QUERIES_JSON), openrouter_response(selection_json("W3"))])
        same_doi_as_w3 = make_work(
            99, "ReAct (duplicate record)", "2022-10-06", 40, doi="https://doi.org/10.1000/test.3"
        )

        result, post, _ = self.run_discovery(
            [openrouter_response(QUERIES_JSON), openrouter_response(selection_json("W1"))],
            foundational=FOUNDATIONAL_WORKS + [same_doi_as_w3],
        )

        prompt = post.call_args_list[1].kwargs["json"]["messages"][0]["content"]
        self.assertNotIn("[ID W99]", prompt)
        self.assertEqual(result["candidate_count"], 3)

    def test_model_may_recommend_nothing(self):
        result, _, _ = self.run_discovery([openrouter_response(QUERIES_JSON),
                                           openrouter_response('{"selections": []}')])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.count("papers"), 0)


class StorageTests(unittest.TestCase):
    def setUp(self):
        temporary_folder = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_folder.cleanup)
        self.db_path = Path(temporary_folder.name) / "nested" / "test.db"

    def recommendation(self, openalex_id: str = "W1", doi: str | None = None) -> dict:
        candidate = openalex.format_paper_for_selection(make_work(int(openalex_id[1:]), "Title", "2024-01-01", 3), "recent")
        if doi is not None:
            candidate["doi"] = doi
        candidate.pop("abstract")
        candidate["category"] = candidate.pop("pool")
        candidate["description"] = "A description."
        candidate["why_worth_reading"] = "A reason."
        return candidate

    def test_saving_the_same_paper_twice_keeps_one_row(self):
        connection = storage.connect(self.db_path)
        self.addCleanup(connection.close)
        run_id = storage.start_run(connection, "discovery", NEMOTRON_FREE_MODEL)
        self.assertEqual(storage.save_recommendations(connection, run_id, [self.recommendation()]), 1)
        self.assertEqual(storage.save_recommendations(connection, run_id, [self.recommendation()]), 0)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 1)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0], 1)

    def orphaned_recommendation_count(self, connection: sqlite3.Connection) -> int:
        return connection.execute(
            """SELECT COUNT(*) FROM recommendations
               LEFT JOIN papers ON papers.openalex_id = recommendations.openalex_id
               WHERE papers.openalex_id IS NULL"""
        ).fetchone()[0]

    def test_shared_doi_in_one_batch_keeps_the_first_paper_and_skips_the_duplicate(self):
        connection = storage.connect(self.db_path)
        self.addCleanup(connection.close)
        run_id = storage.start_run(connection, "discovery", NEMOTRON_FREE_MODEL)
        shared_doi = "https://doi.org/10.1000/shared"

        saved = storage.save_recommendations(
            connection, run_id, [self.recommendation("W1", shared_doi), self.recommendation("W2", shared_doi)]
        )

        self.assertEqual(saved, 1)
        self.assertEqual([row[0] for row in connection.execute("SELECT openalex_id FROM papers")], ["W1"])
        self.assertEqual([row[0] for row in connection.execute("SELECT openalex_id FROM recommendations")], ["W1"])
        self.assertEqual(self.orphaned_recommendation_count(connection), 0)

    def test_shared_doi_with_an_earlier_run_does_not_discard_other_papers(self):
        connection = storage.connect(self.db_path)
        self.addCleanup(connection.close)
        shared_doi = "https://doi.org/10.1000/shared"
        first_run = storage.start_run(connection, "discovery", NEMOTRON_FREE_MODEL)
        storage.save_recommendations(connection, first_run, [self.recommendation("W1", shared_doi)])

        second_run = storage.start_run(connection, "discovery", NEMOTRON_FREE_MODEL)
        saved = storage.save_recommendations(
            connection, second_run, [self.recommendation("W3"), self.recommendation("W2", shared_doi)]
        )

        self.assertEqual(saved, 1)
        rows = connection.execute("SELECT run_id, openalex_id FROM recommendations ORDER BY id").fetchall()
        self.assertEqual([tuple(row) for row in rows], [(first_run, "W1"), (second_run, "W3")])
        self.assertEqual(self.orphaned_recommendation_count(connection), 0)

    def test_data_persists_after_reopening(self):
        connection = storage.connect(self.db_path)
        run_id = storage.start_run(connection, "discovery", NEMOTRON_FREE_MODEL)
        storage.save_recommendations(connection, run_id, [self.recommendation()])
        storage.finish_run(connection, run_id, "completed", "recent q", "foundational q", 4)
        connection.close()

        reopened = storage.connect(self.db_path)
        self.addCleanup(reopened.close)
        run = reopened.execute("SELECT * FROM runs").fetchone()
        paper = reopened.execute("SELECT * FROM papers").fetchone()
        self.assertEqual((run["status"], run["recent_query"], run["candidate_count"]), ("completed", "recent q", 4))
        self.assertEqual(json.loads(paper["authors"]), ["Ada Lovelace", "Alan Turing"])
        self.assertEqual(storage.known_paper_identifiers(reopened), {"W1", "https://doi.org/10.1000/test.1"})


class BoundaryTests(unittest.TestCase):
    def test_research_desk_never_touches_evaluation_material(self):
        for path in (PROJECT_ROOT / "research_desk").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for forbidden in ("run_evaluation", "ground_truth", "evaluation/", "extract_responses"):
                self.assertNotIn(forbidden, source, f"{path.name} references {forbidden}")


if __name__ == "__main__":
    unittest.main()
