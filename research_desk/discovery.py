import sqlite3
from datetime import date, timedelta
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from brain.openrouter import NEMOTRON_FREE_MODEL
from research_desk import ResearchDeskError, openalex, storage
from research_desk.llm import call_model, parse_json_object

TOPIC = "Agentic AI"
MAX_RECOMMENDATIONS = 3
RECENT_WINDOW_DAYS = 365
ABSTRACT_CHARACTERS_IN_PROMPT = 1200
AUTHORS_IN_PROMPT = 5


class DiscoveryState(TypedDict, total=False):
    recent_query: str
    foundational_query: str
    candidates: list[dict]
    recommendations: list[dict]


class SearchQueries(BaseModel):
    recent_query: str = Field(min_length=2, max_length=120)
    foundational_query: str = Field(min_length=2, max_length=120)


class Selection(BaseModel):
    paper_id: str
    description: str = Field(min_length=20, max_length=600)
    why_worth_reading: str = Field(min_length=20, max_length=800)


class Selections(BaseModel):
    selections: list[Selection] = Field(max_length=MAX_RECOMMENDATIONS)


def build_query_prompt(recent_since: date) -> str:
    return (
        f"You are helping a researcher find papers worth reading about {TOPIC}.\n"
        "\n"
        "Write two keyword searches for the OpenAlex scholarly database, which matches words in paper "
        "titles and abstracts:\n"
        f"1. recent_query: finds recent {TOPIC} research (published since {recent_since.isoformat()}).\n"
        f"2. foundational_query: finds widely cited, foundational research that {TOPIC} builds on.\n"
        "\n"
        "Use 2 to 6 plain keywords per search, with no commas, quotation marks or boolean operators.\n"
        "\n"
        "Respond with only a JSON object in this form:\n"
        '{"recent_query": "...", "foundational_query": "..."}'
    )


def describe_candidate(candidate: dict) -> str:
    authors = ", ".join(candidate["authors"][:AUTHORS_IN_PROMPT]) or "unknown"
    if len(candidate["authors"]) > AUTHORS_IN_PROMPT:
        authors += f" (+{len(candidate['authors']) - AUTHORS_IN_PROMPT} more)"
    abstract = (candidate["abstract"] or "")[:ABSTRACT_CHARACTERS_IN_PROMPT]
    return (
        f"[ID {candidate['openalex_id']}] ({candidate['pool']}) {candidate['title']}\n"
        f"Authors: {authors}\n"
        f"Published: {candidate['publication_date']} | Venue: {candidate['venue'] or 'unknown'} | "
        f"Cited by: {candidate['cited_by_count']}\n"
        f"Abstract: {abstract}"
    )


def build_selection_prompt(candidates: list[dict]) -> str:
    candidate_text = "\n\n".join(describe_candidate(candidate) for candidate in candidates)
    return (
        f"You are helping a researcher choose papers worth reading about {TOPIC}.\n"
        "\n"
        "Below are candidate papers from OpenAlex. Each is marked \"recent\" (published in the last 12 months) "
        "or \"foundational\" (among the most cited matches), with the metadata OpenAlex provides.\n"
        "\n"
        f"Choose up to {MAX_RECOMMENDATIONS} papers that are genuinely worth reading. Where good candidates "
        "exist, include both recent and foundational research. If none is worth reading, return an empty list.\n"
        "\n"
        "For each paper you choose, write:\n"
        "- description: one or two sentences on what the paper does, based only on its title and abstract.\n"
        "- why_worth_reading: a specific reason it is worth a researcher's time, referring to its actual "
        "contribution.\n"
        "\n"
        "Rules:\n"
        "- Only choose papers from the list, using their exact ID.\n"
        "- Do not call a paper trending, popular or influential beyond what its citation count and "
        "publication date show.\n"
        "- Do not add details that are not in the title or abstract.\n"
        "\n"
        "Respond with only a JSON object in this form:\n"
        '{"selections": [{"paper_id": "W...", "description": "...", "why_worth_reading": "..."}]}\n'
        "\n"
        f"<candidates>\n{candidate_text}\n</candidates>"
    )


def validate_model_output(model_class: type[BaseModel], content: str) -> BaseModel:
    try:
        return model_class.model_validate(parse_json_object(content))
    except ValidationError as exc:
        raise ResearchDeskError(f"Model output did not match the expected format: {exc}") from exc


def build_discovery_graph(connection: sqlite3.Connection, api_key: str, today: date):
    recent_since = today - timedelta(days=RECENT_WINDOW_DAYS)

    def formulate_query(state: DiscoveryState) -> dict:
        response = call_model(api_key, build_query_prompt(recent_since))
        queries = validate_model_output(SearchQueries, response["content"])
        recent_query = openalex.clean_search_query(queries.recent_query)
        foundational_query = openalex.clean_search_query(queries.foundational_query)
        if not recent_query or not foundational_query:
            raise ResearchDeskError("Model produced an empty search query")
        return {"recent_query": recent_query, "foundational_query": foundational_query}

    def search_papers(state: DiscoveryState) -> dict:
        candidates = openalex.find_recent_and_foundational_papers(state["recent_query"], state["foundational_query"], recent_since)
        # Leave out anything recommended in an earlier run, so each run suggests new papers.
        known_identifiers = storage.known_paper_identifiers(connection)
        new_candidates = []
        for candidate in candidates:
            identifiers = {candidate["openalex_id"], candidate["doi"]} - {None}
            if not identifiers & known_identifiers:
                new_candidates.append(candidate)
        return {"candidates": new_candidates}

    def route_after_search(state: DiscoveryState) -> str:
        if state["candidates"]:
            return "select_papers"
        return END

    def select_papers(state: DiscoveryState) -> dict:
        response = call_model(api_key, build_selection_prompt(state["candidates"]))
        selections = validate_model_output(Selections, response["content"]).selections

        candidates_by_id = {candidate["openalex_id"]: candidate for candidate in state["candidates"]}
        chosen_ids = [selection.paper_id for selection in selections]
        unknown_ids = [paper_id for paper_id in chosen_ids if paper_id not in candidates_by_id]
        if unknown_ids:
            raise ResearchDeskError(f"Model chose papers that were not among the candidates: {unknown_ids}")
        if len(set(chosen_ids)) != len(chosen_ids):
            raise ResearchDeskError("Model chose the same paper more than once")

        recommendations = []
        for selection in selections:
            # Bibliographic details always come from OpenAlex; only the two explanations come from the model.
            candidate = candidates_by_id[selection.paper_id]
            recommendations.append({
                "openalex_id": candidate["openalex_id"],
                "doi": candidate["doi"],
                "title": candidate["title"],
                "authors": candidate["authors"],
                "publication_date": candidate["publication_date"],
                "source_url": candidate["source_url"],
                "venue": candidate["venue"],
                "cited_by_count": candidate["cited_by_count"],
                "category": candidate["pool"],
                "description": selection.description,
                "why_worth_reading": selection.why_worth_reading,
            })
        return {"recommendations": recommendations}

    graph = StateGraph(DiscoveryState)
    graph.add_node("formulate_query", formulate_query)
    graph.add_node("search_papers", search_papers)
    graph.add_node("select_papers", select_papers)
    graph.add_edge(START, "formulate_query")
    graph.add_edge("formulate_query", "search_papers")
    graph.add_conditional_edges("search_papers", route_after_search, ["select_papers", END])
    graph.add_edge("select_papers", END)
    return graph.compile()


def run_discovery(connection: sqlite3.Connection, api_key: str, today: date | None = None) -> dict:
    today = today or date.today()
    run_id = storage.start_run(connection, "discovery", NEMOTRON_FREE_MODEL)
    graph = build_discovery_graph(connection, api_key, today)

    try:
        final_state = graph.invoke({})
    except ResearchDeskError as exc:
        storage.finish_run(connection, run_id, "failed", error=str(exc))
        raise

    recommendations = final_state.get("recommendations")
    if recommendations is None:
        status = "no_candidates"
        recommendations = []
    else:
        status = "completed"
        storage.save_recommendations(connection, run_id, recommendations)

    storage.finish_run(
        connection,
        run_id,
        status,
        recent_query=final_state["recent_query"],
        foundational_query=final_state["foundational_query"],
        candidate_count=len(final_state["candidates"]),
    )
    return {
        "run_id": run_id,
        "status": status,
        "recent_query": final_state["recent_query"],
        "foundational_query": final_state["foundational_query"],
        "candidate_count": len(final_state["candidates"]),
        "recommendations": recommendations,
    }
