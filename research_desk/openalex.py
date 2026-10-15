import re
from datetime import date

import httpx

from research_desk import ResearchDeskError

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
TIMEOUT_SECONDS = 30
RESULTS_PER_SEARCH = 10
SELECTED_FIELDS = (
    "id,doi,display_name,publication_date,authorships,primary_location,cited_by_count,abstract_inverted_index"
)


def clean_search_query(query: str) -> str:
    # Commas, pipes and colons are filter syntax in OpenAlex URLs, so they can't appear inside a search term.
    return re.sub(r"\s+", " ", re.sub(r"[,|:\"]", " ", query)).strip()


def reconstruct_abstract_text(inverted_index: dict | None) -> str | None:
    # OpenAlex stores abstracts as {word: [positions]} rather than plain text.
    if not inverted_index:
        return None
    positioned_words = []
    for word, positions in inverted_index.items():
        for position in positions:
            positioned_words.append((position, word))
    return " ".join(word for _, word in sorted(positioned_words))


def format_paper_for_selection(work: dict, pool: str) -> dict:
    authors = []
    for authorship in work.get("authorships") or []:
        name = (authorship.get("author") or {}).get("display_name")
        if name:
            authors.append(name)

    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    source_url = work.get("doi") or primary_location.get("landing_page_url") or work["id"]

    return {
        "openalex_id": work["id"].rsplit("/", 1)[-1],
        "doi": work.get("doi"),
        "title": work["display_name"],
        "authors": authors,
        "publication_date": work.get("publication_date"),
        "source_url": source_url,
        "venue": source.get("display_name"),
        "cited_by_count": work.get("cited_by_count"),
        "abstract": reconstruct_abstract_text(work.get("abstract_inverted_index")),
        "pool": pool,
    }


def search_openalex_papers(query: str, extra_filters: list[str], sort: str, pool: str) -> list[dict]:
    filters = [f"title_and_abstract.search:{clean_search_query(query)}", "has_abstract:true"] + extra_filters
    params = {
        "filter": ",".join(filters),
        "sort": sort,
        "per-page": RESULTS_PER_SEARCH,
        "select": SELECTED_FIELDS,
    }
    try:
        response = httpx.get(OPENALEX_WORKS_URL, params=params, timeout=TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        raise ResearchDeskError(f"OpenAlex request failed: {type(exc).__name__}: {exc}") from exc

    if response.status_code != 200:
        raise ResearchDeskError(f"OpenAlex returned HTTP {response.status_code}: {response.text[:300]}")
    try:
        works = response.json()["results"]
    except (ValueError, KeyError) as exc:
        raise ResearchDeskError("OpenAlex response did not contain a results list") from exc

    return [format_paper_for_selection(work, pool) for work in works if work.get("id") and work.get("display_name")]


def find_recent_and_foundational_papers(recent_query: str, foundational_query: str, recent_since: date) -> list[dict]:
    recent = search_openalex_papers(
        recent_query, [f"from_publication_date:{recent_since.isoformat()}"], "relevance_score:desc", "recent"
    )
    foundational = search_openalex_papers(foundational_query, [], "cited_by_count:desc", "foundational")

    # The two searches can overlap, and OpenAlex sometimes holds one paper under two IDs that share a DOI.
    candidates = []
    seen_identifiers = set()
    for candidate in recent + foundational:
        identifiers = {candidate["openalex_id"], candidate["doi"]} - {None}
        if identifiers & seen_identifiers:
            continue
        seen_identifiers |= identifiers
        candidates.append(candidate)
    return candidates
