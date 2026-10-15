import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from research_desk import ResearchDeskError, storage
from research_desk.discovery import run_discovery

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "research_desk.db"


def print_result(result: dict) -> None:
    print(f"Run {result['run_id']}: {result['status']}")
    print(f"Searches: recent = {result['recent_query']!r}; foundational = {result['foundational_query']!r}")
    print(f"New candidates considered: {result['candidate_count']}")
    if not result["recommendations"]:
        print("No recommendations this run.")
    for number, paper in enumerate(result["recommendations"], start=1):
        print(f"\n{number}. {paper['title']} [{paper['category']}]")
        print(f"   Authors: {', '.join(paper['authors']) or 'unknown'}")
        print(f"   Published: {paper['publication_date']} | Cited by: {paper['cited_by_count']}")
        print(f"   Source: {paper['source_url']}")
        print(f"   What it is: {paper['description']}")
        print(f"   Why read it: {paper['why_worth_reading']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="research_desk", description="AI Research Desk")
    subcommands = parser.add_subparsers(dest="command", required=True)
    discover = subcommands.add_parser("discover", help="Find up to three Agentic AI papers worth reading")
    discover.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite database file")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY is not set. Add it to .env in the project root.")

    connection = storage.connect(args.db)
    try:
        result = run_discovery(connection, api_key)
    except ResearchDeskError as exc:
        sys.exit(f"Discovery failed (recorded in run history): {exc}")
    finally:
        connection.close()

    # Paper titles can contain characters the Windows console code page can't print.
    sys.stdout.reconfigure(errors="replace")
    print_result(result)


if __name__ == "__main__":
    main()
