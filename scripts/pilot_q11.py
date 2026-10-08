import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "nvidia/nemotron-3-super-120b-a12b:free"
TIMEOUT_SECONDS = 900
PAPER_PATH = Path("evaluation/parsed/paper_01.md")
OUTPUT_DIR = Path("evaluation/pilots")

# Copied by hand rather than read from the ground truth file, so expected answers can't be sent.
QUESTION = "What does the paper see as the main priorities for Agentic AI in the short, medium, and long term?"


def build_prompt(paper_markdown: str) -> str:
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
        f"Question: {QUESTION}"
    )


def parse_response(raw_response: str | None) -> dict:
    returned_model = None
    finish_reason = None
    usage = None
    visible_tokens = None
    answer = None
    error = None

    if raw_response is not None:
        try:
            response_data = json.loads(raw_response)
        except json.JSONDecodeError:
            response_data = None
            error = "Response body was not valid JSON"

        if response_data is not None:
            # A provider failure after OpenRouter has sent 200 arrives as an error body with no choices.
            if "error" in response_data:
                error = str(response_data["error"])

            returned_model = response_data.get("model")
            usage = response_data.get("usage")

            choices = response_data.get("choices")
            if choices:
                finish_reason = choices[0].get("finish_reason")
                answer = choices[0].get("message", {}).get("content")

            if usage and usage.get("completion_tokens_details"):
                completion_tokens = usage.get("completion_tokens")
                reasoning_tokens = usage["completion_tokens_details"].get("reasoning_tokens")
                if completion_tokens is not None and reasoning_tokens is not None:
                    visible_tokens = completion_tokens - reasoning_tokens

    return {
        "returned_model": returned_model,
        "finish_reason": finish_reason,
        "usage": usage,
        "visible_tokens": visible_tokens,
        "answer": answer,
        "error": error,
    }


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY is not set. Add it to .env in the project root.")

    try:
        paper_markdown = PAPER_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        sys.exit(f"Paper not found: {PAPER_PATH}")

    if not paper_markdown.strip():
        sys.exit(f"Paper is empty: {PAPER_PATH}")

    request_body = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": build_prompt(paper_markdown)
            }
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 32000,
        "stream": False,
    }

    sent_at = datetime.now(timezone.utc)
    http_status = None
    raw_response = None
    error = None

    start = time.perf_counter()
    try:
        response = httpx.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=request_body,
            timeout=TIMEOUT_SECONDS,
        )
        http_status = response.status_code
        raw_response = response.text
        if http_status != 200:
            error = f"HTTP {http_status}"
    except httpx.HTTPError as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency_seconds = time.perf_counter() - start

    parsed = parse_response(raw_response)
    if error is None:
        error = parsed["error"]

    record = {
        "purpose": "calibration pilot - not an official evaluation result",
        "timestamp_utc": sent_at.isoformat(),
        "request": {
            "url": OPENROUTER_URL,
            "timeout_seconds": TIMEOUT_SECONDS,
            "body": request_body,
        },
        "requested_model": MODEL_ID,
        "http_status": http_status,
        "latency_seconds": round(latency_seconds, 2),
        "returned_model": parsed["returned_model"],
        "finish_reason": parsed["finish_reason"],
        "usage": parsed["usage"],
        "visible_tokens": parsed["visible_tokens"],
        "error": error,
        "raw_response": raw_response,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"q11_nemotron_{sent_at:%Y%m%dT%H%M%SZ}.json"
    output_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"HTTP status: {http_status}")
    print(f"Returned model: {parsed['returned_model']}")
    print(f"Finish reason: {parsed['finish_reason']}")
    print(f"Usage: {parsed['usage']}")
    print(f"Visible tokens: {parsed['visible_tokens']}")
    print(f"Latency: {latency_seconds:.1f} s")
    print(f"Saved: {output_path}")

    if error:
        sys.exit(f"Pilot failed: {error}")

    print(f"\nAnswer:\n{parsed['answer']}")


if __name__ == "__main__":
    main()
