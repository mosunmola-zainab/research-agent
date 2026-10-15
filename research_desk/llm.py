import json

from brain.openrouter import NEMOTRON_FREE_MODEL, parse_response, send_request
from research_desk import ResearchDeskError

TIMEOUT_SECONDS = 120


def call_model(api_key: str, prompt: str) -> dict:
    if not NEMOTRON_FREE_MODEL.endswith(":free"):
        raise ResearchDeskError(f"Refusing to call {NEMOTRON_FREE_MODEL}: only :free models are allowed")

    # Like the evaluation runner, no sampling or max_tokens settings are sent, so the model runs on its defaults.
    request_body = {
        "model": NEMOTRON_FREE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    result = send_request(api_key, request_body, TIMEOUT_SECONDS)
    if result["transport_error"] is not None:
        raise ResearchDeskError(f"OpenRouter request failed: {result['transport_error']}")

    parsed = parse_response(result["raw_response"])
    if result["http_status"] != 200 or parsed["error"] is not None:
        raise ResearchDeskError(f"OpenRouter returned HTTP {result['http_status']}: {parsed['error']}")
    if parsed["returned_model"] != NEMOTRON_FREE_MODEL:
        raise ResearchDeskError(
            f"Requested {NEMOTRON_FREE_MODEL} but OpenRouter answered with {parsed['returned_model']}"
        )

    cost = (parsed["usage"] or {}).get("cost")
    if cost is not None and cost > 0:
        raise ResearchDeskError(f"OpenRouter reported a cost of {cost} for a call that must be free")

    if parsed["finish_reason"] != "stop" or not parsed["answer"]:
        raise ResearchDeskError(f"Incomplete model response: finish_reason={parsed['finish_reason']!r}")

    return {
        "content": parsed["answer"],
        "usage": parsed["usage"],
        "latency_seconds": result["latency_seconds"],
    }


def parse_json_object(text: str) -> dict:
    # Models often wrap JSON in prose or ``` fences, so take the outermost {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        raise ResearchDeskError("Model response did not contain a JSON object")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ResearchDeskError(f"Model response was not valid JSON: {exc}") from exc
