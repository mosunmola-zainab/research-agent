import json
import time
from datetime import datetime, timezone

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# The only model the research desk may use; the :free suffix keeps it on OpenRouter's no-cost endpoint.
NEMOTRON_FREE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


def send_request(api_key: str, request_body: dict, timeout_seconds: int) -> dict:
    sent_at = datetime.now(timezone.utc)
    http_status = None
    response_headers = None
    raw_response = None
    transport_error = None

    start = time.perf_counter()
    try:
        response = httpx.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=request_body,
            timeout=timeout_seconds,
        )
        http_status = response.status_code
        response_headers = dict(response.headers)
        raw_response = response.text
    except httpx.HTTPError as exc:
        transport_error = f"{type(exc).__name__}: {exc}"
    latency_seconds = time.perf_counter() - start

    return {
        "sent_at": sent_at,
        "latency_seconds": latency_seconds,
        "http_status": http_status,
        "response_headers": response_headers,
        "raw_response": raw_response,
        "transport_error": transport_error,
    }


def parse_response(raw_response: str | None) -> dict:
    returned_model = None
    finish_reason = None
    native_finish_reason = None
    usage = None
    visible_tokens = None
    answer = None
    provider = None
    system_fingerprint = None
    service_tier = None
    error = None
    error_code = None

    if raw_response is not None:
        try:
            response_data = json.loads(raw_response)
        except json.JSONDecodeError:
            response_data = None
            error = "Response body was not valid JSON"

        if response_data is not None and not isinstance(response_data, dict):
            response_data = None
            error = "Response body was not a JSON object"

        if response_data is not None:
            # A provider failure after OpenRouter has sent 200 arrives as an error body with no choices.
            if "error" in response_data:
                error = str(response_data["error"])
                if isinstance(response_data["error"], dict):
                    error_code = response_data["error"].get("code")

            returned_model = response_data.get("model")
            usage = response_data.get("usage")
            provider = response_data.get("provider")
            system_fingerprint = response_data.get("system_fingerprint")
            service_tier = response_data.get("service_tier")

            choices = response_data.get("choices")
            if choices:
                finish_reason = choices[0].get("finish_reason")
                native_finish_reason = choices[0].get("native_finish_reason")
                answer = choices[0].get("message", {}).get("content")

            if usage and usage.get("completion_tokens_details"):
                completion_tokens = usage.get("completion_tokens")
                reasoning_tokens = usage["completion_tokens_details"].get("reasoning_tokens")
                if completion_tokens is not None and reasoning_tokens is not None:
                    visible_tokens = completion_tokens - reasoning_tokens

    return {
        "returned_model": returned_model,
        "finish_reason": finish_reason,
        "native_finish_reason": native_finish_reason,
        "usage": usage,
        "visible_tokens": visible_tokens,
        "answer": answer,
        "provider": provider,
        "system_fingerprint": system_fingerprint,
        "service_tier": service_tier,
        "error": error,
        "error_code": error_code,
    }
