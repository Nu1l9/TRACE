"""Resilient OpenAI-compatible chat transport for remote analysis tools."""

from __future__ import annotations

import time
import json
from collections.abc import Callable
from typing import Any


def _is_retryable_transport_error(error: Exception) -> bool:
    """Retry temporary provider or network failures, never request mistakes."""
    status_code = getattr(error, "status_code", None)
    if status_code in {408, 409, 429}:
        return True
    if isinstance(status_code, int) and status_code >= 500:
        return True
    return error.__class__.__name__ in {"APIConnectionError", "APITimeoutError"}


def _request_size(request: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return visible message, structured-schema, JSON-body and message counts.

    Analysis components construct ordinary chat-completion requests locally, so
    logging the sizes at this shared transport boundary gives one consistent
    diagnostic for Planner, RuntimeFacets, reviewers and selectors.  The
    counts are diagnostic only; no request content is logged.
    """
    messages = request.get("messages") or []
    prompt_chars = sum(
        len(item.get("content") or "")
        for item in messages
        if isinstance(item, dict) and isinstance(item.get("content"), str)
    )
    response_format = request.get("response_format") or {}
    schema_chars = 0
    if isinstance(response_format, dict):
        schema = response_format.get("json_schema")
        if schema is not None:
            schema_chars = len(json.dumps(schema, ensure_ascii=False, separators=(",", ":")))
        elif response_format.get("type"):
            schema_chars = len(json.dumps(response_format, ensure_ascii=False, separators=(",", ":")))
    request_chars = len(json.dumps(request, ensure_ascii=False, separators=(",", ":"), default=str))
    return prompt_chars, schema_chars, request_chars, len(messages)


def create_chat_completion(
    client: Any,
    request: dict[str, Any],
    *,
    component: str,
    max_retries: int,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Call a chat endpoint with visible bounded backoff for transient failures."""
    attempts = max(1, max_retries + 1)
    prompt_chars, schema_chars, request_chars, message_count = _request_size(request)
    print(
        f"[llm-input] component={component} messages={message_count} "
        f"prompt_chars={prompt_chars} schema_chars={schema_chars} "
        f"request_chars={request_chars} attempts={attempts}",
        flush=True,
    )
    for attempt in range(attempts):
        try:
            return client.chat.completions.create(**request)
        except Exception as error:
            if not _is_retryable_transport_error(error) or attempt + 1 == attempts:
                raise
            wait_seconds = min(2 ** (attempt + 1), 8)
            status_code = getattr(error, "status_code", "network")
            print(
                f"[{component}] Transient service error (HTTP {status_code}); "
                f"retrying in {wait_seconds}s ({attempt + 2}/{attempts}).",
                flush=True,
            )
            sleep(wait_seconds)

    raise RuntimeError("Unreachable retry state.")
