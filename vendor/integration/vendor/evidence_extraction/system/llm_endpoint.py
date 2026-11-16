"""Shared OpenAI-compatible endpoint settings for Meta-Agent workflows."""

from __future__ import annotations

import os

import httpx
from langchain_openai import ChatOpenAI


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.4-mini"
OFFICIAL_OPENAI_BASE_URL = DEFAULT_BASE_URL


def _provider() -> str:
    value = os.getenv("META_AGENT_LLM_PROVIDER", os.getenv("TEXT_PROVIDER_NAME", "official_openai")).casefold()
    if value not in {"official_openai", "openai_compatible"}:
        raise RuntimeError("Unknown configured text provider")
    return value


def _get_api_key(provider: str) -> str:
    value = os.getenv("TEXT_PROVIDER_API_KEY") or os.getenv("PIPELINE_LLM_API_KEY")
    if not value:
        value = os.getenv({"official_openai": "OPENAI_API_KEY", "openai_compatible": "COMPATIBLE_API_KEY"}[provider])
    if not value:
        raise RuntimeError("Set the API key for the selected text provider")
    return value


def _get_base_url(provider: str) -> str:
    value = os.getenv("TEXT_PROVIDER_BASE_URL") or os.getenv("PIPELINE_LLM_BASE_URL")
    if not value:
        value = os.getenv({"official_openai":"OFFICIAL_OPENAI_BASE_URL", "openai_compatible":"META_AGENT_COMPATIBLE_BASE_URL_OVERRIDE"}[provider])
    if not value and provider == "openai_compatible": value = os.getenv("COMPATIBLE_BASE_URL")
    if not value and provider == "official_openai": value = DEFAULT_BASE_URL
    if not value: raise RuntimeError("Set the base URL for the selected text provider")
    return value.rstrip("/")


def _get_model(provider: str, configured_model: str | None) -> str:
    value = os.getenv("TEXT_PROVIDER_MODEL") or os.getenv({"official_openai":"OFFICIAL_OPENAI_MODEL", "openai_compatible":"META_AGENT_COMPATIBLE_MODEL_OVERRIDE"}[provider])
    return value or configured_model or DEFAULT_MODEL


def chat_model(
    configured_model: str | None = None,
    *,
    timeout: float = 90.0,
    max_retries: int = 3,
) -> ChatOpenAI:
    provider = _provider()
    model = _get_model(provider, configured_model)
    settings = {
        "model": model,
        "api_key": _get_api_key(provider),
        "base_url": _get_base_url(provider),
        "timeout": timeout,
        "max_retries": max_retries,
    }
    # Official Chat Completions rejects function tools for GPT-5 reasoning
    # models unless reasoning is disabled. Extraction uses function-calling
    # structured output and validates the result locally with Pydantic, so
    # explicitly select the supported non-reasoning tool mode. Compatible
    # gateways keep their existing parameters unchanged.
    if provider == "official_openai" and model.casefold().startswith("gpt-5"):
        settings["reasoning_effort"] = "none"
    return ChatOpenAI(
        **settings,
    )
