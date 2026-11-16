"""Project-selected text and embedding providers with isolated credentials."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import subprocess

from .common import require, digest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROVIDER_CONFIG = ROOT / "config" / "provider.json"
PROVIDER_CONFIG_SCHEMA = "meta_pipeline/provider_selection/1.0"
LEGACY_PROVIDER_CONFIG_SCHEMA = "meta_pipeline_integrated/provider_selection/1.0"

TEXT_NAMES = {"official_openai", "openai_compatible"}
EMBEDDING_NAMES = {"official_openai", "openai_compatible"}
TEXT_KEY_FIELDS = (
    "PIPELINE_LLM_API_KEY", "COMPATIBLE_API_KEY", "ANALYSIS_LLM_API_KEY",
    "BRAIN_API_KEY", "BIAS_API_KEY",
)
TEXT_URL_FIELDS = (
    "PIPELINE_LLM_BASE_URL", "COMPATIBLE_BASE_URL", "ANALYSIS_LLM_BASE_URL",
    "BRAIN_BASE_URL", "BIAS_BASE_URL",
)
EMBEDDING_KEY_FIELDS = ("ANALYSIS_EMBEDDING_API_KEY",)
EMBEDDING_URL_FIELDS = ("ANALYSIS_EMBEDDING_BASE_URL",)


def file_values(env_file):
    """Read the simple KEY=VALUE format used by the private source configuration.

    This keeps provider-check available before optional project dependencies are
    installed. The frozen modules still receive only the derived child env.
    """
    path = Path(env_file)
    require(path.is_file(), "missing env file")
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        require(key and key.replace("_", "a").isalnum(), "Invalid env variable name")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def selection_values(config_file=DEFAULT_PROVIDER_CONFIG):
    """Read the non-secret, automatically loaded provider selection."""
    path = Path(config_file)
    require(path.is_file(), "missing provider selection config: " + str(path))
    import json
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema_version") in {PROVIDER_CONFIG_SCHEMA, LEGACY_PROVIDER_CONFIG_SCHEMA, "meta_pipeline_legacy/provider_selection/1.0", "meta_pipeline_compatible/provider_selection/1.0"},
            "invalid provider selection config schema_version")
    text = payload.get("text_provider")
    embedding = payload.get("embedding_provider")
    require(set(payload) == {"schema_version", "text_provider", "embedding_provider"}, "Provider config accepts only schema_version, text_provider and embedding_provider; remove obsolete alternate routes")
    require(isinstance(text, dict) and isinstance(embedding, dict),
            "provider selection config requires text_provider and embedding_provider")
    require(set(text) == {"name", "base_url", "model"}, "invalid text_provider fields")
    require(set(embedding) == {"name", "base_url", "model"}, "invalid embedding_provider fields")
    return {
        "TEXT_PROVIDER_NAME": _text(text["name"], "text_provider.name"),
        "TEXT_PROVIDER_BASE_URL": _text(text["base_url"], "text_provider.base_url"),
        "TEXT_PROVIDER_MODEL": _text(text["model"], "text_provider.model"),
        "EMBEDDING_PROVIDER_NAME": _text(embedding["name"], "embedding_provider.name"),
        "EMBEDDING_PROVIDER_BASE_URL": _text(embedding["base_url"], "embedding_provider.base_url"),
        "EMBEDDING_PROVIDER_MODEL": _text(embedding["model"], "embedding_provider.model"),
    }


def _load(env_file=None):
    # Provider choice is project-level, rather than a field of every launch
    # request. Environment values may deliberately override this local default.
    env = selection_values()
    env.update(os.environ)
    if env_file:
        env.update(file_values(env_file))
    return env


def _text(value, name):
    require(isinstance(value, str) and bool(value.strip()), "Set " + name)
    return value.strip()


def _no_conflict(env, fields, expected, description):
    for field in fields:
        if env.get(field):
            require(env[field].rstrip("/") == expected.rstrip("/"), f"Conflicting {field}; set only {description} settings")


def _profile(env, request=None):
    text_name = env.get("TEXT_PROVIDER_NAME", "official_openai").strip()
    require(text_name in TEXT_NAMES, "TEXT_PROVIDER_NAME must be official_openai or openai_compatible")
    raw_text_key = env.get("TEXT_PROVIDER_API_KEY") or (env.get("OPENAI_API_KEY") if text_name == "official_openai" else None)
    text_key = _text(raw_text_key, "OPENAI_API_KEY or TEXT_PROVIDER_API_KEY" if text_name == "official_openai" else "TEXT_PROVIDER_API_KEY")
    text_default_base = "https://api.openai.com/v1" if text_name == "official_openai" else ""
    text_base = _text(env.get("TEXT_PROVIDER_BASE_URL", text_default_base), "TEXT_PROVIDER_BASE_URL").rstrip("/")
    default_model = (request or {}).get("options", {}).get("model", "gpt-5.6-luna")
    text_model = _text(env.get("TEXT_PROVIDER_MODEL", default_model), "TEXT_PROVIDER_MODEL")
    embedding_name = env.get("EMBEDDING_PROVIDER_NAME", "official_openai").strip()
    require(embedding_name in EMBEDDING_NAMES, "EMBEDDING_PROVIDER_NAME must be official_openai or openai_compatible")
    raw_embedding_key = env.get("EMBEDDING_PROVIDER_API_KEY") or (env.get("OPENAI_API_KEY") if embedding_name == "official_openai" else None)
    embedding_key = _text(raw_embedding_key, "OPENAI_API_KEY or EMBEDDING_PROVIDER_API_KEY" if embedding_name == "official_openai" else "EMBEDDING_PROVIDER_API_KEY")
    embedding_default_base = "https://api.openai.com/v1" if embedding_name == "official_openai" else ""
    embedding_base = _text(env.get("EMBEDDING_PROVIDER_BASE_URL", embedding_default_base), "EMBEDDING_PROVIDER_BASE_URL").rstrip("/")
    embedding_model = _text(env.get("EMBEDDING_PROVIDER_MODEL", "text-embedding-3-small"), "EMBEDDING_PROVIDER_MODEL")
    _no_conflict(env, TEXT_KEY_FIELDS, text_key, "TEXT_PROVIDER")
    _no_conflict(env, TEXT_URL_FIELDS, text_base, "TEXT_PROVIDER")
    _no_conflict(env, EMBEDDING_KEY_FIELDS, embedding_key, "EMBEDDING_PROVIDER")
    _no_conflict(env, EMBEDDING_URL_FIELDS, embedding_base, "EMBEDDING_PROVIDER")
    if env.get("ANALYSIS_EMBEDDING_MODEL"):
        require(env["ANALYSIS_EMBEDDING_MODEL"] == embedding_model, "Conflicting ANALYSIS_EMBEDDING_MODEL")
    for field in ("UNIFIED_PROVIDER_API_KEY", "UNIFIED_PROVIDER_BASE_URL", "UNIFIED_PROVIDER_MODEL", "OFFICIAL_OPENAI_BASE_URL", "OFFICIAL_OPENAI_MODEL"):
        require(not env.get(field), f"Remove {field}; configure endpoints in config/provider.json or .env")
    if text_name != "official_openai" and embedding_name != "official_openai":
        require(not env.get("OPENAI_API_KEY"), "Remove OPENAI_API_KEY when neither text nor embedding route uses official_openai")
    return {
        "schema_version": "provider_routes/1.0",
        "text": {"name": text_name, "base_url": text_base, "model": text_model},
        "embedding": {"name": embedding_name, "base_url": embedding_base, "model": embedding_model},
        "routing": {"screening": "text", "extraction": "text", "brain": "text", "bias": "text", "analysis_llm": "text", "analysis_embeddings": "embedding"},
        "text_and_embedding_are_isolated": True,
    }


def profile(env_file=None, request=None):
    """Validate and resolve non-secret text and embedding routes."""
    return _profile(_load(env_file), request)


def environment(env_file=None, request=None):
    """Build child settings for the explicitly selected, isolated text and embedding routes."""
    env = _load(env_file)
    p = _profile(env, request)
    text, embedding = p["text"], p["embedding"]
    text_key = env.get("TEXT_PROVIDER_API_KEY") or (env["OPENAI_API_KEY"] if text["name"] == "official_openai" else "")
    embedding_key = env.get("EMBEDDING_PROVIDER_API_KEY") or (env["OPENAI_API_KEY"] if embedding["name"] == "official_openai" else "")
    env["TEXT_PROVIDER_API_KEY"] = text_key
    env["EMBEDDING_PROVIDER_API_KEY"] = embedding_key
    for field in TEXT_KEY_FIELDS:
        env[field] = text_key
    for field in TEXT_URL_FIELDS:
        env[field] = text["base_url"]
    env["COMPATIBLE_API_KEY"] = text_key
    env["COMPATIBLE_BASE_URL"] = text["base_url"]
    env["ANALYSIS_LLM_MODEL"] = text["model"]
    env["BRAIN_MODEL"] = text["model"]
    env["BIAS_MODEL"] = text["model"]
    env["ANALYSIS_EMBEDDING_API_KEY"] = embedding_key
    env["ANALYSIS_EMBEDDING_BASE_URL"] = embedding["base_url"]
    env["ANALYSIS_EMBEDDING_MODEL"] = embedding["model"]
    env["TEXT_PROVIDER_NAME"] = text["name"]
    env["TEXT_PROVIDER_BASE_URL"] = text["base_url"]
    env["TEXT_PROVIDER_MODEL"] = text["model"]
    env["EMBEDDING_PROVIDER_NAME"] = embedding["name"]
    env["EMBEDDING_PROVIDER_BASE_URL"] = embedding["base_url"]
    env["EMBEDDING_PROVIDER_MODEL"] = embedding["model"]
    return env, p


def require_local_documents(request):
    """Fail before frozen code blocks on an iCloud-only source file (macOS)."""
    unavailable = []
    for document in request.get("documents", []):
        paths = document.get("paths", {})
        # Match the preparation policy: an existing Markdown/content-list is
        # authoritative, so an archived PDF need not be downloaded merely to
        # run the pipeline.
        active_kinds = ("markdown", "content_list") if paths.get("markdown") or paths.get("content_list") else ("pdf",)
        for kind in active_kinds:
            raw_path = paths.get(kind)
            if kind not in {"markdown", "content_list", "pdf"} or not raw_path:
                continue
            path = Path(raw_path)
            try:
                result = subprocess.run(["stat", "-f", "%Sf", str(path)], capture_output=True, text=True, timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode == 0 and "dataless" in result.stdout:
                unavailable.append(f"{document.get('article_id', 'unknown')}:{kind}")
    require(not unavailable, "Download iCloud-only source files before running: " + ", ".join(unavailable))


def runtime(request, env_file=None):
    """Route frozen pipeline children without changing their request contract."""
    require_local_documents(request)
    env, p = environment(env_file, request)
    text = p["text"]
    derived = deepcopy(request)
    adapter = text["name"]
    derived.setdefault("options", {})["provider"] = adapter
    derived["options"]["model"] = text["model"]
    # The local positive/negative corpus has an external Markdown derivative.
    # Prefer it consistently so MINERS does not require an optional PDF reader.
    for document in derived.get("documents", []):
        paths = document.get("paths", {})
        if paths.get("markdown") or paths.get("content_list"):
            paths.pop("pdf", None)
    from metapipeline.pipeline import runtime_environment
    built = runtime_environment(derived)
    built.update(env)
    built["PYTHONDONTWRITEBYTECODE"] = "1"
    built["META_AGENT_LLM_PROVIDER"] = adapter
    built["META_AGENT_COMPATIBLE_STRICT"] = "1"
    built["META_AGENT_COMPATIBLE_BASE_URL_OVERRIDE"] = text["base_url"]
    built["META_AGENT_COMPATIBLE_MODEL_OVERRIDE"] = text["model"]
    # Explicit official selection uses the same text credential, never embedding credentials.
    if adapter == "official_openai":
        built["OPENAI_API_KEY"] = env["TEXT_PROVIDER_API_KEY"]
        built["OFFICIAL_OPENAI_BASE_URL"] = text["base_url"]
        built["OFFICIAL_OPENAI_MODEL"] = text["model"]
    else:
        built.pop("OPENAI_API_KEY", None)
    return derived, built, p


def fingerprint(p):
    return digest({"schema_version": p["schema_version"], "text": p["text"], "embedding": p["embedding"], "routing": p["routing"]})


def audit(p):
    return {**p, "fingerprint": fingerprint(p), "credentials_written": False}
