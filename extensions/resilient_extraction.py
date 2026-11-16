"""Checkpointed, bounded stage-A2 extraction coordinator.

This coordinator composes the typed locator/extractor tools.
It keeps deterministic preparation and one locator decision per
article, then extracts the locator-selected source regions in bounded batches.
Every completed batch is persisted so a provider failure never restarts a
completed article from its first model call.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import importlib
import json
import os
import re
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extensions.common import activate
activate()
from trace_runtime.contracts import ContractError, digest, read, require, write


STATE_SCHEMA = "trace/article_extraction_state/1.0"
BATCH_SCHEMA = "trace/extraction_batch/1.0"
EXTRACTION_CONTRACT_VERSION = "source_transcription/2.2"
_FAMILY_CHECKPOINT_DIR: ContextVar[Path | None] = ContextVar("resilient_family_checkpoint_dir", default=None)
_OUTPUT_MODEL_CACHE: dict[tuple, Any] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    value = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def error_info(exc: Exception) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "status_code": _status_code(exc),
    }


def is_permanent_provider_error(exc: Exception) -> bool:
    """Do not spend four full article attempts on bad credentials or quota."""
    if _status_code(exc) in {400, 401, 403, 404, 422}:
        return True
    message = str(exc).casefold()
    return any(token in message for token in (
        "insufficient_user_quota", "credit_balance_exhausted", "model_not_found",
        "insufficient_quota", "billing_hard_limit_reached",
    ))


def is_capacity_error(exc: Exception) -> bool:
    """Recognize provider throttling even when it is not exposed as HTTP 429.

    OpenAI-compatible gateways do not consistently encode request-capacity
    failures.  Some return a normal 429, while others wrap a provider-side
    ``model:rate_limited`` response in a 503/InternalServerError.  Both mean
    that replaying the same locator or extraction request immediately is
    counterproductive.  Keep this classification message-specific so an
    ordinary transient 503 can still use the bounded transport retry.
    """
    message = str(exc).casefold()
    return _status_code(exc) == 429 or any(token in message for token in (
        "concurrency limit",
        "gateway_concurrency_limit",
        "model:rate_limited",
        "rate_limited",
        "rate limited",
        "too frequently",
        "too many requests",
        "throttled",
        "throttling",
    ))


def is_provider_outage_error(exc: Exception) -> bool:
    """Errors for which an in-process retry only makes a stalled run longer.

    Cloudflare 524 means the upstream accepted the request but did not finish
    within its fixed window.  Retrying immediately resends the same large
    prompt while the origin is still overloaded.  A local SDK timeout has the
    same operational meaning here.  Preserve the checkpoints and let an
    explicit resume retry after the provider has recovered.
    """
    return _status_code(exc) == 524 or type(exc).__name__ == "APITimeoutError"


def is_retryable_provider_error(exc: Exception) -> bool:
    if is_permanent_provider_error(exc):
        return False
    # Capacity and origin-outage errors are handled by the outer scheduler.
    # Do not make a worker sleep and replay a large request while peer workers
    # continue sending traffic to the same constrained provider.
    if is_capacity_error(exc) or is_provider_outage_error(exc):
        return False
    if _status_code(exc) in {408, 409, 500, 502, 503, 504}:
        return True
    return type(exc).__name__ in {
        "APIConnectionError", "InternalServerError",
    }




def retry_delay_seconds(exc: Exception, attempt: int) -> float:
    message = str(exc).casefold()
    if _status_code(exc) == 524 or "error code: 524" in message:
        return 120.0
    if is_capacity_error(exc):
        return min(90.0, 20.0 * attempt)
    if "timeout" in message:
        return min(60.0, 10.0 * attempt)
    return min(45.0, 5.0 * attempt)


def _call_attempts(request: dict) -> int:
    raw = os.environ.get("TRACE_EXTRACT_CALL_ATTEMPTS")
    if raw is None:
        # The archived request contract commonly has max_retries=3.  Combining
        # that with its former article-wide retry loop was the main source of
        # repeated long requests.  This coordinator keeps one bounded retry for transient
        # transport faults; completed batches make further attempts safe to
        # request explicitly through --resume.
        # Initial request plus two retries; a 524/capacity/error route moves
        # to the scheduler for an explicit resume without waiting in place.
        return 3
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ContractError("TRACE_EXTRACT_CALL_ATTEMPTS must be an integer") from exc


def _batch_budget() -> int:
    raw = os.environ.get("TRACE_EXTRACT_BATCH_CHARS", "20000")
    try:
        return max(4000, int(raw))
    except ValueError as exc:
        raise ContractError("TRACE_EXTRACT_BATCH_CHARS must be an integer") from exc


def _extract_timeout_seconds() -> float:
    """Wait for an upstream response before imposing a local extraction timeout."""
    raw = os.environ.get("TRACE_EXTRACT_TIMEOUT_SECONDS", "300")
    try:
        return max(1.0, float(raw))
    except ValueError as exc:
        raise ContractError("TRACE_EXTRACT_TIMEOUT_SECONDS must be a number") from exc


def _load_runtime(request: dict):
    """Load the frozen runtime chosen by the existing request contract."""
    from trace_runtime.worker import load_extractor

    load_case_config, frozen_run_article = load_extractor(request)
    controller = sys.modules[frozen_run_article.__module__]
    tools = importlib.import_module(controller.__package__ + ".tools")
    models = importlib.import_module(controller.__package__ + ".extraction_models")
    # The Baseline frozen helper silently asks its SDK for three additional retries.
    # The coordinator owns retry/backoff at the individual locator/batch boundary, so hidden
    # SDK retries would multiply both latency and provider load.
    if request["options"].get("extraction_runtime") == "baseline" and not getattr(tools, "_resilient_retry_controlled", False):
        # Baseline's frozen endpoint chooses a provider through process environment.
        # A ContextVar gives each concurrent article call its own provider, and
        # a tiny construction lock prevents one model object reading another
        # thread's temporary provider setting.  Once created, ChatOpenAI holds
        # its own credentials/base URL, so invocation does not need the lock.
        tools._resilient_active_provider = ContextVar("resilient_extraction_provider", default=request["options"].get("provider", "official_openai"))
        tools._resilient_provider_lock = threading.Lock()
        def resilient_llm(task, role):
            provider = tools._resilient_active_provider.get()
            with tools._resilient_provider_lock:
                previous = os.environ.get("META_AGENT_LLM_PROVIDER")
                os.environ["META_AGENT_LLM_PROVIDER"] = provider
                try:
                    return tools.chat_model(
                        task.raw.get("models", {}).get(role),
                        timeout=_extract_timeout_seconds(),
                        max_retries=0,
                    )
                finally:
                    if previous is None:
                        os.environ.pop("META_AGENT_LLM_PROVIDER", None)
                    else:
                        os.environ["META_AGENT_LLM_PROVIDER"] = previous
        tools._llm = resilient_llm
        tools._resilient_retry_controlled = True
    if request["options"].get("extraction_runtime") == "typed" and not getattr(tools, "_resilient_timeout_controlled", False):
        # Mirror the frozen provider lock and routing behaviour, but do not
        # impose its 110-second local timeout before an upstream gateway can
        # return its own diagnostic response.
        def resilient_llm(task, role, provider="official_openai"):
            with tools._PROVIDER_ENV_LOCK:
                previous = os.environ.get("META_AGENT_LLM_PROVIDER")
                os.environ["META_AGENT_LLM_PROVIDER"] = provider
                try:
                    return tools.chat_model(
                        task.raw.get("models", {}).get(role),
                        timeout=_extract_timeout_seconds(),
                        max_retries=0,
                    )
                finally:
                    if previous is None:
                        os.environ.pop("META_AGENT_LLM_PROVIDER", None)
                    else:
                        os.environ["META_AGENT_LLM_PROVIDER"] = previous
        tools._llm = resilient_llm
        tools._resilient_timeout_controlled = True
    return load_case_config, controller, tools, models


def _call_static_clean(tools, task, document, is_typed: bool):
    if is_typed:
        return tools.static_clean_document(
            document,
            remove_reference_items_only=True,
        )
    return tools.static_clean_document(document)


@contextmanager
def _provider_context(tools, provider: str):
    """Bind the Baseline sidecar's provider for this thread's model construction."""
    active = getattr(tools, "_resilient_active_provider", None)
    if active is None:
        yield
        return
    token = active.set(provider)
    try:
        yield
    finally:
        active.reset(token)


def _call_locator(tools, task, catalog, is_typed: bool, provider: str):
    with _provider_context(tools, provider):
        if is_typed:
            return tools.locate_relevant_sources(task, catalog, provider)
        return tools.locate_relevant_sources(task, catalog)


def _compact_extraction_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the model-only source view for extraction.

    The complete source catalog stays local and is passed unchanged to the
    deterministic organizer.  ``label`` is a preview duplicated from content,
    while ``page`` can be recovered from the catalog by ``source_id``.  Neither
    is a model output field or part of source-quote verification.
    """
    return [
        {
            "source_id": source["source_id"],
            "source_type": source["source_type"],
            "content": source["content"],
            **({"context_only": True} if source.get("context_only") else {}),
        }
        for source in sources
    ]


def _template_families(task, tools) -> set[str]:
    aliases = getattr(tools, "DATA_FAMILY_ALIASES", {})
    return {
        aliases.get(family, family)
        for target in task.raw.get("extraction", {}).get("targets", [])
        for family in target.get("accepted_data_families", [])
    }


def _route_template_families(sources, families: set[str]) -> set[str]:
    """Keep every protocol-allowed family; optional calls resolve applicability.

    A paper can contain comparative and single-arm summaries together. Word
    counts at article level cannot safely remove a permitted numeric family.
    """
    return set(families)


def _family_target_ids(task, tools, family: str) -> list[str]:
    aliases = getattr(tools, "DATA_FAMILY_ALIASES", {})
    return sorted({str(target["target_id"])
                   for target in task.raw.get("extraction", {}).get("targets", [])
                   if target.get("target_id") and family in {
                       aliases.get(value, value) for value in target.get("accepted_data_families", [])}})


def _single_family_output_model(tools, family: str, target_ids: list[str] | None = None):
    """Create a strict extraction envelope containing one row schema only."""
    from typing import Literal
    from pydantic import ConfigDict, create_model

    models = getattr(tools, "DATA_FAMILY_TEMPLATE_MODELS", {})
    require(family in models, f"missing Pydantic template for routed family: {family}")
    row_model = models[family]
    cache_key = (row_model, family, tuple(target_ids) if target_ids is not None else None)
    if cache_key in _OUTPUT_MODEL_CACHE:
        return _OUTPUT_MODEL_CACHE[cache_key]
    # The legacy comparative-effect base model permits a nullable free-text
    # family because it predates routed templates.  Under Typed, specialize that
    # field to the one selected route so structured output cannot legally omit
    # it or return a sibling family.
    if family in set(getattr(tools, "LEGACY_EFFECT_FAMILIES", set())):
        row_model = create_model(
            "RoutedRow_" + re.sub(r"[^A-Za-z0-9_]", "_", family),
            __base__=row_model,
            data_family=(Literal[family], family),
        )
    if target_ids is not None:
        require(bool(target_ids), f"no declared extraction target accepts family: {family}")
        row_model = create_model(
            "TargetedRow_" + re.sub(r"[^A-Za-z0-9_]", "_", family),
            __base__=row_model, target_id=(Literal[tuple(target_ids)], ...),
        )
    output_model = create_model(
        "RoutedDataFamilyExtraction_" + re.sub(r"[^A-Za-z0-9_]", "_", family),
        __config__=ConfigDict(extra="forbid"),
        rows=(list[row_model], ...),
        unresolved_items=(list[str], ...),
    )
    _OUTPUT_MODEL_CACHE[cache_key] = output_model
    return output_model


def _schema_fields(schema: dict, definitions: dict) -> str:
    """Render a short, field-name-level contract from a frozen Pydantic schema."""
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    pieces = ["row required=" + ",".join(required)]
    raw = properties.get("raw_data", {})
    ref = raw.get("$ref", "") if isinstance(raw, dict) else ""
    definition = definitions.get(ref.rsplit("/", 1)[-1], {}) if ref else {}
    raw_properties = definition.get("properties", {}) if isinstance(definition, dict) else {}
    raw_required = definition.get("required", []) if isinstance(definition, dict) else []
    if raw_properties:
        nested = []
        for field, value in raw_properties.items():
            if isinstance(value, dict) and "$ref" in value:
                nested_name = value["$ref"].rsplit("/", 1)[-1]
                nested_schema = definitions.get(nested_name, {})
                nested_fields = ",".join(nested_schema.get("properties", {}))
                nested.append(f"{field}{{{nested_fields}}}")
            elif isinstance(value, dict) and isinstance(value.get("additionalProperties"), dict) and "$ref" in value["additionalProperties"]:
                nested_name = value["additionalProperties"]["$ref"].rsplit("/", 1)[-1]
                nested_schema = definitions.get(nested_name, {})
                nested_fields = ",".join(nested_schema.get("properties", {}))
                nested.append(f"{field}{{key:{{{nested_fields}}}}}")
            else:
                nested.append(field)
        pieces.append("raw_data required=" + ",".join(raw_required) + "; fields=" + ",".join(nested))
    return "; ".join(pieces)


def _compact_template_contract(tools, families: set[str], *, full_schema: bool) -> str:
    """Send routed field contracts, retaining full schemas only as recovery."""
    if full_schema:
        return tools._template_schemas(families)
    schemas = json.loads(tools._template_schemas(families))
    lines = [
        'Return one JSON object: {"rows":[...],"unresolved_items":[...]}.',
        "Every row needs outcome, exposure, comparator, population_or_subgroup, estimand, source_id, source_quote, data_family, and raw_data.",
        "Use only a listed data_family. Keep source numbers verbatim; do not calculate an effect.",
    ]
    for family in sorted(schemas):
        schema = schemas[family]
        lines.append(f"{family}: {_schema_fields(schema, schema.get('$defs', {}))}")
    return "\n".join(lines)


def _compact_extraction_prompt(tools, task, sources, feedback: str, is_typed: bool,
                               template_families: set[str] | None = None,
                               *, full_schema: bool = False):
    """Build the external compact prompt and routed template contract.

    Full frozen Pydantic schemas previously travelled in every template request.
    Derive a concise field contract for only the source-routed families.
    The full frozen schema is used only for an immediate local-validation
    recovery, while the local parser still accepts all protocol-allowed rows.
    """
    from langchain_core.prompts import ChatPromptTemplate

    targets = task.raw.get("extraction", {}).get("targets", [])
    raw_families = {
        family
        for target in targets
        for family in target.get("accepted_data_families", [])
    }
    template_partitions = set(getattr(
        tools,
        "TEMPLATE_PARTITIONS",
        {"locator_extractor_structured_23", "locator_extractor_typed_data_family"},
    ))
    compact_sources = json.dumps(_compact_extraction_sources(sources), ensure_ascii=False)
    template_mode = is_typed and task.raw.get("system_partition") in template_partitions

    if not template_mode:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are the extraction stage for known-included reports. Your job is transcription and structured extraction, not screening, planning, criticism, study selection, or meta-analysis. Inspect every located source and return every source-supported candidate row matching the target exposure, comparator, and outcome concepts.

The PICO tells you what to look for. Do not suppress a reported candidate merely because one population qualifier is not restated in the located excerpt, because the article population is broader, or because a reported model/effect label differs from the preferred model. Preserve those facts in population_or_subgroup, model_name, adjustment, effect_type, and unresolved_items. Downstream analysis will decide final eligibility and poolability.

Use the target's default quantitative_usability. Use source_only only when the target explicitly requires it or when the accepted data family is inherently source-only. Never infer or manufacture values, never convert effect scales silently, and quote the exact contiguous source text supporting each value. Return one row per distinct target, subgroup, timepoint, comparison, and model.

Data-family instructions:\n{family_instructions}""",
                ),
                (
                    "user",
                    "Extraction contract:\n{contract}\n\nLocated source regions:\n{sources}\n\nRetry feedback:\n{feedback}",
                ),
            ]
        )
        return "structured", prompt, {
            "contract": json.dumps(_runtime_contract(tools, task), ensure_ascii=False),
            "sources": compact_sources,
            "feedback": feedback or "No prior extraction attempt.",
            "family_instructions": tools._family_instructions(raw_families),
        }, raw_families

    allowed_families = _template_families(task, tools)
    display_families = set(template_families or allowed_families)
    require(display_families and display_families.issubset(allowed_families), "invalid routed template family")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are the extraction stage for known-included reports. Your job is transcription and structured extraction, not screening, planning, criticism, study selection, or meta-analysis. Inspect every located source and return every source-supported candidate row matching the target exposure, comparator, and outcome concepts.

The PICO tells you what to look for. Do not suppress a reported candidate merely because one population qualifier is not restated in the located excerpt, because the article population is broader, or because a reported model/effect label differs from the preferred model. Preserve those facts in population_or_subgroup, model_name, adjustment, effect_type, and unresolved_items. Downstream analysis will decide final eligibility and poolability.

If no numeric template fits the located evidence, use generic_text_evidence to preserve a plain-language source description without inventing numeric values.

Use the target's default quantitative_usability when applicable. For source-data templates, use calculator_eligible only for complete source data and extraction_only or requires_analysis_profile when the source lacks a calculable contrast or needs later profile selection. Never infer or manufacture values, never convert effect scales silently, and quote the exact contiguous source text supporting each value. Return one row per distinct target, subgroup, timepoint, comparison, and model.

Follow the routed template field contract exactly. Return exactly one JSON object with keys rows and unresolved_items. Do not wrap JSON in prose or Markdown. Choose generic_text_evidence only when none of the supplied numeric templates fits the source; never use it to avoid transcribing available values.

Data-family instructions:\n{family_instructions}""",
            ),
            (
                "user",
                "Extraction contract:\n{contract}\n\nRouted template field contract:\n{template_schemas}\n\nLocated source regions:\n{sources}\n\nRetry feedback:\n{feedback}",
            ),
        ]
    )
    inputs = {
        "contract": json.dumps(_runtime_contract(tools, task), ensure_ascii=False),
        "template_schemas": _compact_template_contract(tools, display_families, full_schema=full_schema),
        "sources": compact_sources,
        "feedback": feedback or "No prior extraction attempt.",
        "family_instructions": tools._family_instructions(display_families),
    }
    # Validation remains broad even when the prompt is routed.  This stops a
    # conservative router from silently discarding another allowed family.
    return "template", prompt, inputs, allowed_families


def _runtime_contract(tools, task) -> dict[str, Any]:
    """Return the compact runtime contract; schemas carry field constraints."""
    return dict(tools._contract(task))


def _single_family_contract(tools, task, family: str) -> dict[str, Any]:
    contract = deepcopy(_runtime_contract(tools, task))
    aliases = getattr(tools, "DATA_FAMILY_ALIASES", {})
    if "targets" in contract:
        contract["targets"] = [target for target in contract["targets"]
                               if family in {aliases.get(value, value) for value in target.get("accepted_data_families", [])}]
        for target in contract["targets"]:
            target["accepted_data_families"] = [family]
    return contract


def _uses_family_templates(task, tools, is_typed: bool) -> bool:
    return is_typed and task.raw.get("system_partition") in set(getattr(
        tools, "TEMPLATE_PARTITIONS", {"locator_extractor_structured_23", "locator_extractor_typed_data_family"}))


def _compact_prompt_chars(tools, task, sources, is_typed: bool,
                          template_families: set[str] | None = None) -> int:
    """Count actual first-call message text; schema is accounted separately."""
    if _uses_family_templates(task, tools, is_typed):
        selected = set(template_families or _template_families(task, tools))
        sizes = []
        for family in selected:
            output_model = _single_family_output_model(tools, family, _family_target_ids(task, tools, family))
            prompt, inputs = _single_family_prompt(tools, task, sources, "", output_model, family)
            sizes.append(sum(len(str(message.content)) for message in prompt.format_messages(**inputs)))
        return max(sizes, default=0)
    _, prompt, inputs, _ = _compact_extraction_prompt(tools, task, sources, "", is_typed, template_families)
    return sum(len(str(message.content)) for message in prompt.format_messages(**inputs))


def _family_transcription_rules(tools, family: str) -> str:
    """Short semantic rules complement the schema without repeating its fields."""
    original = tools._family_instructions({family}) if hasattr(tools, "_family_instructions") else ""
    rules = {
        "binary_arm_counts": "Copy only explicitly reported integer event counts and arm denominators for the same outcome and follow-up. Never reconstruct counts from percentages, a Kaplan–Meier curve, survival estimates, HRs, or rates. A percentage times baseline N is not a reported count. Do not substitute baseline enrollment for the outcome-specific denominator. If any required count is missing, preserve the reported values and limitation in unresolved_items.",
        "single_arm_event_counts": "Copy explicitly reported events and their matching denominator. Never back-calculate an event count from percentages, survival estimates, incidence rates or person-time.",
        "percentage_only": "Preserve printed percentages and their reported denominators; never reconstruct counts or treat Kaplan–Meier survival as an observed event fraction.",
        "independent_continuous_arms": "Use two independent named groups with reported n, mean and SD. Paired/planned-versus-achieved measurements on the same subjects are not independent arms. Never substitute SE, CI, range, zero or epsilon for a missing SD.",
        "continuous_arm_summary": "Preserve the source's group n, mean and SD. Do not convert median/IQR, SE, CI or range to mean/SD during transcription. Keep missing quantities unresolved.",
        "continuous_change_score_summary": "Use the source's change-score mean and SD. Baseline and follow-up SDs are not change-score SDs. Do not invent correlations or derive missing quantities.",
        "single_arm_continuous_summary": "Keep reported summaries and planned targets distinct. Never construct a comparator or substitute a planned target for an observed group. Do not invent SDs or convert SE/ranges to SDs.",
        "diagnostic_four_cell_counts": "Copy the four reported cells for one test, threshold and reference standard. Do not reconstruct cells from rounded sensitivity/specificity or combine different test sets.",
        "complete_genotype_case_control_counts": "Preserve every printed genotype label and nonnegative case/control count for one variant. Do not infer genotype counts from allele frequencies or merge distinct populations.",
        "allele_frequency_summary_only": "Preserve printed allele frequencies and sample sizes. Do not infer genotypes or treat allele counts as participant counts.",
        "event_counts_with_person_time": "Preserve episodes, persons, observation time and printed rate units separately. Recurrent episodes can exceed persons; person-time is not participant N. Never derive counts from rounded rates.",
    }
    return "\n".join(part for part in (original, rules.get(family, "Preserve effect scale, confidence bounds, reference category, timepoint, model and adjustment exactly. Do not convert HR to RR or derive missing uncertainty.")) if part)


def _validate_numeric_rows(parsed, source_json: str) -> None:
    """Enforce arithmetic/identity invariants; this is not source verification.

    Numeric validity is necessary but never certifies that a number was printed.
    The evidence reviewer receives raw_data and source context for that judgment.
    """
    import math
    sources = json.loads(source_json)
    primary_ids = {source.get("source_id") for source in sources if not source.get("context_only")}
    for index, row in enumerate(parsed.model_dump().get("rows", [])):
        if primary_ids and row.get("source_id") not in primary_ids:
            raise ValueError(f"rows.{index}.source_id must name a supplied primary source, not a context-only block")
        raw = row.get("raw_data") or {}
        family = row.get("data_family")
        if family == "binary_arm_counts":
            for arm in ("intervention", "control"):
                if raw.get(arm + "_events", 0) > raw.get(arm + "_total", 0):
                    raise ValueError(f"rows.{index}: {arm} events exceed its denominator")
        if family == "complete_genotype_case_control_counts":
            labels = raw.get("genotype_labels", [])
            if len(labels) != len(set(labels)):
                raise ValueError(f"rows.{index}: duplicate genotype labels")
            for group in ("case_counts", "control_counts"):
                counts = raw.get(group, {})
                if set(counts) != set(labels) or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts.values()):
                    raise ValueError(f"rows.{index}: {group} must contain every declared genotype with a nonnegative integer count")
        lower, upper, effect = row.get("ci_lower"), row.get("ci_upper"), row.get("effect_value")
        if family in {"reported_ratio_effect", "adjusted_ratio_effect", "time_to_event"}:
            if any(value is not None and value <= 0 for value in (effect, lower, upper)):
                raise ValueError(f"rows.{index}: reported ratio effects and confidence limits must be positive; do not substitute log effects")
        if family == "diagnostic_four_cell_counts" and sum(raw.get(key, 0) for key in (
                "true_positive", "false_positive", "false_negative", "true_negative")) <= 0:
            raise ValueError(f"rows.{index}: diagnostic test set must contain at least one participant")
        if family == "event_counts_with_person_time":
            rates = raw.get("reported_rates_per_person_month") or {}
            if any(value < 0 for value in rates.values()):
                raise ValueError(f"rows.{index}: reported event rates must be nonnegative")
        if lower is not None and upper is not None:
            if lower > upper:
                raise ValueError(f"rows.{index}: confidence limits must be ordered")
        def finite(value):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"rows.{index}: numeric evidence must be finite")
            if isinstance(value, dict):
                for v in value.values(): finite(v)
            elif isinstance(value, list):
                for v in value: finite(v)
        finite(row)


def _quarantine_numeric_rows(parsed, source_json: str, *, empty_diagnostics_only=False):
    """Preserve invalid numeric evidence as explicit issues, keeping valid rows.

    This never edits a reported number. The full rejected row is retained in
    unresolved_items, so an unsupported calculation cannot silently disappear.
    """
    from types import SimpleNamespace
    rows = []
    issues = list(parsed.unresolved_items)
    for row in parsed.rows:
        payload = row.model_dump()
        wrapper = SimpleNamespace(model_dump=lambda payload=payload: {"rows": [payload]})
        if empty_diagnostics_only:
            raw = payload.get("raw_data") or {}
            if payload.get("data_family") != "diagnostic_four_cell_counts" or any(
                    raw.get(key, 0) for key in ("true_positive", "false_positive", "false_negative", "true_negative")):
                rows.append(payload)
                continue
        try:
            _validate_numeric_rows(wrapper, source_json)
        except ValueError as exc:
            issues.append("Numeric evidence retained for review; no quantitative row emitted: " +
                          json.dumps({"reason": str(exc), "source_row": payload}, ensure_ascii=False))
        else:
            rows.append(payload)
    return type(parsed).model_validate({"rows": rows, "unresolved_items": list(dict.fromkeys(issues))})


def _semantic_fingerprint(task, tools, is_typed: bool) -> str:
    contract = _runtime_contract(tools, task) if hasattr(tools, "_contract") else getattr(task, "raw", {})
    families = _template_families(task, tools) if is_typed else set()
    schemas = {family: _single_family_output_model(tools, family, _family_target_ids(task, tools, family)).model_json_schema()
               for family in sorted(families)} if hasattr(tools, "DATA_FAMILY_TEMPLATE_MODELS") else {}
    return digest({"version": EXTRACTION_CONTRACT_VERSION, "contract": contract,
                   "models": getattr(task, "raw", {}).get("models", {}), "schemas": schemas})


def _request_schema_chars(task, tools, is_typed: bool, template_families=None) -> int:
    if not is_typed or not hasattr(tools, "DATA_FAMILY_TEMPLATE_MODELS"):
        return 0
    from langchain_core.utils.function_calling import convert_to_openai_tool
    families = set(template_families or _template_families(task, tools))
    return max((len(json.dumps(convert_to_openai_tool(
        _single_family_output_model(tools, family, _family_target_ids(task, tools, family))),
        ensure_ascii=False, separators=(",", ":"))) for family in families), default=0)


def _check_reported_sd(parsed, source_json: str):
    """Detect SD values not recognized in source text for semantic review.

    Lexical absence is not proof of fabrication: PDF typography can alter
    decimal/exponent formatting. The caller preserves the row and flags its
    numeric provenance instead of treating this heuristic as a hard gate.
    """
    from decimal import Decimal
    sources = json.loads(source_json)
    text = "\n".join(str(source.get("content", "")) for source in sources)
    import unicodedata
    text = unicodedata.normalize("NFKC", text).replace("−", "-")
    numbers = {Decimal(token.replace(",", "")) for token in re.findall(
        r"(?<![\w.])[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?", text
    )}

    def visit(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if (key == "sd" or key.endswith("_sd") or key.endswith("_sd_change")) and isinstance(item, (int, float)) and Decimal(str(item)) not in numbers:
                    raise ValueError(f"{path}.{key}={item}: SD not present in recognized source numbers; verify source typography and reported uncertainty. Do not invent a replacement or assume a placeholder is valid.")
                visit(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}")
    visit(parsed.model_dump(), "result")


def _annotate_sd_provenance(parsed, source_json: str):
    from types import SimpleNamespace
    rows = []
    issues = list(parsed.unresolved_items)
    for row in parsed.rows:
        payload = row.model_dump()
        wrapper = SimpleNamespace(model_dump=lambda payload=payload: {"rows": [payload]})
        try:
            _check_reported_sd(wrapper, source_json)
        except ValueError as exc:
            # Keep exact values for the source-aware reviewer. Do not turn a
            # tokenization mismatch into evidence deletion or fabricate an SD.
            if "quantitative_usability" in type(row).model_fields:
                payload["quantitative_usability"] = "needs_extraction"
            issues.append("Numeric provenance requires semantic review for source " +
                          str(payload.get("source_id")) + ": " + str(exc))
        rows.append(payload)
    return type(parsed).model_validate({"rows": rows, "unresolved_items": list(dict.fromkeys(issues))})


def _tool_arguments_from_raw(raw):
    """Return the model's tool arguments without trusting them as valid output.

    ``with_structured_output(include_raw=True)`` preserves an AI message when
    Pydantic rejects the parsed tool call.  Different OpenAI-compatible
    transports expose that call either through ``tool_calls`` or through the
    OpenAI ``additional_kwargs`` shape.  This helper merely retrieves the
    arguments for local, per-row validation; it never turns an invalid payload
    into accepted evidence.
    """
    def arguments(value):
        if isinstance(value, dict):
            value = value.get("args") or value.get("arguments") or value.get("function", {}).get("arguments")
        else:
            value = getattr(value, "args", None) or getattr(value, "arguments", None)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        return value if isinstance(value, dict) else None

    calls = getattr(raw, "tool_calls", None)
    if not calls and isinstance(raw, dict):
        calls = raw.get("tool_calls")
    if not calls:
        extra = getattr(raw, "additional_kwargs", None)
        if extra is None and isinstance(raw, dict):
            extra = raw.get("additional_kwargs")
        calls = (extra or {}).get("tool_calls", []) if isinstance(extra, dict) else []
    for call in calls or []:
        payload = arguments(call)
        if payload is not None:
            return payload
    return None


def _salvage_schema_invalid_rows(output_model, payload, error):
    """Quarantine invalid tool-call rows while retaining independently valid rows.

    Schema rejection happens before the normal numeric quarantine receives a
    Pydantic object.  Treat every raw row as an independent candidate so that a
    mistaken table-cell ``n=1`` cannot erase an otherwise valid arm-level row
    from the same tool call.  Rejected values stay in diagnostics and never
    become a quantitative record.
    """
    from pydantic import ValidationError

    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        return None
    issues = [str(item) for item in payload.get("unresolved_items", []) if isinstance(item, str)]
    rows = []
    for index, row in enumerate(payload["rows"]):
        try:
            validated = output_model.model_validate({"rows": [row], "unresolved_items": []})
        except ValidationError as row_error:
            issues.append("Schema-invalid extraction row omitted; source evidence requires review: " + json.dumps({
                "row_index": index,
                "validation_error": row_error.errors(include_url=False),
                "source_row": row,
            }, ensure_ascii=False))
        else:
            rows.extend(item.model_dump() for item in validated.rows)
    if not rows and not issues:
        issues.append("Structured output could not be parsed: " + str(error))
    return output_model.model_validate({"rows": rows, "unresolved_items": list(dict.fromkeys(issues))})


def _structured_family_extract(model, output_model, prompt, inputs, family: str):
    """Bounded schema repair, retaining the source and failed tool arguments.

    Transport exceptions propagate to the existing provider recovery. Only
    parsing/validation failures get correction turns; no source values are
    coerced or fabricated locally to satisfy a schema.
    """
    from langchain_core.messages import HumanMessage
    from langchain_core.exceptions import OutputParserException
    from pydantic import ValidationError

    messages = prompt.format_messages(**inputs)
    # Keep the raw tool call for schema-invalid responses.  It allows per-row
    # quarantine below; without it, one malformed row (for example a table
    # index reported as n=1) makes every otherwise valid row unrecoverable.
    chain = model.with_structured_output(output_model, method="function_calling", include_raw=True)
    last_error = None
    last_schema_valid = None
    last_rejected_arguments = ""
    for attempt in range(3):
        try:
            response = chain.invoke(messages)
            parsed = response
            if isinstance(response, dict) and ("parsed" in response or "parsing_error" in response):
                parsed = response.get("parsed")
                parsing_error = response.get("parsing_error")
                if parsed is None:
                    payload = _tool_arguments_from_raw(response.get("raw"))
                    salvaged = _salvage_schema_invalid_rows(output_model, payload, parsing_error)
                    if salvaged is not None:
                        last_schema_valid = salvaged
                        last_rejected_arguments = json.dumps(payload, ensure_ascii=False)[:20000]
                    if isinstance(parsing_error, BaseException):
                        raise parsing_error
                    raise ValueError(str(parsing_error or "structured output returned no parsed value"))
            # Independent local validation remains mandatory even if an SDK or
            # third-party endpoint claims to have parsed the result.
            validated = output_model.model_validate(
                parsed.model_dump() if isinstance(parsed, output_model) else parsed
            )
            validated = _quarantine_numeric_rows(validated, inputs.get("sources", "[]"), empty_diagnostics_only=True)
            validated = _annotate_sd_provenance(validated, inputs.get("sources", "[]"))
            last_schema_valid = validated
            _validate_numeric_rows(validated, inputs.get("sources", "[]"))
            return validated
        except (ValidationError, OutputParserException, ValueError) as exc:
            # Transport/auth/quota exceptions belong to the outer retry and
            # provider recovery. Replaying them as schema repair multiplied
            # upstream timeouts and delayed checkpoint recovery.
            if _status_code(exc) is not None:
                raise
            last_error = exc
        if attempt == 2:
            if last_schema_valid is not None:
                return _quarantine_numeric_rows(last_schema_valid, inputs.get("sources", "[]"))
            raise last_error
        if hasattr(last_error, "errors"):
            details = [{"field": ".".join(map(str, row["loc"])),
                        "error": row["msg"], "type": row["type"]}
                       for row in last_error.errors()][:24]
            error_text = json.dumps(details, ensure_ascii=False)
        else:
            error_text = str(last_error)[:4000]
        # Do not append an AI tool call without its ToolMessage response. A
        # user repair message safely carries the validation error instead.
        previous = last_rejected_arguments
        messages.append(HumanMessage(content=(
            "The previous tool result failed local validation. Correct the result using "
            "the same source evidence. Preserve valid supported rows; never invent or "
            "change numeric evidence to meet a constraint. If a row does not fit this "
            "family, omit it and explain the source-specific limitation in unresolved_items.\n"
            f"Validation errors: {error_text}\nRejected arguments: {previous}"
        )))
    raise last_error


def _single_family_prompt(tools, task, sources, feedback: str, output_model, family: str):
    """Build one optional extraction request for one exact data family.

    Numeric families are deliberately isolated.  An empty ``rows`` array is a
    valid result when the source does not satisfy the selected template; the
    model must not fall back to ``generic_text_evidence`` merely to populate a
    result.
    """
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Transcribe only source-supported evidence matching the supplied single data-family schema. "
            "This call handles exactly {family}; other data families are handled in separate calls. "
            "The schema describes representable data, not a demand to produce rows. Return rows=[] when "
            "this template does not fit. Preserve relevant unrepresentable evidence in unresolved_items. "
            "Do not return generic_text_evidence, invent sample sizes or SDs, treat individual patients "
            "as study arms, or convert ranges, thresholds, qualitative statements, and cited secondary "
            "studies into cohort summaries. Keep distinct source-supported outcomes, subgroups, timepoints, "
            "comparisons and models in distinct rows. Do not suppress otherwise relevant reported evidence "
            "because its model differs from a preferred model; downstream analysis decides eligibility. "
            "A source marked context_only provides definitions, denominators and interpretation; extract "
            "rows only from primary sources and use a primary source_id. Do not borrow a denominator "
            "across populations, timepoints or risk sets. Preserve source-specific uncertainty in unresolved_items. "
            "Follow target.source_constraints for numerical provenance and derivation limits.\n"
            "Family-specific transcription rules:\n{family_instructions}"
        ),
        (
            "user",
            "Task contract:\n{contract}\n"
            "The exact response schema is attached through the API structured-output contract.\n"
            "Source evidence:\n{sources}\nPrevious feedback:\n{feedback}"
        ),
    ])
    inputs = {
        "family": family,
        "family_instructions": _family_transcription_rules(tools, family),
        "contract": json.dumps(_single_family_contract(tools, task, family), ensure_ascii=False),
        "sources": json.dumps(_compact_extraction_sources(sources), ensure_ascii=False),
        "feedback": feedback or "No prior extraction attempt.",
    }
    return prompt, inputs


def _cached_family_extract(model, output_model, prompt, inputs, family):
    directory = _FAMILY_CHECKPOINT_DIR.get()
    path = directory / (family + ".json") if directory is not None else None
    fingerprint = digest({"version": EXTRACTION_CONTRACT_VERSION,
                          "schema": output_model.model_json_schema(), "inputs": inputs})
    if path is not None and path.is_file():
        saved = read(path)
        if saved.get("input_fingerprint") == fingerprint:
            return output_model.model_validate(saved["extraction"])
        # A failed article can resume with an updated pending-batch prompt or a
        # reviewed recovery patch.  The completed result for just this family
        # is no longer evidence for the new exact input, but it must not make
        # the whole article unrecoverable.  Preserve it for audit and rerun
        # only this family against the current sources/schema.
        prior_fingerprint = str(saved.get("input_fingerprint") or "unknown")
        superseded = path.parent / "superseded" / f"{family}.{prior_fingerprint[:16]}.json"
        if not superseded.is_file():
            superseded.parent.mkdir(parents=True, exist_ok=True)
            write(superseded, saved)
    parsed = _structured_family_extract(model, output_model, prompt, inputs, family)
    parsed = _validate_exact_family(parsed, family)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, {"schema_version": "trace/extraction_family/1.0",
                     "input_fingerprint": fingerprint, "family": family,
                     "status": "completed", "extraction": parsed.model_dump()})
    return parsed


def _validate_exact_family(extraction, family: str):
    unexpected = sorted({row.data_family for row in extraction.rows if row.data_family != family})
    require(not unexpected, "single-family extractor returned another data family: " + ", ".join(unexpected))
    return extraction


def _compact_extractor(tools, task, response_model, sources, feedback: str, is_typed: bool,
                       provider: str, template_families: set[str] | None = None):
    """Run compact routed extraction, recovering with frozen schemas if needed."""
    if not _uses_family_templates(task, tools, is_typed):
        _, prompt, inputs, _ = _compact_extraction_prompt(tools, task, sources, feedback, is_typed, template_families)
        model = tools._llm(task, "extractor", provider) if is_typed else tools._llm(task, "extractor")
        return (prompt | model.with_structured_output(response_model)).invoke(inputs)
    families = _template_families(task, tools)
    selected = set(template_families or families)
    require(selected and selected.issubset(families), "invalid routed template family")

    # Every routed family gets an independent optional call.  This prevents a
    # multi-schema prompt from steering the model toward the generic textual
    # fallback when one numeric schema is inapplicable.  Every route uses the
    # exact Pydantic model through the API's structured-output/tool interface.
    # An intermediary that cannot implement function calling fails visibly and
    # is reported on the configured provider instead of silently
    # weakening the response contract to free-form JSON.
    selected = set(template_families or families)
    model = tools._llm(task, "extractor", provider)
    rows = []
    unresolved = []
    for family in sorted(selected):
        output_model = _single_family_output_model(tools, family, _family_target_ids(task, tools, family))
        family_prompt, family_inputs = _single_family_prompt(
            tools, task, sources, feedback, output_model, family
        )
        routed = _cached_family_extract(
            model, output_model, family_prompt, family_inputs, family
        )
        routed = _validate_exact_family(routed, family)
        rows.extend(row.model_dump() for row in routed.rows)
        unresolved.extend(routed.unresolved_items)
    validated = response_model.model_validate({
        "rows": rows,
        "unresolved_items": list(dict.fromkeys(unresolved)),
    })
    return tools._validate_template_choice(validated, families)


def _call_extractor(tools, task, response_model, sources, feedback: str, is_typed: bool, provider: str,
                    template_families: set[str] | None = None):
    # Preserve support for minimal in-process adapters used by integration
    # tests and third-party runtimes that expose only the frozen public tool.
    with _provider_context(tools, provider):
        if not hasattr(tools, "_llm"):
            if is_typed:
                return tools.extract_from_locations(task, sources, feedback, provider)
            return tools.extract_from_locations(task, sources, feedback)
        if template_families is None:
            return _compact_extractor(tools, task, response_model, sources, feedback, is_typed, provider)
        return _compact_extractor(
            tools, task, response_model, sources, feedback, is_typed, provider, template_families
        )


def _response_model(task, tools, models):
    families = {
        family
        for target in task.raw.get("extraction", {}).get("targets", [])
        for family in target.get("accepted_data_families", [])
    }
    legacy = set(getattr(tools, "LEGACY_EFFECT_FAMILIES", {"reported_ratio_effect", "adjusted_ratio_effect", "time_to_event"}))
    return models.EffectExtraction if families and families.issubset(legacy) else models.DataFamilyExtraction


def _source_fingerprint(catalog: list[dict[str, Any]]) -> str:
    return digest([
        {
            "source_id": source["source_id"],
            "source_type": source.get("source_type"),
            "page": source.get("page"),
            "content": source.get("content", ""),
        }
        for source in catalog
    ])


def _batch_metrics(selected: list[dict[str, Any]], *, task=None, tools=None,
                   is_typed: bool = False, template_families: set[str] | None = None) -> tuple[int, int]:
    """Return source text and actual model-message character counts.

    ``source_chars`` is useful audit information but is not a safe request
    limit: the extraction contract, schema, prompt instructions, and JSON
    framing add substantial fixed and per-source text. When a frozen runtime
    is available, count the formatted messages that will actually be sent.
    The fallback keeps the small public-adapter test/runtime surface working.
    """
    source_chars = sum(len(str(source.get("content", ""))) for source in selected)
    if tools is None or task is None or not hasattr(tools, "_contract"):
        return source_chars, source_chars
    if template_families is None:
        return source_chars, _compact_prompt_chars(tools, task, selected, is_typed)
    return source_chars, _compact_prompt_chars(tools, task, selected, is_typed, template_families)


_DESIGN_CONTEXT_RE = re.compile(
    r"\b(denominator|follow[ -]?up|enrol(?:led|lment)|sample size|"
    r"randomi[sz](?:e[ds]?|ation)?|allocat(?:e[ds]?|ion)|assign(?:ed|ment)?|"
    r"1\s*:\s*1|per (?:arm|group)|each (?:surgeon|investigator|site)|"
    r"included patients|participants|missing data|footnote)\b",
    re.I,
)


def _design_context_sources(selected, primary_ids, *, allowance=4000):
    """Return whole source blocks that define denominators or study arms.

    These blocks are required interpretation context for numeric tables.  They
    are selected before batch packing so their serialized size is part of the
    request budget instead of depending on whatever space happens to remain.
    """
    candidates = []
    for position, source in enumerate(selected):
        content = str(source.get("content", ""))
        if (source["source_id"] in primary_ids or source.get("source_type") == "table"
                or not _DESIGN_CONTEXT_RE.search(content)):
            continue
        # Explicit allocation and arm-size statements are more useful than a
        # generic mention of participants when the allowance cannot hold all
        # matching blocks. Preserve document order within the same priority.
        priority = 0 if re.search(
            r"\b(sample size|randomi[sz](?:e[ds]?|ation)?|allocat(?:e[ds]?|ion)|"
            r"assign(?:ed|ment)?|1\s*:\s*1|per (?:arm|group)|each (?:surgeon|investigator|site))\b",
            content, re.I,
        ) else 1
        candidates.append((priority, position, source))
    contexts = []
    context_chars = 0
    for _, _, source in sorted(candidates):
        size = len(str(source.get("content", "")))
        if context_chars + size <= allowance:
            contexts.append(source)
            context_chars += size
    return contexts


def _attach_batch_context(batches, selected, budget, *, task=None, tools=None,
                          is_typed=False, template_families=None):
    """Repeat bounded whole interpretation blocks without extracting them twice.

    Primary source coverage remains exactly once. Nearby footnotes and compact
    population/follow-up definitions may accompany another batch if both the
    4,000-character context allowance and total request budget permit them.
    No source is cut and every repeated block keeps its original ID.
    """
    if len(batches) < 2:
        for batch in batches:
            batch["context_source_ids"] = []
        return batches
    by_id = {source["source_id"]: source for source in selected}
    position = {source["source_id"]: i for i, source in enumerate(selected)}
    for batch in batches:
        primary_ids = set(batch["source_ids"])
        positions = [position[sid] for sid in primary_ids]
        adjacent = [i for i in (min(positions)-1, max(positions)+1) if 0 <= i < len(selected)]
        interpretation = _design_context_sources(selected, primary_ids)
        contexts = []
        context_chars = 0
        primary = [by_id[sid] for sid in batch["source_ids"]]
        nearby = [selected[i] for i in adjacent]
        # Denominators and allocation come first. Nearby blocks remain useful
        # optional context, but cannot displace the study-design information.
        for source in [*interpretation, *nearby]:
            if source["source_id"] in primary_ids or source in contexts:
                continue
            size = len(str(source.get("content", "")))
            if context_chars + size > 4000:
                continue
            proposed = [*primary, *({**row, "context_only": True} for row in contexts), {**source, "context_only": True}]
            source_chars, prompt_chars = _batch_metrics(
                proposed, task=task, tools=tools, is_typed=is_typed, template_families=template_families)
            if prompt_chars + batch["schema_chars"] > budget:
                continue
            contexts.append(source)
            context_chars += size
            batch.update(source_chars=source_chars, prompt_chars=prompt_chars,
                         request_chars=prompt_chars+batch["schema_chars"])
        batch["context_source_ids"] = [source["source_id"] for source in contexts]
        batch["context_chars"] = context_chars
    return batches


def _attach_extraction_diagnostics(predictions, plan, extractions) -> None:
    """Carry batch caveats to the reviewer without treating them as row verdicts."""
    by_source = {}
    for batch, extraction in zip(plan, extractions):
        if not extraction.unresolved_items:
            continue
        diagnostic = {
            "stage": "extraction",
            "scope": "batch_context_not_row_verdict",
            "batch_id": batch["batch_id"],
            "source_ids": list(batch["source_ids"]),
            "notes": list(extraction.unresolved_items),
        }
        for sid in batch["source_ids"]:
            by_source.setdefault(sid, []).append(diagnostic)
    for prediction in predictions:
        sid = (prediction.get("source_location") or {}).get("source_id")
        if sid in by_source:
            prior = prediction.get("issues") or prediction.get("organizer_issues") or []
            if not isinstance(prior, list):
                prior = [prior]
            prediction["issues"] = [*prior, *deepcopy(by_source[sid])]


def _attach_prediction_sources(predictions, catalog, plan, work: Path) -> None:
    """Give downstream reviewers actual source blocks rather than quotes alone."""
    by_id = {source["source_id"]: source for source in catalog}
    context_by_id = {sid: batch.get("context_source_ids", [])
                     for batch in plan for sid in batch["source_ids"]}
    evidence_path = work / "source_evidence.json"
    write(evidence_path, {"schema_version": "trace/source_evidence/1.0",
                         "sources": _compact_extraction_sources(catalog)})
    for prediction in predictions:
        location = prediction.get("source_location")
        if not isinstance(location, dict):
            continue
        sid = location.get("source_id")
        if sid not in by_id:
            continue  # Existing organizer issues retain unresolved source IDs.
        evidence = []
        for source_id in dict.fromkeys([sid, *context_by_id.get(sid, [])]):
            source = by_id[source_id]
            evidence.append({"source_id": source_id, "source_type": source.get("source_type"),
                             "text": source.get("content", ""), "context_only": source_id != sid})
        location["source_evidence"] = evidence
        location["source_evidence_file"] = str(evidence_path)


def _batch_plan(selected: list[dict[str, Any]], budget: int, *, task=None,
                tools=None, is_typed: bool = False, template_families: set[str] | None = None) -> list[dict[str, Any]]:
    """Keep a source atomic and bound each full serialized model request.

    A single large source is deliberately never split because the frozen
    organizer relies on source-level evidence boundaries. Such a source is
    flagged as oversized and can still be sent alone; otherwise batches are
    cut before their *formatted prompt* exceeds the configured budget.
    """
    result: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    schema_chars = _request_schema_chars(task, tools, is_typed, template_families) if tools is not None and task is not None else 0
    def request_metrics(primary):
        primary_ids = {row["source_id"] for row in primary}
        contexts = _design_context_sources(selected, primary_ids)
        model_sources = [*primary, *({**row, "context_only": True} for row in contexts)]
        return _batch_metrics(
            model_sources, task=task, tools=tools, is_typed=is_typed,
            template_families=template_families,
        )

    for source in selected:
        proposed = [*current, source]
        _, proposed_prompt_chars = request_metrics(proposed)
        if current and proposed_prompt_chars + schema_chars > budget:
            source_chars, prompt_chars = _batch_metrics(
                current, task=task, tools=tools, is_typed=is_typed, template_families=template_families
            )
            result.append({
                "source_ids": [row["source_id"] for row in current],
                "source_chars": source_chars,
                "prompt_chars": prompt_chars,
            })
            current = []
        current.append(source)
    if current:
        source_chars, prompt_chars = _batch_metrics(
            current, task=task, tools=tools, is_typed=is_typed, template_families=template_families
        )
        result.append({
            "source_ids": [row["source_id"] for row in current],
            "source_chars": source_chars,
            "prompt_chars": prompt_chars,
        })
    for index, batch in enumerate(result, start=1):
        batch["batch_id"] = f"batch_{index:03d}"
        batch["schema_chars"] = schema_chars
        batch["request_chars"] = batch["prompt_chars"] + schema_chars
        batch["oversized_source_ids"] = batch["source_ids"] if batch["request_chars"] > budget else []
    return _attach_batch_context(result, selected, budget, task=task, tools=tools,
                                 is_typed=is_typed, template_families=template_families)


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_now()
    write(path, state)


def _resume_batch_plan(selected, budget, state, work, *, task=None, tools=None,
                       is_typed: bool = False, template_families: set[str] | None = None):
    """Split pending batches only; completed batch identities remain immutable."""
    by_id = {s['source_id']: s for s in selected}
    old = state['batches']
    require([sid for b in old for sid in b['source_ids']] == [s['source_id'] for s in selected],
            'extraction checkpoint source coverage/order changed')
    updated = []
    for batch in old:
        output = work / 'batches' / (batch['batch_id'] + '.json')
        batch_sources = [by_id[sid] for sid in batch['source_ids']]
        prior_context = [by_id[sid] for sid in batch.get('context_source_ids', [])]
        _, prompt_chars = _batch_metrics(
            batch_sources + [{**row, "context_only": True} for row in prior_context],
            task=task, tools=tools, is_typed=is_typed, template_families=template_families
        )
        if output.is_file():
            updated.append(batch)
            continue
        schema_chars = _request_schema_chars(task, tools, is_typed, template_families) if tools is not None and task is not None else 0
        if prompt_chars + schema_chars <= budget:
            updated.append({**batch, 'prompt_chars': prompt_chars, 'schema_chars': schema_chars,
                            'request_chars': prompt_chars + schema_chars, 'oversized_source_ids': []})
            continue
        children = _batch_plan(batch_sources, budget, task=task, tools=tools,
                               is_typed=is_typed, template_families=template_families)
        if len(children) == 1:
            updated.append(batch)  # Never cut a single source/table in half.
            continue
        state.setdefault('batch_plan_history', []).append(deepcopy(batch))
        for index, child in enumerate(children, 1):
            updated.append({**child, 'batch_id': batch['batch_id'] + f'_part{index:03d}',
                            'status': 'pending', 'attempts': []})
    return updated


def _load_or_prepare_state(task, controller, tools, article_id: str, work: Path, is_typed: bool):
    document = task.documents[article_id]
    print(f"[{article_id}] inspect_document", flush=True)
    inspection = controller.inspect_document(document)
    print(f"[{article_id}] static_clean_document", flush=True)
    catalog, cleanup_audit = _call_static_clean(tools, task, document, is_typed)
    require(bool(catalog), f"Static cleanup retained no source regions for {article_id}")
    source_fingerprint = _source_fingerprint(catalog)
    semantic_fingerprint = _semantic_fingerprint(task, tools, is_typed)
    state_path = work / "resilient_article_state.json"
    if state_path.is_file():
        state = read(state_path)
        require(state.get("schema_version") == STATE_SCHEMA, f"{article_id}: invalid article extraction checkpoint")
        require(state.get("article_id") == article_id, f"{article_id}: checkpoint article mismatch")
        require(state.get("source_fingerprint") == source_fingerprint, f"{article_id}: source content changed; use a new run")
        require(state.get("semantic_fingerprint") == semantic_fingerprint,
                f"{article_id}: extraction task/schema changed; use a new extraction output")
        return catalog, cleanup_audit, inspection, state, state_path
    state = {
        "schema_version": STATE_SCHEMA,
        "article_id": article_id,
        "created_at": _utc_now(),
        "source_fingerprint": source_fingerprint,
        "semantic_fingerprint": semantic_fingerprint,
        "inspection": inspection,
        "cleanup_audit": cleanup_audit,
        "location": None,
        "locator_issues": [],
        "batches": [],
    }
    _save_state(state_path, state)
    return catalog, cleanup_audit, inspection, state, state_path


def _retry_call(call: Callable[[], Any], *, article_id: str, stage: str, request: dict,
                on_failure: Callable[[int, Exception], None]) -> Any:
    """Retry the selected provider; preserve failures for explicit resume."""
    attempts = _call_attempts(request)
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            last_error = exc
            on_failure(attempt, exc)
            # Upstream outages return to the scheduler for a later explicit resume.
            if is_provider_outage_error(exc) or is_capacity_error(exc) or attempt == attempts or not is_retryable_provider_error(exc):
                break
            delay = retry_delay_seconds(exc, attempt)
            print(
                f"[extraction-resilient] {article_id} {stage} retry {attempt + 1}/{attempts} "
                f"after {type(exc).__name__}; waiting {delay:g}s",
                flush=True,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def _locate(task, tools, catalog, state, state_path: Path, article_id: str, request: dict, is_typed: bool, provider: str):
    by_id = {source["source_id"]: source for source in catalog}
    saved = state.get("location")
    if saved and saved.get("status") == "completed":
        ids = saved.get("source_ids", [])
        require(isinstance(ids, list) and all(isinstance(item, str) and item in by_id for item in ids),
                f"{article_id}: invalid locator checkpoint")
        return [by_id[item] for item in ids]
    state["location"] = {"status": "running", "attempts": []}
    _save_state(state_path, state)

    def record_failure(attempt: int, exc: Exception):
        state["location"]["attempts"].append({"attempt": attempt, "status": "failed", **error_info(exc)})
        _save_state(state_path, state)

    print(f"[{article_id}] locate_relevant_sources", flush=True)
    locations, selected, issues = _retry_call(
        lambda: _call_locator(tools, task, catalog, is_typed, provider),
        article_id=article_id,
        stage="locator",
        request=request,
        on_failure=record_failure,
    )
    if is_typed:
        selected, added = _include_explicit_mortality_tables(task, catalog, selected)
        if added:
            issues.append({"stage": "locate", "issues": ["explicit_mortality_table_retained"], "source_ids": added})
    state["location"] = {
        "status": "completed",
        "source_ids": [source["source_id"] for source in selected],
        "unresolved_location_terms": list(locations.unresolved_location_terms),
        "attempts": state["location"].get("attempts", []),
    }
    state["locator_issues"] = issues
    _save_state(state_path, state)
    return selected


def _include_explicit_mortality_tables(task, catalog, selected):
    """Keep arm-level death tables even when a locator chooses only prose.

    Some PDFs put outcome tables after the bibliography; their preview can be
    omitted by an LLM locator despite the Typed cleaner retaining the table.
    This only adds tables with an explicit death row and arm denominators.
    """
    targets = (getattr(task, "raw", {}) or {}).get("extraction", {}).get("targets", [])
    wants_mortality = any(
        any(token in str(target.get("outcome", "")).casefold() for token in ("mortality", "death"))
        and "binary_arm_counts" in target.get("accepted_data_families", [])
        for target in targets if isinstance(target, dict)
    )
    if not wants_mortality:
        return selected, []
    seen = {source["source_id"] for source in selected}
    added = []
    for source in catalog:
        if len(added) >= 3:
            break
        content = str(source.get("content") or "")
        if source.get("source_type") != "table" or source["source_id"] in seen:
            continue
        if not re.search(r"<tr[^>]*>\s*<td[^>]*>\s*(?:death|mortality)\b", content, re.I):
            continue
        if len(re.findall(r"\bn\s*=\s*\d+", content, re.I)) < 2:
            continue
        selected.append(source)
        seen.add(source["source_id"])
        added.append(source["source_id"])
    return selected, added


def _load_batch_result(path: Path, model):
    payload = read(path)
    require(payload.get("schema_version") == BATCH_SCHEMA, f"invalid batch checkpoint: {path}")
    require(payload.get("status") == "completed", f"incomplete batch checkpoint: {path}")
    return model.model_validate(payload["extraction"])


def _extract_batch(task, tools, model, sources, batch, state, state_path: Path, work: Path,
                   article_id: str, request: dict, is_typed: bool, provider: str,
                   template_families: set[str] | None = None):
    batch_id = batch["batch_id"]
    output = work / "batches" / f"{batch_id}.json"
    if output.is_file():
        saved = read(output)
        require(saved.get('article_id') == article_id and saved.get('batch_id') == batch_id
                and saved.get('source_ids') == batch['source_ids']
                and saved.get('source_chars') == batch['source_chars']
                and saved.get('context_source_ids', []) == batch.get('context_source_ids', [])
                and saved.get('semantic_fingerprint') == state.get('semantic_fingerprint'),
                f'{article_id}: completed batch identity/source mismatch: {batch_id}')
        print(f"[extraction-resilient] {article_id} {batch_id}: resume completed batch", flush=True)
        return _load_batch_result(output, model)
    record = next(item for item in state["batches"] if item["batch_id"] == batch_id)
    record["status"] = "running"
    _save_state(state_path, state)
    feedback = ""
    semantic_attempts = 1  # Empty rows are a valid optional-family result, not a retry trigger.
    response = None
    call_attempts = []
    for semantic_attempt in range(1, semantic_attempts + 1):
        print(
            f"[{article_id}] extract batch={batch_id} sources={len(sources)} source_chars={batch['source_chars']} "
            f"prompt_chars={batch.get('prompt_chars', batch['source_chars'])} "
            f"schema_chars={batch.get('schema_chars', 0)} request_chars={batch.get('request_chars', batch.get('prompt_chars', batch['source_chars']))} "
            f"attempt={semantic_attempt}",
            flush=True,
        )

        def record_failure(attempt: int, exc: Exception):
            item = {"semantic_attempt": semantic_attempt, "transport_attempt": attempt, "status": "failed", **error_info(exc)}
            call_attempts.append(item)
            record["attempts"] = call_attempts
            record["last_error"] = item
            _save_state(state_path, state)

        family_token = _FAMILY_CHECKPOINT_DIR.set(work / "batches" / batch_id / "families")
        try:
            response = _retry_call(
                lambda: _call_extractor(tools, task, model, sources, feedback, is_typed, provider, template_families),
                article_id=article_id,
                stage=batch_id,
                request=request,
                on_failure=record_failure,
            )
        finally:
            _FAMILY_CHECKPOINT_DIR.reset(family_token)
        call_attempts.append({"semantic_attempt": semantic_attempt, "transport_attempt": 1, "status": "completed"})
        if response.rows:
            break
        feedback = "; ".join(response.unresolved_items) or (
            "No rows were returned. Reinspect every supplied source region and transcribe all source-supported candidates."
        )
    require(response is not None, f"{article_id}: no extraction response")
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, {
        "schema_version": BATCH_SCHEMA,
        "article_id": article_id,
        "batch_id": batch_id,
        "source_ids": batch["source_ids"],
        "context_source_ids": batch.get("context_source_ids", []),
        "semantic_fingerprint": state.get("semantic_fingerprint"),
        "schema_chars": batch.get("schema_chars", 0),
        "request_chars": batch.get("request_chars", batch.get("prompt_chars", batch["source_chars"])),
        "source_chars": batch["source_chars"],
        "prompt_chars": batch.get("prompt_chars", batch["source_chars"]),
        "status": "completed",
        "completed_at": _utc_now(),
        "attempts": call_attempts,
        "extraction": response.model_dump(),
    })
    record.update({"status": "completed", "output": str(output.relative_to(work)), "attempts": call_attempts})
    record.pop("last_error", None)
    _save_state(state_path, state)
    return response


def run_article(request: dict, task, article_id: str, work: Path, runtime=None) -> dict[str, Any]:
    """Run one article with locator/batch checkpoints, without changing frozen code."""
    _, controller, tools, models = runtime or _load_runtime(request)
    is_typed = request["options"].get("extraction_runtime") == "typed"
    provider = request["options"].get("provider", "official_openai")
    # TEMPORARY DEVELOPMENT ONLY — configured by provider.py and intentionally
    # absent from request JSON and all run artifacts.
    work.mkdir(parents=True, exist_ok=True)
    catalog, cleanup_audit, inspection, state, state_path = _load_or_prepare_state(
        task, controller, tools, article_id, work, is_typed
    )
    selected = _locate(task, tools, catalog, state, state_path, article_id, request, is_typed, provider)
    template_families = _route_template_families(selected, _template_families(task, tools)) if is_typed else None
    budget = _batch_budget()
    plan = _batch_plan(selected, budget, task=task, tools=tools, is_typed=is_typed,
                       template_families=template_families)
    if not state["batches"]:
        state["batches"] = [{**batch, "status": "pending", "attempts": []} for batch in plan]
        write(work / "batch_manifest.json", {
            "schema_version": "trace/extraction_batch_manifest/1.0",
            "article_id": article_id,
            "budget_chars": budget,
            "batches": plan,
        })
        _save_state(state_path, state)
    else:
        state['batches'] = _resume_batch_plan(
            selected, budget, state, work, task=task, tools=tools, is_typed=is_typed,
            template_families=template_families
        )
        # A resumed batch must reproduce the model-bound input exactly.  In
        # particular, interpretation-only context blocks participate in each
        # family checkpoint fingerprint; dropping them would falsely make a
        # completed family look stale and silently change its evidence basis.
        plan = [{key: value for key, value in batch.items() if key in
                 {'batch_id', 'source_ids', 'context_source_ids', 'source_chars',
                  'prompt_chars', 'schema_chars', 'request_chars',
                  'oversized_source_ids'}} for batch in state['batches']]
        _save_state(state_path, state)
        write(work / 'batch_manifest.json', {'schema_version':'trace/extraction_batch_manifest/1.0',
              'article_id':article_id, 'budget_chars':budget, 'batches':plan,
              'policy':'completed batches retained; only unfinished batches split at source boundaries'})

    by_id = {source["source_id"]: source for source in catalog}
    response_model = _response_model(task, tools, models)
    extractions = []
    for batch in plan:
        sources = [by_id[source_id] for source_id in batch["source_ids"]]
        sources.extend({**by_id[source_id], "context_only": True}
                       for source_id in batch.get("context_source_ids", []))
        extractions.append(_extract_batch(
            task, tools, response_model, sources, batch, state, state_path, work,
            article_id, request, is_typed, provider, template_families,
        ))

    locator_issues = list(state.get("locator_issues") or [])
    extraction_issues = []
    for batch, extraction in zip(plan, extractions):
        if not extraction.rows:
            extraction_issues.append({
                "stage": "extract",
                "batch_id": batch["batch_id"],
                "issues": ["no_effect_rows_extracted", *extraction.unresolved_items],
            })
        elif extraction.unresolved_items:
            extraction_issues.append({
                "stage": "extract",
                "batch_id": batch["batch_id"],
                "issues": ["partially_unresolved_extraction", *extraction.unresolved_items],
            })
    print(f"[{article_id}] organize_extractions", flush=True)
    if extractions:
        # The frozen public helper accepts one extraction and wraps it in a
        # list.  Each source batch already yields one response, so call the
        # same frozen deterministic normalizer with that response list.
        organizer = getattr(tools, "normalize_and_verify_effects", None)
        if organizer is None:
            require(len(extractions) == 1, f"{article_id}: batch organizer unavailable")
            predictions, verification_issues = tools.organize_extractions(task, extractions[0], catalog, article_id)
        else:
            predictions, verification_issues = organizer(task, extractions, catalog, article_id)
    else:
        predictions, verification_issues = [], []
    _attach_extraction_diagnostics(predictions, plan, extractions)
    _attach_prediction_sources(predictions, catalog, plan, work)
    all_issues = [*locator_issues, *extraction_issues, *verification_issues]
    if not predictions:
        all_issues.append({"stage": "organize", "issues": ["no_predictions_produced"]})
    state["status"] = "completed"
    state["prediction_count"] = len(predictions)
    _save_state(state_path, state)
    print(f"[{article_id}] completed predictions={len(predictions)} issues={len(all_issues)}", flush=True)
    manifest = {
        "case_id": task.case_id,
        "task_id": task.task_id,
        "article_id": article_id,
        "started_at": state.get("created_at"),
        "completed_at": _utc_now(),
        "mode": "locator_extractor_tools_resilient_batched",
        "workflow": ["static_clean_document", "locate_relevant_sources", "bounded_extract_batches", "organize_extractions"],
        "controller_policy": "Typed locator and deterministic organizer with typed per-family schemas, source context, numeric diagnostics, and input-bound batch checkpoints",
        "allowed_representations": task.allowed_representations,
        "included_article_ids": task.included_article_ids,
        "pico": task.pico,
        "inspection": inspection,
    }
    return {
        "manifest": manifest,
        "controller_state": state,
        "raw_system_output": {
            "cleanup_audit": cleanup_audit,
            "source_catalog": [{key: value for key, value in source.items() if key != "content"} for source in catalog],
            "locations": state.get("location"),
            "located_sources": [source["source_id"] for source in selected],
            "extraction_batches": [response.model_dump() for response in extractions],
            "batch_manifest": "batch_manifest.json",
        },
        "extraction_submission": {
            "case_id": task.case_id,
            "task_id": task.task_id,
            "task_type": "extraction",
            "submission_status": "scoreable" if predictions and not all_issues else "requires_human_review",
            "predictions": predictions,
        },
        "issues": all_issues,
    }
