"""Render an interactive audit dashboard for TRACE pipeline runs and saved artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
import time


def read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def compact(value, limit=180):
    value = " ".join(str(value or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def counts(raw):
    case, control = (raw or {}).get("case_counts"), (raw or {}).get("control_counts")
    if not isinstance(case, dict) or not isinstance(control, dict):
        return ""
    return "Cases: " + ", ".join(f"{k} {v}" for k, v in case.items()) + "; Controls: " + ", ".join(f"{k} {v}" for k, v in control.items())


def source_metadata(document):
    paths = document.get("paths") or {}
    markdown = paths.get("markdown")
    text = ""
    if markdown:
        try:
            text = Path(markdown).read_text(encoding="utf-8", errors="replace")[:8000]
        except OSError:
            pass
    document_id = document.get("document_id") or document.get("article_id", "")
    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", text, flags=re.I)
    doi = doi_match.group(0).rstrip(".,;)") if doi_match else ""
    supplied_title = document.get("citation") or document.get("title") or ""
    is_placeholder = supplied_title == document_id or str(supplied_title).lower().startswith("candidate ")
    title = supplied_title if supplied_title and not is_placeholder else ""
    if not title:
        ignored = {"original article", "research article", "review article", "review", "open access", "accepted manuscript"}
        lines = [" ".join(line.split()) for line in text.splitlines()]
        title = next((line for line in lines[:45] if len(line) > 28 and line.lower() not in ignored and not line.startswith("<!--")), document_id)
    title = re.sub(r"^#{1,6}\s*", "", str(title)).strip()
    input_kind = "Markdown (MinerU)" if markdown and "prepared_mineru" in str(markdown) else "Markdown" if markdown else "PDF" if paths.get("pdf") else "Not provided"
    return {"id": document_id, "title": compact(title, 220), "doi": doi, "input": input_kind, "path": markdown or paths.get("pdf", "")}


def effect_record(row):
    effect = row.get("used_effect") or {}
    return {"type": effect.get("effect_type", ""), "value": effect.get("value"), "ci_lower": effect.get("ci_lower"), "ci_upper": effect.get("ci_upper")}


def issue_inventory(run: Path):
    rows, by_evidence = [], {}
    for path in sorted((run / "02_extraction" / "articles").glob("*/attempt_*/issues.json")):
        article, attempt = path.parents[1].name, path.parent.name
        for record in read(path, []):
            if isinstance(record, str):
                record = {"issues": [record]}
            if not isinstance(record, dict):
                continue
            identity = record.get("prediction_id", "")
            messages = record.get("issues", [])
            if isinstance(messages, str):
                messages = [messages]
            messages = [
                compact(value, 360) for value in messages if value
                and value not in {"source_quote_not_exact", "source_quote_not_found_in_markdown"}
            ]
            if messages:
                rows.append({"article": article, "attempt": attempt, "evidence_id": identity, "issues": messages})
                if identity:
                    by_evidence.setdefault(identity, []).extend(messages)
    return rows, by_evidence


def domain_evidence(domain):
    raw = list(domain.get("evidence") or [])
    for item in domain.get("assessment_items") or []:
        raw.extend(item.get("evidence") or [])
    seen, result = set(), []
    for evidence in raw:
        if not isinstance(evidence, dict):
            continue
        quote = compact(evidence.get("quote"), 420)
        locator = evidence.get("locator") or {}
        key = (quote, json.dumps(locator, sort_keys=True, ensure_ascii=False))
        if quote and key not in seen:
            seen.add(key)
            result.append({"quote": quote, "locator": locator, "source_type": evidence.get("source_type", "")})
    return result


def stage_artifact_links(run: Path):
    """Return a small, stage-local list of useful artifacts for the UI."""
    groups = {
        "preparation": [("Document preparation record", "document_preparation.json")],
        "screening": [
            ("Screening result", "01_screening/effective_screening_results.json"),
            ("Raw screening result", "01_screening/screening_results.json"),
            ("Screening completion marker", "01_screening/screening_complete.json"),
        ],
        "screening_rob": [
            ("RoB assessment", "01_screening/risk_of_bias_assessments.json"),
            ("RoB routing", "01_screening/risk_of_bias_effective_routes.json"),
            ("RoB completion marker", "01_screening/risk_of_bias_complete.json"),
            ("RoB failure log", "01_screening/risk_of_bias_failures.json"),
            ("RoB retry instructions", "01_screening/risk_of_bias_retry.md"),
        ],
        "extraction": [
            ("Extraction result", "02_extraction/case_extraction_submission.json"),
            ("Extraction progress", "02_extraction/extraction_progress.json"),
            ("Extraction completion marker", "02_extraction/extraction_complete.json"),
        ],
        "review": [
            ("Evidence review", "03_review/candidate_evidence_review_structured.json"),
            ("Review summary", "03_review/review_summary.md"),
            ("Review completion marker", "03_review/review_complete.json"),
        ],
        "analysis_check": [
            ("Analysis task", "04_analysis_input/analysis_task.json"),
            ("Analysis profile", "04_analysis_input/analysis_profile_structured.json"),
        ],
        "analysis": [
            ("Analysis report", "05_analysis/analysis_report.md"),
            ("Analysis result JSON", "05_analysis/analysis_result.json"),
            ("Statistical quality audit", "05_analysis/statistics_quality_audit.json"),
            ("Extension summary", "05_analysis/extension_summary.json"),
        ],
        "audit": [
            ("Run manifest", "pipeline_manifest.json"),
            ("Normalized request", "request.normalized.json"),
            ("Handoff audit", "handoff_audit.json"),
            ("Recovery audit", "analysis_metadata_recovery_audit.json"),
        ],
    }
    result = {}
    for stage, candidates in groups.items():
        links = [{"label": label, "path": path} for label, path in candidates if (run / path).is_file()]
        if stage in {"screening", "screening_rob", "extraction", "review", "analysis_check", "analysis", "audit"}:
            log_name = {"screening": "screening", "screening_rob": "screening_rob", "extraction": "extraction", "review": "review", "analysis_check": "analysis_check", "analysis": "analysis"}.get(stage)
            if log_name:
                log_path = run / "logs" / f"{log_name}.log"
                if log_path.is_file():
                    links.append({"label": f"Log {log_name}", "path": log_path.relative_to(run).as_posix()})
        forest_paths = sorted((run / "05_analysis" / "forest_plots").glob("*")) if stage == "analysis" else []
        for path in forest_paths:
            if path.is_file():
                links.append({"label": f"Forest plot {path.suffix.lstrip('.').upper()}", "path": path.relative_to(run).as_posix()})
        if links:
            result[stage] = links
    return result


def data_for(run: Path):
    try:
        from .retrieval_screening_panel import context_for
    except ImportError:
        from retrieval_screening_panel import context_for
    run = Path(run)
    manifest = read(run / "pipeline_manifest.json", {})
    request = read(run / "request.normalized.json", {})
    screening = read(run / "01_screening" / "effective_screening_results.json", {})
    if not screening.get("decisions"):
        screening = read(run / "01_screening" / "screening_results.json", {"decisions": []})
    screening_rob = read(run / "01_screening" / "risk_of_bias_assessments.json", {"assessments": []})
    if not screening_rob.get("assessments"):
        screening_rob = read(run / "01_screening" / "risk_of_bias.progress.json", {"assessments": []})
    rob_routes = read(run / "01_screening" / "risk_of_bias_effective_routes.json", {"recommendations": []})
    submission = read(run / "02_extraction" / "case_extraction_submission.json", {"predictions": []})
    review = read(run / "03_review" / "candidate_evidence_review_structured.json", {})
    result = read(run / "05_analysis" / "analysis_result.json", {})
    preparation = read(run / "document_preparation.json", {})
    extensions = read(run / "05_analysis" / "extension_summary.json", {})
    facets = read(run / "05_analysis" / "runtime_facets.json", {})

    decision_rows = screening.get("decisions", [])
    article_sources = {x.get("document_id") or x.get("article_id"): source_metadata(x) for x in request.get("documents", [])}
    include_counts = {key: sum(x.get("final_status") == key for x in decision_rows) for key in ("Include", "Exclude", "Unsure")}
    issues, issues_by_evidence = issue_inventory(run)
    reviews = {x.get("prediction_id"): x.get("review", {}) for x in review.get("candidate_reviews", [])}

    extracted, by_id = [], {}
    for row in submission.get("predictions", []):
        identity = row.get("prediction_id", "")
        reviewed, source = reviews.get(identity, {}), row.get("source_location") or {}
        item = {
            "id": identity, "article": row.get("article_id", ""), "study_id": row.get("study_id", ""), "target_id": row.get("target_id", ""),
            "citation": article_sources.get(row.get("article_id", ""), {}), "population": compact(row.get("population_or_subgroup"), 180),
            "outcome": compact(row.get("outcome"), 180), "exposure": compact(row.get("exposure"), 180), "comparison": compact(row.get("comparator"), 180),
            "estimand": row.get("estimand", ""), "effect": effect_record(row), "counts": compact(counts(row.get("raw_data")), 220),
            "model": compact(row.get("model_name"), 150), "adjustment": compact(row.get("adjustment") or (row.get("raw_data") or {}).get("adjustment_set"), 260),
            "p_value": row.get("p_value", ""), "timepoint": compact(row.get("timepoint"), 140), "source": source.get("source_id", ""),
            "source_quote": compact(source.get("source_quote"), 520), "locator": {key: source.get(key) for key in ("page", "page_index", "content_list_item_index", "bbox") if source.get(key) is not None},
            "verified": row.get("verification_status", ""), "review": reviewed.get("disposition", "pending"), "role": reviewed.get("evidence_role", ""),
            "reason": compact(reviewed.get("reason"), 260), "issues": issues_by_evidence.get(identity, []),
        }
        extracted.append(item); by_id[identity] = item

    route_by_article = {x.get("article_id"): x for x in rob_routes.get("recommendations", [])}
    rob_assessments = screening_rob.get("assessments", [])
    rob_counts = {key: sum(x.get("overall_judgment") == key for x in rob_assessments) for key in ("low", "moderate", "high", "unknown")}
    rob_rows = []
    for assessment in rob_assessments:
        route = route_by_article.get(assessment.get("article_id"), {})
        domains = [{"id": domain.get("domain_id") or domain.get("domain") or "", "name": domain.get("domain_name") or "", "judgment": domain.get("judgment", ""), "rationale": compact(domain.get("rationale"), 600), "item_count": len(domain.get("assessment_items") or []), "evidence": domain_evidence(domain)} for domain in assessment.get("domains") or []]
        rob_rows.append({
            "article": assessment.get("article_id"), "citation": article_sources.get(assessment.get("article_id"), {}), "target": (assessment.get("scope") or {}).get("target_id"),
            "tool": (assessment.get("tool") or {}).get("name"), "tool_version": (assessment.get("tool") or {}).get("version", ""),
            "judgment": assessment.get("overall_judgment"), "status": assessment.get("status"), "rationale": compact(assessment.get("overall_rationale"), 800), "domains": domains,
            "route_confidence": route.get("confidence", ""), "route_state": route.get("recommendation_state", ""), "route_flags": route.get("integrated_route_warning_flags") or route.get("validation_flags") or [],
        })

    analyses, alerts = [], []
    for analysis in result.get("analyses", []):
        assignments = []
        for row in analysis.get("assignments", []):
            enriched = dict(by_id.get(row.get("evidence_id"), {}))
            enriched.update({"id": row.get("evidence_id", ""), "role": row.get("role", ""), "selected": row.get("selected_within_context", False), "selection_reason": compact(row.get("reason"), 260)})
            assignments.append(enriched)
            if row.get("role") == "needs_review":
                alerts.append({"severity": "warning", "stage": "analysis_selection", "subject": row.get("evidence_id", ""),
                               "message": compact(row.get("reason"), 360)})
        statistics = dict(analysis.get("statistics") or {})
        definition = analysis.get("analysis") or {}
        analyses.append({
            "id": definition.get("analysis_id", ""), "title": definition.get("title") or definition.get("outcome_group") or definition.get("analysis_id", ""), "status": analysis.get("status", ""),
            "definition": {key: definition.get(key) for key in ("kind", "target_id", "outcome_group", "outcome_terms", "estimands", "eligible_data_families", "accepted_effect_types", "population_policy", "timepoint_policy", "model_policy", "dependence_policy", "minimum_studies")},
            "rules": definition.get("selection_rules", []), "inventory": analysis.get("candidate_inventory", {}), "assignments": assignments, "selected_ids": analysis.get("selected_evidence_ids", []),
            "poolability": (analysis.get("poolability") or {}).get("decisions", []), "study_effects": (analysis.get("poolability") or {}).get("study_effects", []),
            "statistics": statistics, "method": analysis.get("method_selection") or {}, "robustness": analysis.get("robustness") or {},
            "reporting_bias": analysis.get("reporting_bias") or {}, "warnings": analysis.get("warnings") or [],
        })

    pending_reviews = []
    for row in review.get("candidate_reviews", []):
        reviewed = row.get("review") or {}
        if reviewed.get("disposition") == "needs_further_review":
            pending = {"id": row.get("prediction_id"), "article": row.get("article_id"), "disposition": reviewed.get("disposition"), "role": reviewed.get("evidence_role"), "reason": reviewed.get("reason")}
            pending_reviews.append(pending)
            alerts.append({"severity": "warning", "stage": "review", "subject": row.get("prediction_id", ""), "message": compact(reviewed.get("reason"), 360)})

    completion = screening_rob.get("completion", {})
    if completion.get("status") in ("partial", "failed"):
        alerts.append({"severity": "warning", "stage": "risk_of_bias", "subject": "Incomplete RoB",
                       "message": f"RoB completed {completion.get('completed', 0)}/{completion.get('total', 0)}; tool-uncovered {completion.get('uncovered', 0)}; provider failures {completion.get('provider_failures', 0)}. Unassessed evidence is retained in the primary analysis."})
    validation = screening_rob.get("validation_summary") or {}
    draft_count = sum(item.get("status") == "draft" for item in rob_assessments)
    confirmed_count = sum(item.get("status") == "confirmed" for item in rob_assessments)
    unresolved_count = sum(item.get("status") not in ("draft", "confirmed") for item in rob_assessments)
    sensitivities = []
    for sensitivity in extensions.get("bias_sensitivities") or []:
        audit_path = sensitivity.get("audit_path", "")
        audit = read(run / "05_analysis" / audit_path, {}) if audit_path else {}
        records = audit.get("records") or []
        sensitivities.append({"id": sensitivity.get("sensitivity_id"), "status": sensitivity.get("status"), "excluded": sum(bool(x.get("excluded")) for x in records), "retained": sum(not bool(x.get("excluded")) for x in records), "policy": audit.get("policy") or {}})

    preparation_rows = preparation.get("documents") or []
    stages = manifest.get("stages", {})
    return {
        "at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), "run": str(run), "case": manifest.get("case_id", request.get("case_id", "unknown")), "status": manifest.get("status", "not_started"), "topic": request.get("topic", ""),
        "stages": [{"name": name, "status": stages.get(name, {}).get("status", "pending"), "at": stages.get(name, {}).get("completed_at") or stages.get(name, {}).get("paused_at") or stages.get(name, {}).get("started_at")} for name in ("screening", "screening_rob", "extraction", "review", "analysis_check", "analysis")],
        "sources": list(article_sources.values()),
        "retrieval_screening": context_for(run),
        "update": {"manifest": read(run / "update_manifest.json", {}),
                   "reuse": read(run / "update_reuse.json", {}),
                   "comparison": read(run / "update_comparison.json", {})},
        "screening": {"total": len(request.get("documents", [])), "counts": include_counts, "rows": [{"id": x.get("candidate_id"), "title": compact(x.get("title") or x.get("candidate_id"), 100), "status": x.get("final_status", "pending"), "mode": x.get("screening_stage", ""), "reason": compact(x.get("reason"), 300), "flags": x.get("validation_flags") or []} for x in decision_rows]},
        "rob": {"count": len(rob_assessments), "counts": rob_counts, "validation": validation, "rows": rob_rows,
                "completed_count": draft_count + confirmed_count, "draft_count": draft_count,
                "confirmed_count": confirmed_count, "unresolved_count": unresolved_count},
        "extraction": {"rows": extracted, "issues": issues, "review_counts": review.get("disposition_counts", {}), "reviewed": review.get("completed_review_count", 0), "failed": review.get("failed_review_count", 0), "pending_reviews": pending_reviews, "batching": review.get("batching") or {}},
        "analyses": analyses, "alerts": alerts, "sensitivity": sensitivities,
        "graph": {}, "brain": {},
        "facets": {"mode": facets.get("mode"), "status": facets.get("status"), "calls_made": facets.get("calls_made"), "annotation_count": len(facets.get("annotations") or []), "dimension_count": len(facets.get("dimensions") or []), "note": facets.get("note")},
        "preparation": {"policy": preparation.get("policy", ""), "counts": dict(Counter(row.get("status", "unknown") for row in preparation_rows)), "rows": preparation_rows},
        "providers": {"text": {}, "embedding": {}, "routing": {}},
        "recovery_history": manifest.get("recovery_history") or [], "artifacts": [], "stage_artifacts": stage_artifact_links(run),
        "forest": [x.relative_to(run).as_posix() for x in sorted((run / "05_analysis" / "forest_plots").glob("*.png"))],
    }


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">__REFRESH__<title>TRACE · End-to-end run</title><style>
:root{--bg:#f4f7fb;--paper:#fff;--ink:#14213d;--muted:#62708a;--line:#dce4f0;--blue:#2563eb;--green:#15803d;--red:#b91c1c;--amber:#b45309}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1480px;margin:auto;padding:26px 20px 52px}h1{font-size:26px;margin:0}h2{font-size:19px;margin:0 0 14px}h3{font-size:15px;margin:22px 0 8px}.top{display:flex;justify-content:space-between;gap:18px;margin-bottom:16px}.muted{color:var(--muted)}.stamp{text-align:right;font-size:12px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:14px 0 20px}.grid.six{grid-template-columns:repeat(6,minmax(0,1fr))}.card,.panel{background:var(--paper);border:1px solid var(--line);border-radius:11px;padding:15px;margin-bottom:14px}.value{font-size:22px;font-weight:700}.label{font-size:12px;color:var(--muted)}.timeline{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;margin-bottom:20px}.stage{background:var(--paper);border:1px solid var(--line);padding:10px;border-radius:8px}.stage strong{display:block}.tabs{display:flex;gap:5px;border-bottom:1px solid var(--line);margin-bottom:16px;overflow:auto}button{font:inherit;border:0;background:transparent;padding:10px 13px;color:var(--muted);cursor:pointer;white-space:nowrap}button[aria-selected=true]{color:var(--blue);border-bottom:2px solid var(--blue);font-weight:650}section[hidden]{display:none}table{width:100%;border-collapse:collapse}th,td{padding:9px 7px;vertical-align:top;text-align:left;border-bottom:1px solid var(--line)}th{font-size:12px;color:var(--muted);font-weight:650;position:sticky;top:0;background:var(--paper)}.table-wrap{overflow:auto;max-height:620px;border:1px solid var(--line);border-radius:8px}code{font-size:11px;word-break:break-all}.status{font-weight:650}.completed,.Include,.included,.keep_as_valid_candidate,.poolable,.low,.consistent{color:var(--green)}.Exclude,.failed,.excluded,.discard_clear_mismatch,.high,.error{color:var(--red)}.Unsure,.paused,.needs_further_review,.moderate,.unknown,.unclear,.warning{color:var(--amber)}.running{color:var(--blue)}.pending,.waiting,.draft{color:var(--muted)}.filter{width:100%;padding:9px 11px;border:1px solid var(--line);border-radius:7px;margin:0 0 10px;font:inherit}.pill{display:inline-block;background:#edf3ff;color:#1d4ed8;padding:2px 7px;border-radius:999px;font-size:12px;margin:2px}.pill.error{background:#fee2e2;color:#991b1b}.pill.warning{background:#fff7d6;color:#854d0e}.forest{max-width:100%;height:auto;border:1px solid var(--line);border-radius:9px;background:#fff}.note{background:#fff8e7;border:1px solid #f2d48d;border-radius:8px;padding:10px;color:#774d00;margin:8px 0}.alert{border-left:4px solid var(--amber);padding:9px 12px;background:#fff;margin:7px 0;border-radius:6px}.alert.error{border-color:var(--red);background:#fff5f5}.quote{border-left:3px solid var(--line);padding-left:9px;color:#44506a;margin:5px 0}details{border:1px solid var(--line);border-radius:7px;padding:8px;background:#fbfcfe;margin:6px 0}summary{cursor:pointer;font-weight:650}.artifacts a{display:inline-block;margin:4px 12px 4px 0;color:#1d4ed8}a{color:#1d4ed8}@media(max-width:900px){.grid,.grid.six{grid-template-columns:repeat(2,minmax(0,1fr))}.timeline{grid-template-columns:repeat(2,minmax(0,1fr))}.top{display:block}.stamp{text-align:left;margin-top:8px}}
</style></head><body><main id="app"></main><script id="data" type="application/json">__PAYLOAD__</script><script>
const d=JSON.parse(document.getElementById('data').textContent),app=document.getElementById('app');
const e=v=>String(v??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const n=v=>Number(v||0).toLocaleString('en-US'),f=v=>v===null||v===undefined||v===''?'—':Number.isFinite(Number(v))?Number(v).toLocaleString('en-US',{maximumFractionDigits:4}):e(v);
const s=v=>`<span class="status ${e(v)}">${e(v||'pending')}</span>`,pills=(xs,level='')=>(xs||[]).map(x=>`<span class="pill ${level}">${e(x)}</span>`).join('');
const table=(id,h,rows)=>`<input class="filter" data-filter="${id}" placeholder="Filter this table…"><div class="table-wrap"><table id="${id}"><thead><tr>${h.map(x=>`<th>${e(x)}</th>`).join('')}</tr></thead><tbody>${rows||`<tr><td colspan="${h.length}" class="muted">No data</td></tr>`}</tbody></table></div>`;
const effect=x=>{const q=x?.effect||x||{};return q.value===null||q.value===undefined?'—':`${e(q.type||q.effect_type||'')} ${f(q.value)}${q.ci_lower!==null&&q.ci_lower!==undefined?` [${f(q.ci_lower)}, ${f(q.ci_upper)}]`:''}`},paper=x=>{const c=x.citation||{};return `<strong>${e(c.title||x.article||'Unidentified article')}</strong><br><code>${e(c.id||x.article||'')}</code>${c.doi?`<br><a href="https://doi.org/${e(c.doi)}" target="_blank">DOI: ${e(c.doi)}</a>`:''}`};
const stages=d.stages.map(x=>`<div class="stage"><strong>${e(x.name)}</strong>${s(x.status)}<div class="muted">${e(x.at||'Not started')}</div></div>`).join(''),alerts=(d.alerts||[]).map(x=>`<div class="alert ${e(x.severity)}"><strong>${e(x.stage)} · ${e(x.subject)}</strong><div>${e(x.message)}</div></div>`).join('')||'<p class="muted">No automatically detected items require expert attention.</p>';
const screen=d.screening.rows.map(x=>`<tr><td><code>${e(x.id)}</code></td><td>${e(x.title)}</td><td>${s(x.status)}</td><td>${e(x.mode)}</td><td>${e(x.reason)}<br>${pills(x.flags,'warning')}</td></tr>`).join(''),sources=d.sources.map(x=>`<tr><td><code>${e(x.id)}</code></td><td><strong>${e(x.title)}</strong>${x.doi?`<br><a href="https://doi.org/${e(x.doi)}" target="_blank">${e(x.doi)}</a>`:''}</td><td>${e(x.input)}</td><td><code>${e(x.path)}</code></td></tr>`).join('');
const rob=d.rob.rows.map(x=>{const domains=(x.domains||[]).map(z=>`<details><summary>${e(z.id)} ${e(z.name)} · ${s(z.judgment)}</summary><div>${e(z.rationale)}</div><div class="muted">Items ${n(z.item_count)} · Evidence locators ${n(z.evidence.length)}</div>${z.evidence.map(q=>`<div class="quote">${e(q.quote)}<br><small>${e(JSON.stringify(q.locator||{}))}</small></div>`).join('')}</details>`).join('');return `<tr><td>${paper(x)}</td><td><code>${e(x.target)}</code></td><td>${e(x.tool)} ${e(x.tool_version)}<br>${s(x.status)}</td><td>${s(x.judgment)}<div>${e(x.rationale)}</div></td><td>${domains}</td><td>${s(x.route_state)}<br>confidence=${e(x.route_confidence)}<br>${pills(x.route_flags,'warning')}</td></tr>`}).join('');
const extraction=d.extraction.rows.map(x=>`<tr><td><code>${e(x.id)}</code><br>${paper(x)}</td><td><code>${e(x.target_id)}</code><br>${e(x.outcome)}<br><span class="muted">${e(x.population)}</span></td><td>${e(x.exposure)}<br>${e(x.comparison)}<br><code>${e(x.estimand)}</code></td><td><strong>${effect(x)}</strong><br>p=${e(x.p_value||'—')}<br>${e(x.model)}<br><span class="muted">${e(x.adjustment)}</span></td><td>${s(x.review)}<br>${s(x.role)}<br>${e(x.reason)}</td><td>${e(x.source)}<br><code>${e(JSON.stringify(x.locator||{}))}</code><div class="quote">${e(x.source_quote)}</div></td><td>${pills(x.issues,'error')}</td></tr>`).join('');
const issueRows=(d.extraction.issues||[]).map(x=>`<tr><td><code>${e(x.article)}</code></td><td><code>${e(x.evidence_id||'article-level')}</code></td><td>${pills(x.issues,x.evidence_id?'error':'warning')}</td></tr>`).join(''),reviewRows=(d.extraction.pending_reviews||[]).map(x=>`<tr><td><code>${e(x.id)}</code></td><td><code>${e(x.article)}</code></td><td>${s(x.disposition)}</td><td>${e(x.role)}</td><td>${e(x.reason)}</td></tr>`).join('');
function analysisBlock(a,i){const st=a.statistics||{},m=a.method||{},ro=a.robustness||{},rb=a.reporting_bias||{},comp=ro.model_comparison||{},def=a.definition||{},inv=a.inventory||{};const assign=(a.assignments||[]).map(x=>`<tr><td><code>${e(x.id)}</code><br>${paper(x)}</td><td>${effect(x)}</td><td><code>${e(x.estimand)}</code><br>${e(x.comparison)}</td><td>${e(x.model)}<br><span class="muted">${e(x.adjustment)}</span></td><td>${s(x.role)}<br>${x.selected?'<strong>Selected</strong>':'Not selected'}</td><td>${e(x.selection_reason)}</td><td>${pills(x.issues,'error')}</td></tr>`).join(''),pool=(a.poolability||[]).map(x=>`<tr><td><code>${e(x.study_id)}</code></td><td><code>${e(x.evidence_id)}</code></td><td>${s(x.status)}</td><td>${e((x.reasons||[]).join('; '))}</td></tr>`).join(''),loo=(ro.leave_one_out||[]).map(x=>`<tr><td><code>${e(x.evidence_id||x.study_id)}</code></td><td>${f(x.pooled_effect||x.pooled_log_effect)}</td><td>[${f(x.ci_lower||x.ci_lower_log)}, ${f(x.ci_upper||x.ci_upper_log)}]</td><td>${f(x.i_squared)}%</td><td>${f(x.tau_squared)}</td></tr>`).join(''),drivers=(ro.heterogeneity_drivers||[]).map(x=>`<tr><td><code>${e(x.evidence_id||x.study_id)}</code></td><td>${f(x.heterogeneity_contribution)}</td><td>${f(x.influence_on_pooled_effect)}</td></tr>`).join('');const warnings=(a.warnings||[]).length?`<div class="note"><strong>Analysis warning: </strong>${e(a.warnings.join('; '))}</div>`:'';return `<div class="panel"><h2>${e(a.title||a.id)}</h2><div class="muted"><code>${e(a.id)}</code> · ${s(a.status)}</div><h3>Analysis contract &amp; candidate inventory</h3><div class="grid"><div class="card"><div class="label">Target / type</div><div class="value">${e(def.target_id||'—')}</div><div>${e(def.kind||'')} · ${e(def.outcome_group||'')}</div></div><div class="card"><div class="label">Candidate records / studies</div><div class="value">${n(inv.candidate_record_count)} / ${n(inv.candidate_study_count)}</div><div>${e(inv.embedding_retrieval_summary||'')}</div></div><div class="card"><div class="label">Effect / estimand</div><div>${e((def.accepted_effect_types||[]).join(', ')||'Not specified')}</div><div>${e((def.estimands||[]).join(', ')||'Not specified')}</div></div><div class="card"><div class="label">Selection policy</div><div>${e(def.population_policy||'')}</div><div>${e(def.timepoint_policy||'')} · ${e(def.model_policy||'')}</div></div></div><p><strong>Dependence policy: </strong>${e(def.dependence_policy||'Not declared')}; <strong>Minimum studies: </strong>${f(def.minimum_studies)}; <strong>Explicit selection rules: </strong>${e((a.rules||[]).join('; ')||'None')}</p>${warnings}<h3>Final selection &amp; in-scope candidates</h3>${table(`select-${i}`,['Evidence / article','Effect [95% CI]','Estimand / comparison','Model / adjustment','Role','Selection rationale','Issues'],assign)}<h3>Poolability review</h3>${table(`pool-${i}`,['Study','Evidence','Status','Reason'],pool)}<h3>Statistical results</h3><div class="grid six"><div class="card"><div class="label">Pooled effect</div><div class="value">${e(st.effect_type)} ${f(st.pooled_effect)}</div><div>[${f(st.ci_lower)}, ${f(st.ci_upper)}]</div></div><div class="card"><div class="label">Studies k</div><div class="value">${f(st.k)}</div></div><div class="card"><div class="label">Heterogeneity I²</div><div class="value">${f(st.i_squared)}%</div></div><div class="card"><div class="label">Q / p</div><div class="value">${f(st.q)}</div><div>p=${f(st.q_p_value)}</div></div><div class="card"><div class="label">tau²</div><div class="value">${f(st.tau_squared)}</div></div><div class="card"><div class="label">Analysis method</div><div class="value">${e(m.pooling_model||st.model||'—')}</div><div>${e(m.heterogeneity_estimator||'')} ${e(m.confidence_interval||'')}</div></div></div><div class="note"><strong>Method selection rationale: </strong>${e(m.rationale||'Not available')}</div><h3>Robustness diagnostics</h3><div class="grid"><div class="card"><div class="label">Stability</div><div class="value">${e(ro.stability||ro.status||'Not available')}</div></div><div class="card"><div class="label">Fixed effects</div><div class="value">${f(comp.fixed_effect_pooled_effect)}</div></div><div class="card"><div class="label">Random effects</div><div class="value">${f(comp.random_effects_pooled_effect)}</div></div><div class="card"><div class="label">Robustness I²</div><div class="value">${f(comp.random_effects_i_squared)}%</div></div></div><h3>Leave-one-out</h3>${table(`loo-${i}`,['Evidence removed','Pooled effect','Confidence interval','I²','tau²'],loo)}<h3>Heterogeneity contribution</h3>${table(`driver-${i}`,['Evidence','Heterogeneity contribution','Influence on pooled value'],drivers)}<h3>Publication bias</h3><p>${s(rb.status)} · ${e(rb.interpretation||'')}</p><p class="muted">${e((rb.caveats||[]).join('; '))}</p></div>`}
const analysis=(d.analyses||[]).map(analysisBlock).join('')||'<div class="panel">No analysis result yet.</div>',forests=d.forest.map(x=>`<figure><img class="forest" src="${e(x)}"><figcaption>${e(x)}</figcaption></figure>`).join('')||'<p class="muted">No forest plot generated yet.</p>';
const sensitivities=(d.sensitivity||[]).map(x=>`<tr><td><code>${e(x.id)}</code></td><td>${s(x.status)}</td><td>${n(x.excluded)}</td><td>${n(x.retained)}</td><td><code>${e(JSON.stringify(x.policy))}</code></td></tr>`).join('');
const prep=(d.preparation.rows||[]).map(x=>`<tr><td><code>${e(x.article_id)}</code></td><td>${s(x.status)}</td><td><code>${e(x.pdf||'')}</code></td><td><code>${e(x.markdown||'')}</code></td></tr>`).join('');
const recovery=(d.recovery_history||[]).map(x=>`<div class="note"><code>${e(x.event)}</code> · ${e(x.previous_error?.type)}: ${e(x.previous_error?.message)}</div>`).join('')||'<p class="muted">No recovery history.</p>',live=d.status==='running'?'Page refreshes every 8 seconds':'Run finished; automatic refresh is stopped';
app.innerHTML=`<div class="top"><div><h1>TRACE · End-to-end run</h1><div>${e(d.topic)}</div><div class="muted">Case ${e(d.case)} · ${s(d.status)} · ${live}</div></div><div class="stamp">Generated ${e(d.at)}<br>${e(d.run)}</div></div><div class="grid"><div class="card"><div class="label">Candidate papers</div><div class="value">${n(d.screening.total)}</div></div><div class="card"><div class="label">Included / Excluded</div><div class="value">${n(d.screening.counts.Include)} / ${n(d.screening.counts.Exclude)}</div></div><div class="card"><div class="label">RoB: High / Unknown</div><div class="value">${n(d.rob.counts.high)} / ${n(d.rob.counts.unknown)}</div></div><div class="card"><div class="label">Expert attention</div><div class="value">${n(d.alerts.length)}</div></div></div><div class="timeline">${stages}</div><div class="tabs" role="tablist"><button role="tab" aria-selected="true" aria-controls="overview">Expert overview</button><button role="tab" aria-selected="false" aria-controls="screening">Screening &amp; sources</button><button role="tab" aria-selected="false" aria-controls="rob">Risk of bias</button><button role="tab" aria-selected="false" aria-controls="extraction">Extraction &amp; review</button><button role="tab" aria-selected="false" aria-controls="analysis">Analysis &amp; forest plots</button><button role="tab" aria-selected="false" aria-controls="audit">Run audit</button></div><section class="panel" id="overview" role="tabpanel"><h2>Expert attention</h2>${alerts}<h3>Pending further review</h3>${table('pending-review',['Evidence','Article','Status','Role','Reason'],reviewRows)}<h3>High-RoB exclusion sensitivity</h3>${table('sensitivity',['Sensitivity','Status','Excluded','Retained','Rules'],sensitivities)}</section><section id="screening" role="tabpanel" hidden><div class="panel"><h2>Screening decision</h2>${table('screen-table',['Candidate','Title','Decision','Stage','Basis / flags'],screen)}</div><div class="panel"><h2>Literature sources</h2>${table('source-table',['Document ID','Title / DOI','Input','Local source'],sources)}</div></section><section class="panel" id="rob" role="tabpanel" hidden><h2>Screening-stage risk of bias</h2><div class="note">Automated assessments completed ${n(d.rob.completed_count)}/${n(d.rob.count)}; drafts ${n(d.rob.draft_count)}, manually confirmed ${n(d.rob.confirmed_count)}, unsupported or pending ${n(d.rob.unresolved_count)}.</div>${table('rob-table',['Article','Target','Tool / status','Overall judgment / rationale','Six domains &amp; evidence','Routing audit'],rob)}</section><section id="extraction" role="tabpanel" hidden><div class="panel"><h2>Extraction &amp; review</h2><p>Extracted ${n(d.extraction.rows.length)} records; review completed ${n(d.extraction.reviewed)}; failed ${n(d.extraction.failed)}.</p>${table('extract-table',['Evidence / article','Target / outcome / population','Exposure / comparison / estimand','Effect / model / adjustment','Review','Source quote / locator','Issues'],extraction)}</div><div class="panel"><h2>Extraction issues</h2>${table('issue-table',['Article','Evidence','Issues'],issueRows)}</div></section><section id="analysis" role="tabpanel" hidden>${analysis}<div class="panel"><h2>Forest plot</h2>${forests}</div></section><section id="audit" role="tabpanel" hidden><div class="panel"><h2>Document preparation</h2><p>Policy: ${e(d.preparation.policy||'Not available')} · ${e(JSON.stringify(d.preparation.counts||{}))}</p>${table('prep-table',['Article','Status','PDF','Markdown'],prep)}</div><div class="panel"><h2>Recovery history</h2>${recovery}</div></section>`;
document.querySelectorAll('[role=tab]').forEach(t=>t.addEventListener('click',()=>{document.querySelectorAll('[role=tab]').forEach(x=>x.setAttribute('aria-selected','false'));document.querySelectorAll('[role=tabpanel]').forEach(x=>x.hidden=true);t.setAttribute('aria-selected','true');document.getElementById(t.getAttribute('aria-controls')).hidden=false}));document.querySelectorAll('[data-filter]').forEach(i=>i.addEventListener('input',()=>{const q=i.value.toLowerCase();document.querySelectorAll(`#${i.dataset.filter} tbody tr`).forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))}));
const stageArtifacts=d.stage_artifacts||{};
const artifactPanel=(title,items)=>{if(!items||!items.length)return '';return `<div class="panel artifacts"><h3>${e(title)} artifacts</h3>${items.map(x=>`<a href="${e(x.path)}" target="_blank">${e(x.label)}</a>`).join('')}</div>`};
const stageTargets={preparation:'screening',screening:'screening',screening_rob:'rob',extraction:'extraction',review:'extraction',analysis_check:'analysis',analysis:'analysis',audit:'audit'},stageTitles={preparation:'Document preparation',screening:'Screening',screening_rob:'Screening RoB',extraction:'Extraction',review:'Review',analysis_check:'Analysis-check',analysis:'Analysis',audit:'Audit'};
Object.entries(stageTargets).forEach(([stage,target])=>{const section=document.getElementById(target);if(section){const block=artifactPanel(stageTitles[stage]||stage,stageArtifacts[stage]);if(block)section.insertAdjacentHTML('beforeend',block);}});
</script></body></html>'''


def update_panel(data):
    update = data.get("update") or {}
    manifest = update.get("manifest") or {}
    if not manifest:
        return ""
    plan = manifest.get("plan", {})
    comparison = update.get("comparison") or {}
    def text(value):
        return escape(str(value if value is not None else "—"))
    labels = ("Added", "Changed", "Removed", "Unchanged")
    doc_counts = " · ".join(f"{label}: {len(plan.get('documents', {}).get(key, []))}"
                            for label, key in zip(labels, ("added", "changed", "removed", "unchanged")))
    reused = ", ".join(f"{text(k)}: {v.get('count', 0)}" for k, v in update.get("reuse", {}).items())
    rows = []
    for row in comparison.get("analyses", []):
        b = row.get("before", {}).get("pooled_result") or {}
        a = row.get("after", {}).get("pooled_result") or {}
        def pair(key):
            return f"{text(b.get(key))} → {text(a.get(key))}"
        pop_b, pop_a = row.get("before_population", {}), row.get("after_population", {})
        rows.append(f"<tr><td>{text(row['analysis_id'])}</td>"
                    f"<td>{text(pop_b.get('selected_studies'))} → {text(pop_a.get('selected_studies'))}</td>"
                    f"<td>{text(pop_b.get('sample_size'))} → {text(pop_a.get('sample_size'))}</td>"
                    f"<td>{text(b.get('effect_type'))} {pair('pooled_effect')}</td>"
                    f"<td>[{text(b.get('ci_lower'))}, {text(b.get('ci_upper'))}] → [{text(a.get('ci_lower'))}, {text(a.get('ci_upper'))}]</td>"
                    f"<td>{pair('heterogeneity_i2')}</td></tr>")
    headers = ("Analysis", "Studies", "N", "Effect", "95% CI", "I²")
    table = ("<div class='table-wrap'><table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers)
             + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>") if rows else ""
    note = "Descriptive comparison; changed criteria or methods also affect results. — means unavailable."
    links = [("update_manifest.json", "Update plan"), ("update_reuse.json", "Reuse audit")]
    if comparison:
        links += [("update_comparison.json", "Comparison data"), ("update_report.md", "Comparison report")]
    return (f"<div class='panel'><h2>Incremental update · Generation {text(manifest.get('generation'))}</h2>"
            f"<p>Parent run: {text(manifest.get('parent_run'))}</p>"
            f"<p>{doc_counts}</p><p>Reused records: {reused or '—'}</p>"
            f"{table}<p class='muted'>{note}</p><div class='artifacts'>" +
            "".join(f'<a href="{path}" target="_blank">{label}</a> ' for path, label in links) + "</div></div>")


def render(data, language="en"):
    refresh = '<meta http-equiv="refresh" content="8">' if data.get("status") == "running" else ""
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = HTML.replace("__REFRESH__", refresh).replace("__PAYLOAD__", payload)
    panel = update_panel(data)
    if panel:
        encoded = json.dumps(panel, ensure_ascii=False).replace("</", "<\\/")
        html = html.replace("</body>", '<script>document.getElementById("overview").insertAdjacentHTML("afterbegin",'
                            + encoded + ');</script></body>')
    if data.get("run") and (Path(data["run"]) / "result_guide.html").is_file():
        html = html.replace('<body>', '<body><nav id="result-guide-entry" style="padding:14px 24px;background:#f0f5ee">'
                            '<a href="result_guide.html" target="_blank" rel="noopener">Explore results &amp; evidence →</a></nav>', 1)
    try:
        from .retrieval_screening_panel import inject
    except ImportError:
        from retrieval_screening_panel import inject
    return inject(html, data.get('retrieval_screening'), language)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=8.0)
    parser.add_argument("--language", choices=("en",), default="en", help="dashboard language (en)")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not (run / "pipeline_manifest.json").is_file():
        raise SystemExit("run directory has no pipeline_manifest.json")
    if args.interval < 1:
        raise SystemExit("--interval must be at least one second")
    output = args.output or run / "dashboard.html"
    while True:
        data = data_for(run)
        output.write_text(render(data, language=args.language), encoding="utf-8")
        print(output, flush=True)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
