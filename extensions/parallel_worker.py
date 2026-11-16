"""Concurrent outer scheduler for archived screening/extraction/review workers.

The implementations invoked here are frozen vendor implementations.  This file
only fans independent articles out to threads and checkpoints completed work in
the same public contracts as the original serial worker.
"""
from __future__ import annotations

import argparse
import os
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.common import activate
activate()
from extensions.transport import retry_text_call, primary_attempts
from metapipeline.contracts import ContractError, read, write, require, screening_protocol

EXTRACT = ROOT / 'vendor/integration/vendor/evidence_extraction/system'
MINERS = ROOT / 'vendor/integration/vendor/miners'


def _workers(request, count):
    requested = _env_int('META_PIPELINE_ACTIVE_WORKERS', request['options']['max_workers'])
    require(requested <= 8, 'META_PIPELINE_ACTIVE_WORKERS must be between 1 and 8')
    return min(requested, max(1, count))


def _env_int(name, default, *, minimum=1):
    """Read an operational tuning value without making request JSON mutable."""
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except ValueError:
        raise ValueError(f'{name} must be an integer') from None


def _env_float(name, default, *, minimum=0.0):
    try:
        return max(minimum, float(os.environ.get(name, default)))
    except ValueError:
        raise ValueError(f'{name} must be a number') from None


def _screening_workers(request, count):
    configured = _env_int('META_PIPELINE_SCREEN_WORKERS', 2)
    return min(_workers(request, count), configured)


def _retryable_extraction_error(exc):
    """Retry transient provider failures only; bad credentials/models need action."""
    status = getattr(exc, 'status_code', None)
    response = getattr(exc, 'response', None)
    status = status or getattr(response, 'status_code', None)
    if status in {400, 401, 403, 404, 422}:
        return False
    return type(exc).__name__ not in {
        'AuthenticationError', 'PermissionDeniedError', 'NotFoundError',
        'BadRequestError', 'UnprocessableEntityError',
    }


def _extraction_workers(request, count):
    """Default to two text calls: faster than a four-way provider 429 loop."""
    configured = _env_int('META_PIPELINE_EXTRACT_WORKERS', min(2, request['options']['max_workers']))
    return min(_workers(request, count), configured)


def _review_workers(request, count):
    """Keep candidate review below the provider's request-rate ceiling.

    Review already subdivides every article into candidate batches.  Reusing
    the outer ``--workers`` setting here can create a large request burst even
    for a modest document set, so it has its own conservative default.
    """
    configured = _env_int(
        'META_PIPELINE_REVIEW_WORKERS',
        min(2, request['options']['max_workers']),
    )
    return min(_workers(request, count), configured)


def _retry_deferred_524() -> bool:
    """Whether this resume deliberately replays previously deferred 524s."""
    return os.environ.get('META_PIPELINE_RETRY_DEFERRED_524', '').strip().casefold() in {
        '1', 'true', 'yes', 'on',
    }


def _is_deferred_outage_record(record):
    return (record.get('status_code') == 524
            or record.get('error_type') == 'APITimeoutError')


def _needs_retry(decision):
    flags = set(decision.get('validation_flags') or [])
    return 'screening_model_error' in flags or any(
        str(flag).startswith('fulltext_response_not_parseable_after_') for flag in flags
    )


def screen(request, output):
    sys.path.insert(0, str(MINERS))
    from miners.paper_node_adapter import build_llm_callback
    from miners.case_screening_rob import FullTextProtocolScreener
    from miners.case_rob import pdf_sections
    from miners.rob_routing import RiskOfBiasRecommendationAgent

    output.mkdir(parents=True, exist_ok=True)
    from extensions.stage_inputs import bind_stage_inputs
    bind_stage_inputs(request, output, 'screening')
    prior_decisions, prior_routes = {}, {}
    if (output / 'screening_results.json').is_file():
        payload = read(output / 'screening_results.json')
        if payload.get('case_id') != request['case_id']:
            raise ValueError('screening checkpoint belongs to a different case')
        prior_decisions = {row['candidate_id']: row for row in payload.get('decisions', [])}
    if (output / 'risk_of_bias_recommendations.json').is_file():
        prior_routes = {row['article_id']: row for row in read(output / 'risk_of_bias_recommendations.json').get('recommendations', [])}
    protocol = screening_protocol(request)
    request_attempts = _env_int('META_PIPELINE_SCREEN_REQUEST_ATTEMPTS', primary_attempts())
    retry_delay = _env_float('META_PIPELINE_SCREEN_RETRY_DELAY_SECONDS', 5.0)

    def one(document):
        aid, paths = document['article_id'], document['paths']
        if aid in prior_decisions and (prior_decisions[aid]['final_status'] != 'Include' or aid in prior_routes or not request['options']['route_rob']):
            return aid, prior_decisions[aid], prior_routes.get(aid), True
        markdown_text = Path(paths['markdown']).read_text(encoding='utf-8') if paths.get('markdown') else None
        sections = ({'page_1': markdown_text} if markdown_text and markdown_text.strip()
                    else pdf_sections(Path(paths['pdf']), max_pages=1000, max_chars_per_page=20_000)
                    if paths.get('pdf') else {'page_1': ''})
        print(
            f'[screening] source candidate={aid} chars={sum(len(value) for value in sections.values())} '
            f'pages={len(sections)}',
            flush=True,
        )
        raw_callback = build_llm_callback(
            request['options']['model'], 'PIPELINE_LLM_API_KEY', os.environ.get('PIPELINE_LLM_BASE_URL')
        )
        def callback(prompt):
            print(f'[screening] request candidate={aid} input_chars={len(prompt)}', flush=True)
            response = retry_text_call(
                lambda: raw_callback(prompt),
                label=f'SCREENING candidate={aid}',
                attempts=request_attempts,
                base_delay=retry_delay,
            )
            print(f'[screening] response candidate={aid} output_chars={len(response)}', flush=True)
            return response

        def screen_with(active_callback):
            return FullTextProtocolScreener(active_callback).screen(
                aid, document.get('title', aid), sections, protocol,
                max_total_chars=sum(len(value) for value in sections.values()) + 100 * len(sections),
                abstract_source_text=document.get('abstract') or markdown_text)

        decision = prior_decisions.get(aid) or screen_with(callback)
        if _needs_retry(decision):
            raise RuntimeError(
                f'{aid} received no schema-valid screening decision; do not checkpoint an uncertain model failure'
            )
        route = prior_routes.get(aid)
        if request['options']['route_rob'] and decision['final_status'] == 'Include' and route is None:
            route_input = {'study_id': document['study_id'], 'title': decision.get('title', aid), 'abstract': document.get('abstract', '')}
            recommendation = RiskOfBiasRecommendationAgent(callback).recommend(
                route_input, sections, review_question=request['topic'])
            route = {'article_id': aid, **recommendation}
        return aid, decision, route, False

    documents = request['documents']
    workers = _screening_workers(request, len(documents))
    print(
        f'[screening] parallel workers={workers}; request_attempts={request_attempts}',
        flush=True,
    )
    failures = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='screen') as executor:
        futures = {executor.submit(one, document): document['article_id'] for document in documents}
        finished = 0
        for future in as_completed(futures):
            aid = futures[future]
            try:
                aid, decision, route, resumed = future.result()
            except Exception as exc:
                failures.append((aid, type(exc).__name__))
                print(f'[screening] failed candidate={aid} error={type(exc).__name__}: {exc}', flush=True)
                continue
            prior_decisions[aid] = decision
            if route is not None:
                prior_routes[aid] = route
            finished += 1
            print(f'[screening] {finished}/{len(documents)} {"resume " if resumed else ""}{aid}', flush=True)
            ordered = [prior_decisions[d['article_id']] for d in documents if d['article_id'] in prior_decisions]
            routes = [prior_routes[d['article_id']] for d in documents if d['article_id'] in prior_routes]
            write(output / 'screening_results.json', {'case_id': request['case_id'], 'decisions': ordered})
            write(output / 'risk_of_bias_recommendations.json', {'recommendations': routes})
    require(
        not failures,
        'screening tasks failed; completed decisions were saved; retry with --resume: ' + str(failures),
    )
    write(output / 'screening_complete.json', {'case_id': request['case_id'], 'article_count': len(prior_decisions)})
    print(f'[risk-of-bias routing] completed recommendations={len(prior_routes)}; detailed assessments are next, before extraction', flush=True)


def extract(request, config, output):
    """Run independently checkpointed Graph article/batch jobs.

    A failed article is recorded without aborting collection of other completed
    futures.  On resume, Graph reuses its locator and completed source-batch
    checkpoints rather than replaying the complete article.
    """
    from extensions.resilient_extraction import (
        _load_runtime,
        error_info, is_capacity_error, is_permanent_provider_error,
        is_provider_outage_error,
        run_article as run_resilient_article,
    )
    load_case_config, controller, tools, models = _load_runtime(request)
    task = load_case_config(config)
    # The sidecar receives frozen controller/tool modules once before its
    # worker threads start. This avoids concurrent sys.path/module mutation.
    runtime = (load_case_config, controller, tools, models)
    output.mkdir(parents=True, exist_ok=True)
    from extensions.stage_inputs import bind_stage_inputs
    bind_stage_inputs(request, output, 'extraction')
    prior = {}
    progress_path = output / 'extraction_progress.json'
    if progress_path.is_file():
        payload = read(progress_path)
        if payload.get('case_id') != request['case_id']:
            raise ValueError('extraction checkpoint belongs to a different case')
        for row in payload.get('article_runs', []):
            aid = row.get('article_id'); run_dir = Path(row.get('run_dir', '')).resolve()
            submission = run_dir / 'extraction_submission.json'
            if row.get('status') != 'completed' or not isinstance(aid, str) or not submission.is_file() or not run_dir.is_relative_to(output.resolve()):
                raise ValueError('invalid extraction checkpoint')
            prior[aid] = (row, read(submission).get('predictions', []))

    # A normal resume first completes articles that have not yet been tried.
    # Known 524s are preserved as explicit deferred work rather than sending
    # the same request again before the user elects to retry it.
    prior_outage_records = []
    prior_failures_path = output / 'extraction_failures.json'
    if prior_failures_path.is_file() and not _retry_deferred_524():
        prior_failures = read(prior_failures_path)
        if prior_failures.get('case_id') == request['case_id']:
            prior_outage_records = [
                row for row in prior_failures.get('failures', [])
                if isinstance(row, dict) and isinstance(row.get('article_id'), str)
                and _is_deferred_outage_record(row)
                and row['article_id'] not in prior
            ]

    def checkpoint():
        ordered_articles = [completed[item][0] for item in ids if item in completed]
        ordered_predictions = [row for item in ids if item in completed for row in completed[item][1]]
        write(progress_path, {
            'case_id': request['case_id'],
            'article_runs': ordered_articles,
            'prediction_count': len(ordered_predictions),
        })

    def one(aid):
        if aid in prior:
            return aid, prior[aid][0], prior[aid][1], True
        article_dir = output / 'articles' / aid / 'attempt_01'
        try:
            result = run_resilient_article(request, task, aid, article_dir, runtime=runtime)
        except Exception as exc:
            article_dir.mkdir(parents=True, exist_ok=True)
            previous = read(article_dir / 'failure_history.json') if (article_dir / 'failure_history.json').is_file() else {'article_id': aid, 'failures': []}
            record = {**error_info(exc), 'retryable': not is_permanent_provider_error(exc), 'recorded_at': time.time()}
            previous['failures'].append(record)
            write(article_dir / 'failure_history.json', previous)
            write(article_dir / 'failure.json', {'article_id': aid, 'status': 'failed', **record})
            raise
        for name, key in [('run_manifest','manifest'), ('raw_system_output','raw_system_output'), ('extraction_submission','extraction_submission'), ('issues','issues'), ('controller_state','controller_state')]:
            write(article_dir / (name + '.json'), result[key])
        return aid, {'article_id': aid, 'status': 'completed', 'run_dir': str(article_dir), 'issues': result['issues']}, result['extraction_submission']['predictions'], False

    ids = list(task.included_article_ids)
    completed = dict(prior)
    if completed:
        checkpoint()
    workers = _extraction_workers(request, len(ids))
    capacity = workers
    print(
        f'[extraction-resilient] parallel workers={workers}; '
        f'message+schema request budget={os.environ.get("META_PIPELINE_EXTRACT_BATCH_CHARS", "20000")} chars; '
        f'client timeout={os.environ.get("META_PIPELINE_EXTRACT_TIMEOUT_SECONDS", "300")}s',
        flush=True,
    )
    prior_deferred_ids = {row['article_id'] for row in prior_outage_records}
    if prior_deferred_ids:
        print(
            f'[extraction-resilient] retaining {len(prior_deferred_ids)} prior 524-deferred article(s); '
            'running remaining articles first (use --retry-deferred to replay them)',
            flush=True,
        )
    queue = deque(aid for aid in ids if aid not in prior and aid not in prior_deferred_ids)
    active, failures, unscheduled = {}, list(prior_outage_records), []
    deferred_outages = [row['article_id'] for row in prior_outage_records]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='extract-graph') as executor:
        def schedule():
            while queue and len(active) < capacity:
                aid = queue.popleft()
                active[executor.submit(one, aid)] = aid

        schedule()
        while active:
            done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in done:
                aid = active.pop(future)
                try:
                    aid, article, predictions, resumed = future.result()
                except Exception as exc:
                    failures.append({'article_id': aid, **error_info(exc)})
                    print(f'[extraction-resilient] failed candidate={aid} error={type(exc).__name__}: {exc}', flush=True)
                    if is_capacity_error(exc) and capacity > 1:
                        capacity = 1
                        print('[extraction-resilient] provider capacity signal: reducing new extraction calls to one worker', flush=True)
                    elif is_capacity_error(exc):
                        # A 429 at one concurrent call is no longer a local
                        # scheduling problem.  Stop the request storm and
                        # preserve every checkpoint for a later resume.
                        unscheduled.extend(queue)
                        queue.clear()
                        print('[extraction-resilient] provider still rejects one concurrent call: checkpoints saved; resume after capacity recovers', flush=True)
                    elif is_provider_outage_error(exc):
                        # A 524 is often payload/article-specific: a compact
                        # article can still finish even when one long article
                        # times out at the gateway. Defer only this article,
                        # retain its locator/batch checkpoints, and continue
                        # the remaining candidates. The command still exits
                        # nonzero below, so the pipeline cannot enter review
                        # or analysis until every article has completed.
                        deferred_outages.append(aid)
                        if capacity > 1:
                            capacity = 1
                            print('[extraction-resilient] provider outage/timeout: deferring this article and continuing remaining articles with one worker', flush=True)
                        else:
                            print('[extraction-resilient] provider outage/timeout: deferring this article and continuing remaining articles', flush=True)
                    if is_permanent_provider_error(exc):
                        unscheduled.extend(queue)
                        queue.clear()
                        print('[extraction-resilient] permanent provider error: no further articles will be submitted in this attempt', flush=True)
                    continue
                completed[aid] = (article, predictions)
                checkpoint()
                print(f'[extraction] {len(completed)}/{len(ids)} {"resume " if resumed else ""}{aid}', flush=True)
            schedule()
    if failures or unscheduled:
        write(output / 'extraction_failures.json', {
            'schema_version': 'meta_pipeline/extraction_failures/1.0',
            'case_id': request['case_id'],
            'completed_article_ids': [aid for aid in ids if aid in completed],
            'failures': failures,
            'deferred_article_ids': deferred_outages,
            'not_submitted_article_ids': list(unscheduled),
            'resume_instruction': 'Use --resume with this same output directory after resolving the provider error. Deferred and failed articles are retried; completed articles and completed source batches are reused. Review and analysis remain blocked until extraction has no deferred or failed articles.',
        })
        raise ContractError('extraction tasks failed; completed article and batch checkpoints were saved; retry with --resume: ' + str([row['article_id'] for row in failures]))
    articles = [completed[aid][0] for aid in ids]
    predictions = [row for aid in ids for row in completed[aid][1]]
    # A successful deferred retry must not leave the root failure artifact in
    # place: the live dashboard and a later resume should see extraction as
    # complete. Preserve the failed-attempt record separately for audit.
    prior_failures_path = output / 'extraction_failures.json'
    if prior_failures_path.is_file():
        history_path = output / 'extraction_failure_history.json'
        history = read(history_path) if history_path.is_file() else {
            'schema_version': 'meta_pipeline/extraction_failure_history/1.0',
            'case_id': request['case_id'], 'events': [],
        }
        history['events'].append({
            'recovered_at': time.time(),
            'failure_artifact': read(prior_failures_path),
        })
        write(history_path, history)
        prior_failures_path.unlink()
    write(output / 'case_extraction_submission.json', {'case_id': request['case_id'], 'task_id': task.task_id, 'predictions': predictions})
    write(output / 'case_run_manifest.json', {'case_id': request['case_id'], 'config_path': str(config), 'article_runs': articles, 'failed_article_ids': [], 'prediction_count': len(predictions)})
    write(output / 'extraction_complete.json', {'case_id': request['case_id'], 'article_count': len(articles), 'prediction_count': len(predictions)})


def review(request, config, submission, output):
    if (output / 'update_review_seed.json').is_file():
        from extensions.update import incremental_review
        return incremental_review(request, config, submission, output)
    # Keep the archived review contract and output files, while applying a
    # separate request-rate limit.  The archived worker binds review fan-out to
    # the global run worker count and retries every 5 seconds, which can turn a
    # provider 503 into a request storm for evidence-dense cases.
    # This module is launched as a standalone worker, so the frozen extractor
    # package is not guaranteed to be on ``sys.path`` yet.
    if str(EXTRACT) not in sys.path:
        sys.path.insert(0, str(EXTRACT))
    from constrained_meta_agent.loader import load_case_config
    from evidence_reviewer.reviewer import write_review_summary
    from extensions.resumable_review import run_review, validate_legacy_review

    payload = read(submission)
    workers = _review_workers(request, len(payload.get('predictions', [])))
    retry_delay = _env_float('META_PIPELINE_REVIEW_RETRY_DELAY_SECONDS', 60.0)
    print(
        f"[review] parallel workers={workers}; retry delay={retry_delay:g}s",
        flush=True,
    )
    task = load_case_config(config)
    result = run_review(
        task, payload, output,
        model=request['options']['model'],
        batch_size=_env_int('META_PIPELINE_REVIEW_BATCH_SIZE', 5, minimum=2),
        workers=workers,
        max_retries=request['options']['max_retries'],
        retry_delay=retry_delay,
        timeout=_env_float('META_PIPELINE_REVIEW_TIMEOUT_SECONDS', 300.0, minimum=1.0),
        legacy_validator=lambda: validate_legacy_review(request, submission, output),
    )
    write(output / 'candidate_evidence_review_structured.json', result)
    write_review_summary(output / 'review_summary.md', result)
    if result['failed_review_count']:
        raise RuntimeError('candidate review failed; see review artifacts')
    write(output / 'review_complete.json', {
        'case_id': request['case_id'],
        'reviewed_count': result['completed_review_count'],
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('screen', 'extract', 'review'))
    parser.add_argument('--request', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path); parser.add_argument('--submission', type=Path)
    args = parser.parse_args(); request = read(args.request)
    if args.stage == 'screen': screen(request, args.output)
    elif args.stage == 'extract': extract(request, args.config, args.output)
    else: review(request, args.config, args.submission, args.output)


if __name__ == '__main__':
    main()
