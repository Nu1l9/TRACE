from __future__ import annotations
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from .common import ROOT, BASELINE, read, write, require, digest
from .provider import fingerprint
from .settings import settings_fingerprint
from trace_runtime.pipeline import Runner, runtime_environment, preflight_runtime
from .handoff import install as install_evidence_handoff

install_evidence_handoff()


def implementation_fingerprint():
    # This checkout owns its embedded stage modules. Their adapters and prompts affect results just
    # as much as the outer coordinator, and must participate in resume checks.
    paths=list((ROOT/'extensions').glob('*.py'))+list((ROOT/'resources/rob').rglob('*.txt'))
    paths.extend((ROOT/'literature_retrieval').glob('*.py'))
    for pattern in ('*.py', '*.R'):
        paths.extend((ROOT/'vendor/integration').rglob(pattern))
    return digest({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})


def compatible_resume_upgrade(previous_lock, current_lock):
    """Allow only a reviewed, exact source-to-target compatibility transition.

    The release record lives outside the hashed implementation. Any further
    code edit changes the target fingerprint and disables this exception.
    Request/source/output validation remains the frozen Runner's responsibility.
    """
    stable = lambda value: {k:v for k,v in value.items() if k != 'implementation_fingerprint'}
    if stable(previous_lock) != stable(current_lock):
        return None
    release_file = ROOT/'compatibility/update_mode_resume.json'
    if not release_file.is_file():
        return None
    release = read(release_file)
    if release.get('schema_version') != 'resume_compatibility/1.0':
        return None
    # Keep prior release transitions auditable while allowing a later, narrowly
    # reviewed recovery patch to name exactly one predecessor fingerprint.
    transitions = [{
        'target_implementation_fingerprint': release.get('target_implementation_fingerprint'),
        'compatible_source_fingerprints': release.get('compatible_source_fingerprints', []),
        'reason': release.get('reason'),
    }, *release.get('resume_transitions', [])]
    for transition in transitions:
        if (transition.get('target_implementation_fingerprint') == current_lock.get('implementation_fingerprint')
                and previous_lock.get('implementation_fingerprint') in transition.get('compatible_source_fingerprints', [])):
            return {
                'reason': transition.get('reason') or 'verified backward-compatible recovery patch',
                'compatibility_record': str(release_file.relative_to(ROOT)),
                'previous_implementation_fingerprint': previous_lock['implementation_fingerprint'],
                'new_implementation_fingerprint': current_lock['implementation_fingerprint'],
            }
    return None


def provider_migration_retry(previous_lock, current_lock, manifest, output):
    """Permit a provider change only for an interrupted extraction stage.

    Completed article checkpoints remain attributable to the previous provider;
    only pending articles use the new provider. No downstream evidence review or
    analysis may exist, because those stages would mix provider-dependent work.
    """
    stable = lambda value: {k: v for k, v in value.items()
                            if k not in {'implementation_fingerprint', 'provider_routes_fingerprint'}}
    if stable(previous_lock) != stable(current_lock):
        return None
    if previous_lock.get('provider_routes_fingerprint') == current_lock.get('provider_routes_fingerprint'):
        return None
    stages = manifest.get('stages', {})
    if (stages.get('screening', {}).get('status') != 'completed'
            or stages.get('screening_rob', {}).get('status') != 'completed'
            or stages.get('extraction', {}).get('status') not in {'failed', 'paused'}
            or any(name in stages for name in ('review', 'analysis_check', 'analysis'))
            or not (Path(output) / '02_extraction' / 'extraction_progress.json').is_file()):
        return None
    release_file = ROOT / 'compatibility' / 'update_mode_resume.json'
    if not release_file.is_file():
        return None
    release = read(release_file)
    if (release.get('schema_version') != 'resume_compatibility/1.0'
            or release.get('target_implementation_fingerprint') != current_lock.get('implementation_fingerprint')
            or previous_lock.get('implementation_fingerprint') not in release.get('compatible_source_fingerprints', [])):
        return None
    return {'reason': 'provider migration after failed extraction; completed article checkpoints retained',
            'compatibility_record': str(release_file.relative_to(ROOT)),
            'previous_implementation_fingerprint': previous_lock['implementation_fingerprint'],
            'new_implementation_fingerprint': current_lock['implementation_fingerprint'],
            'previous_provider_routes_fingerprint': previous_lock['provider_routes_fingerprint'],
            'new_provider_routes_fingerprint': current_lock['provider_routes_fingerprint']}


def pristine_extraction_rebind(previous_lock, current_lock, manifest, output, current_settings):
    """Allow current downstream code/config before extraction produced evidence.

    A run created with reused screening can be interrupted while its first
    locator calls are still pending.  At that point there is no model-derived
    extraction result to mix with a new implementation or provider.  Rebinding
    is safe when screening and RoB are already frozen, their bias settings are
    unchanged, and neither extraction nor any downstream stage has produced a
    completed artefact.
    """
    output = Path(output)
    stages = manifest.get('stages', {})
    if (previous_lock.get('schema_version') != current_lock.get('schema_version')
            or previous_lock.get('baseline_lock_hash') != current_lock.get('baseline_lock_hash')
            or stages.get('screening', {}).get('status') != 'completed'
            or stages.get('screening_rob', {}).get('status') != 'completed'
            or stages.get('extraction', {}).get('status') not in {'failed', 'paused'}
            or any(name in stages for name in ('review', 'analysis_check', 'analysis'))
            or (output/'02_extraction/extraction_complete.json').exists()
            or list((output/'02_extraction/articles').glob('**/batches/*.json'))
            or list((output/'02_extraction/articles').glob('**/extraction_submission.json'))):
        return None
    progress_path = output/'02_extraction/extraction_progress.json'
    if progress_path.is_file() and any(
            row.get('status') == 'completed' for row in read(progress_path).get('article_runs', [])):
        return None
    prior_settings_path = output/'extensions.normalized.json'
    if not prior_settings_path.is_file():
        return None
    prior_settings = read(prior_settings_path)
    if prior_settings.get('bias') != current_settings.get('bias'):
        return None
    return {
        'reason': 'rebind current downstream implementation before any extraction evidence was completed',
        'previous_implementation_fingerprint': previous_lock.get('implementation_fingerprint'),
        'new_implementation_fingerprint': current_lock.get('implementation_fingerprint'),
        'previous_settings_fingerprint': previous_lock.get('settings_fingerprint'),
        'new_settings_fingerprint': current_lock.get('settings_fingerprint'),
        'completed_extraction_articles': 0,
    }

STAGE_PROGRESS = {
    'screening_rob': ('3/7', 'Post-screening risk of bias'),
    'screening': ('2/7', 'Study screening'),
    'extraction': ('4/7', 'Evidence extraction'),
    'review': ('5/7', 'Evidence review'),
    'analysis_check': ('6/7', 'Analysis contract checks'),
    'analysis': ('7/7', 'Statistical analysis and forest plots'),
}


def archive_terminal_analysis_for_retry(output: Path, state: dict, *, preserve_analysis_check=False,
                                        rebuild_handoff=False) -> dict:
    """Archive a terminal analysis while retaining its immutable upstream evidence.

    This is deliberately narrower than a general stage reset. The screening,
    screening-RoB, extraction and candidate-review checkpoints must all be
    complete, and a running analysis is never moved underneath its worker.
    """
    output = Path(output).resolve()
    stages = state.get('stages', {})
    required = ('screening', 'screening_rob', 'extraction', 'review', 'analysis_check')
    require(all(stages.get(name, {}).get('status') == 'completed' for name in required),
            '--retry-analysis requires completed screening, screening-RoB, extraction, review, and analysis-check stages')
    prior_analysis = stages.get('analysis', {})
    require(prior_analysis.get('status') in {'completed', 'failed', 'paused'},
            '--retry-analysis requires a terminal analysis stage; it cannot replace a running analysis')
    require((output / '02_extraction' / 'extraction_complete.json').is_file()
            and (output / '03_review' / 'review_complete.json').is_file(),
            '--retry-analysis requires completed extraction and review artifacts')

    history = output / 'analysis_history' / str(time.time_ns())
    history.mkdir(parents=True, exist_ok=False)
    moved = []
    analysis_dir = output / '05_analysis'
    if analysis_dir.exists():
        shutil.move(str(analysis_dir), str(history / '05_analysis'))
        moved.append('05_analysis')
    brain = output / 'brain_result.json'
    if brain.is_file():
        shutil.move(str(brain), str(history / 'brain_result.json'))
        moved.append('brain_result.json')
    if rebuild_handoff:
        # The handoff is derived entirely from completed screening, extraction,
        # and review artifacts. Archive its old output so the current handoff
        # implementation can safely regenerate candidates before analysis.
        for name in ('04_analysis_input', 'handoff_audit.json'):
            source = output / name
            if source.exists():
                shutil.move(str(source), str(history / name))
                moved.append(name)
    prior_error = state.pop('error', None)
    # A retry launched solely because the analysis implementation changed does
    # not alter the already validated evidence contract.  Keep stage six in
    # that case; the ordinary explicit retry keeps its historical behaviour of
    # rerunning the cheap contract check.
    stage_names = ('analysis',) if preserve_analysis_check and not rebuild_handoff else ('analysis_check', 'analysis')
    for name in stage_names:
        stages.pop(name, None)
    state.pop('analysis_status', None)
    audit = {
        'schema_version': 'analysis_retry/1.0',
        'reason': 'explicit analysis retry from unchanged completed upstream artifacts',
        'archive': str(history.relative_to(output)),
        'archived_artifacts': moved,
        'prior_analysis_stage': prior_analysis,
        'preserved_analysis_check': preserve_analysis_check,
        'rebuilt_handoff': rebuild_handoff,
        'prior_error': prior_error,
    }
    write(output / 'analysis_retry_audit.json', audit)
    state.setdefault('recovery_history', []).append({
        'event': 'explicit_analysis_retry',
        'archive': audit['archive'],
    })
    return audit


def frozen_analysis_request_for_retry(request: dict, output: Path) -> dict:
    """Return the exact request that produced completed upstream evidence.

    The launch request identifies the case, but its PDF paths may have been
    replaced by shared MinerU paths during the original preparation stage. A
    retry of analysis must never recalculate that input binding.
    """
    path = Path(output).resolve() / 'request.normalized.json'
    require(path.is_file(), '--retry-analysis requires the run\'s frozen request.normalized.json')
    frozen = read(path)
    require(request.get('case_id') == frozen.get('case_id'),
            '--retry-analysis request case_id does not match the existing run')
    return frozen


def rebind_runner_input_for_analysis_retry(request: dict, output: Path) -> None:
    """Update only the frozen Runner's code-sensitive fingerprint.

    The archived Runner includes its own Python source in ``input_fingerprint``.
    That makes a reviewed analysis-only code repair look like a document change.
    Stage locks separately bind the request semantics and source bytes, so
    verify them first and then record the new Runner fingerprint in the manifest.
    """
    from trace_runtime.pipeline import fingerprint as runner_fingerprint
    from .stage_inputs import stage_signature

    output = Path(output).resolve()
    manifest_path = output / 'pipeline_manifest.json'
    frozen_path = output / 'request.normalized.json'
    require(manifest_path.is_file() and frozen_path.is_file(),
            '--retry-analysis requires an existing frozen run manifest and request')
    require(read(frozen_path) == request,
            '--retry-analysis must use the exact request frozen by the existing run')
    manifest = read(manifest_path)
    stages = manifest.get('stages', {})
    for stage, directory in (('screening', '01_screening'), ('extraction', '02_extraction')):
        lock = output / directory / f'{stage}_inputs.lock.json'
        require(lock.is_file() and read(lock).get('fingerprint') == stage_signature(request, stage),
                f'--retry-analysis refused: {stage} sources or semantics changed')
    for stage in ('screening', 'screening_rob', 'extraction', 'review'):
        entry = stages.get(stage, {})
        require(entry.get('status') == 'completed' and isinstance(entry.get('outputs'), dict),
                f'--retry-analysis requires a completed {stage} checkpoint')
        for relative, expected in entry['outputs'].items():
            path = output / relative
            require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                    f'--retry-analysis refused: completed {stage} artifact changed: {relative}')
    current = runner_fingerprint(request)
    previous = manifest.get('input_fingerprint')
    if previous == current:
        return
    manifest['input_fingerprint'] = current
    manifest.setdefault('recovery_history', []).append({
        'event': 'analysis_retry_runner_fingerprint_rebind',
        'reason': 'stage-bound request and source files verified; only archived Runner implementation changed',
        'previous_input_fingerprint': previous,
        'new_input_fingerprint': current,
    })
    write(manifest_path, manifest)
    write(output / 'analysis_retry_input_binding_audit.json', {
        'schema_version': 'analysis_retry_input_binding/1.0',
        'reason': 'stage-bound request and source files verified; only archived Runner implementation changed',
        'previous_input_fingerprint': previous,
        'new_input_fingerprint': current,
        'validated_stage_locks': ['01_screening/screening_inputs.lock.json', '02_extraction/extraction_inputs.lock.json'],
    })


class ProgressRunner(Runner):
    """External terminal progress view; the frozen Runner remains unchanged."""

    def __init__(self, *args, **kwargs):
        self._dashboard_stop = threading.Event()
        super().__init__(*args, **kwargs)
        self._render_dashboard()
        url=(self.out / 'dashboard.html').resolve().as_uri()
        print(f'\033]8;;{url}\033\\[dashboard] Live dashboard ready; click to open\033]8;;\033\\  {url}', flush=True)
        self._dashboard_thread = threading.Thread(target=self._monitor_dashboard, daemon=True)
        self._dashboard_thread.start()

    def _render_dashboard(self):
        """Best-effort local dashboard; a rendering issue must not stop a run."""
        try:
            from scripts.render_run_dashboard import data_for, render
            data = data_for(self.out)
            dashboard = self.out / 'dashboard.html'
            temporary = dashboard.with_suffix('.html.tmp')
            temporary.write_text(render(data), encoding='utf-8')
            temporary.replace(dashboard)
        except Exception as exc:
            print(f'[dashboard] update skipped: {type(exc).__name__}', flush=True)

    def _dashboard_signature(self):
        """Watch small checkpoint files, never rescan every article on a timer."""
        paths = (
            self.out / 'pipeline_manifest.json',
            self.out / '01_screening' / 'screening_results.json',
            self.out / '01_screening' / 'risk_of_bias_complete.json',
            self.out / '02_extraction' / 'extraction_progress.json',
            self.out / '02_extraction' / 'extraction_failures.json',
            self.out / '03_review' / 'review_complete.json',
            self.out / '05_analysis' / 'analysis_summary.json',
        )
        result = []
        for path in paths:
            try:
                stat = path.stat()
                result.append((str(path.relative_to(self.out)), stat.st_mtime_ns, stat.st_size))
            except FileNotFoundError:
                result.append((str(path.relative_to(self.out)), None, None))
        return tuple(result)

    def _monitor_dashboard(self):
        previous = self._dashboard_signature()
        while not self._dashboard_stop.wait(2):
            current = self._dashboard_signature()
            if current != previous:
                self._render_dashboard()
                previous = current

    def save(self, status):
        super().save(status)
        self._render_dashboard()

    def close_dashboard_monitor(self):
        self._dashboard_stop.set()
        if hasattr(self, '_dashboard_thread'):
            self._dashboard_thread.join(timeout=1)
        self._render_dashboard()

    def _tail_log(self, path: Path, stop: threading.Event) -> None:
        try:
            position = path.stat().st_size
        except FileNotFoundError:
            position = 0
        while not stop.is_set():
            try:
                with path.open('r', encoding='utf-8', errors='replace') as handle:
                    handle.seek(position)
                    chunk = handle.read()
                    position = handle.tell()
                if chunk:
                    print(chunk, end='' if chunk.endswith('\n') else '\n', flush=True)
            except FileNotFoundError:
                pass
            stop.wait(0.2)
        try:
            with path.open('r', encoding='utf-8', errors='replace') as handle:
                handle.seek(position)
                chunk = handle.read()
            if chunk:
                print(chunk, end='' if chunk.endswith('\n') else '\n', flush=True)
        except FileNotFoundError:
            pass

    def stage(self, name, command, outputs):
        previous = self.state['stages'].get(name, {})
        if previous.get('status') == 'completed':
            return super().stage(name, command, outputs)
        marker, label = STAGE_PROGRESS.get(name, ('?/6', name))
        print(f'\n[progress {marker}] {label}: started', flush=True)
        stop = threading.Event()
        log = self.out / 'logs' / f'{name}.log'
        follower = threading.Thread(target=self._tail_log, args=(log, stop), daemon=True)
        follower.start()
        started = time.monotonic()
        try:
            result = super().stage(name, command, outputs)
            print(f'[progress {marker}] {label}: completed ({time.monotonic() - started:.0f}s)', flush=True)
            return result
        finally:
            stop.set()
            follower.join(timeout=1)


class ExtensionRunner(ProgressRunner):
    """Compose stage execution, contract validation and resumable checkpoints."""
    def __init__(self, request, output, environment, settings, resume=False, provider_profile=None,
                 retry_analysis=False, rebuild_handoff=False):
        self.settings=settings
        self._resume=resume
        output=Path(output).resolve()
        new_run=not (output/'pipeline_manifest.json').exists()
        guard=output/'extensions.lock.json'
        lock={'schema_version':'extensions_lock/8.0','settings_fingerprint':settings_fingerprint(settings),'implementation_fingerprint':implementation_fingerprint(), 'baseline_lock_hash':hashlib.sha256((ROOT/'component_provenance.json').read_bytes()).hexdigest(),
              'provider_routes_fingerprint':fingerprint(provider_profile) if provider_profile else None}
        compatibility_audit = None
        provider_migration_audit = None
        extraction_rebind_audit = None
        explicit_analysis_retry = None
        if guard.exists():
            require(resume,'existing run output requires --resume')
            previous_lock=read(guard)
            if previous_lock != lock:
                compatibility_audit = compatible_resume_upgrade(previous_lock, lock)
                # Permit narrowly scoped recovery of a failed/paused screening
                # checkpoint, a failed screening-time RoB checkpoint, or a
                # failed/paused analysis checkpoint.  A transport hardening
                # change must not discard already checkpointed articles.
                unchanged={key:value for key,value in previous_lock.items()
                           if key!='implementation_fingerprint'} == {key:value for key,value in lock.items()
                                                                       if key!='implementation_fingerprint'}
                manifest=read(output/'pipeline_manifest.json') if (output/'pipeline_manifest.json').is_file() else {}
                provider_migration_audit = provider_migration_retry(previous_lock, lock, manifest, output)
                extraction_rebind_audit = pristine_extraction_rebind(
                    previous_lock, lock, manifest, output, settings)
                stages=manifest.get('stages',{})
                downstream=('extraction','review','analysis_check','analysis')
                screening_retry=(unchanged
                           and stages.get('screening',{}).get('status') in ('failed','paused')
                           and not any(name in stages for name in ('screening_rob',*downstream))
                           and (output/'01_screening').is_dir())
                screening_rob_retry=(unchanged
                           and stages.get('screening',{}).get('status')=='completed'
                           and stages.get('screening_rob',{}).get('status') in ('failed','paused')
                           and not any(name in stages for name in downstream)
                           and (output/'01_screening/risk_of_bias.progress.json').is_file())
                review_retry=(unchanged
                           and all(stages.get(name,{}).get('status')=='completed'
                                   for name in ('screening','screening_rob','extraction'))
                           and stages.get('review',{}).get('status') in ('failed','paused')
                           and not any(name in stages for name in ('analysis_check','analysis'))
                           and (output/'02_extraction/extraction_complete.json').is_file())
                handoff_retry=(unchanged
                           and all(stages.get(name,{}).get('status')=='completed'
                                   for name in ('screening','screening_rob','extraction','review'))
                           and not any(name in stages for name in ('analysis_check','analysis'))
                           and (output/'02_extraction/extraction_complete.json').is_file()
                           and (output/'03_review/review_complete.json').is_file())
                analysis_retry=(unchanged
                           and all(stages.get(name,{}).get('status')=='completed'
                                   for name in ('screening','screening_rob','extraction','review','analysis_check'))
                           and stages.get('analysis',{}).get('status') in ('failed','paused'))
                analysis_metadata_recovery=(analysis_retry
                           and (output/'05_analysis/analysis_result.json').is_file()
                           and (output/'05_analysis/analysis_summary.json').is_file())
                # An explicit analysis retry deliberately applies the current
                # Active analysis implementation and settings.  Its scientific
                # boundary is protected by rebind_runner_input_for_analysis_retry
                # before this constructor: source bytes, stage semantics and
                # every completed upstream artifact must match.  Requiring an
                # old settings fingerprint here would make a repaired analysis
                # setting impossible to use without rerunning stages 1–6.
                explicit_analysis_retry=(retry_analysis
                           and all(stages.get(name,{}).get('status')=='completed'
                                   for name in ('screening','screening_rob','extraction','review','analysis_check'))
                           and stages.get('analysis',{}).get('status') in ('completed','failed','paused'))
                require(compatibility_audit or provider_migration_audit or extraction_rebind_audit or screening_retry or screening_rob_retry or review_retry or handoff_retry or analysis_retry or explicit_analysis_retry,
                        'Pipeline code or configuration changed; start a new output directory')
                audit_name=('extraction_rebind_audit.json' if extraction_rebind_audit else
                            'screening_retry_audit.json' if screening_retry else
                            'screening_rob_retry_audit.json' if screening_rob_retry else
                            'review_retry_audit.json' if review_retry else
                            'handoff_retry_audit.json' if handoff_retry else
                            'analysis_metadata_recovery_audit.json' if analysis_metadata_recovery else
                            'analysis_retry_audit.json')
                if provider_migration_audit:
                    prior_routes = read(output/'provider_routes.json') if (output/'provider_routes.json').is_file() else None
                    write(output/'provider_migration_audit.json', {
                        **provider_migration_audit,
                        'previous_provider_routes': prior_routes,
                        'checkpoint_path': '02_extraction/extraction_progress.json',
                        'policy': 'screening and RoB stay frozen; completed extraction articles are reused; pending articles use the new provider',
                    })
                elif extraction_rebind_audit:
                    write(output/'extraction_rebind_audit.json', extraction_rebind_audit)
                elif not compatibility_audit and not explicit_analysis_retry:
                    write(output/audit_name,{
                        'reason':('retry of failed or paused screening with retained per-article checkpoint'
                                  if screening_retry else
                                  'retry of failed or paused screening-time RoB with retained checkpoint'
                                  if screening_rob_retry else
                                  'retry of failed or paused candidate review with completed extraction artefacts'
                                  if review_retry else
                                  'retry analysis handoff with completed extraction and candidate-review artefacts'
                                  if handoff_retry else
                                  'reuse completed analysis after metadata-write failure'
                                  if analysis_metadata_recovery else
                                  'retry of failed or paused analysis with completed upstream stage artefacts'),
                        'previous_implementation_fingerprint':previous_lock.get('implementation_fingerprint'),
                        'new_implementation_fingerprint':lock['implementation_fingerprint'],
                    })
        elif (output/'pipeline_manifest.json').exists():
            require(False,'cannot attach structured changes to an existing legacy run directory; choose a new output')
        retry_audit = None
        if retry_analysis:
            # Archive before the frozen Runner initialises.  Runner.__init__
            # immediately saves the manifest as ``running``; doing this after
            # that point turns a terminal analysis failure into an apparent
            # live analysis and makes the retry reject itself.
            retry_state = read(output/'pipeline_manifest.json')
            retry_audit = archive_terminal_analysis_for_retry(
                output, retry_state, preserve_analysis_check=not rebuild_handoff,
                rebuild_handoff=rebuild_handoff)
            write(output/'pipeline_manifest.json', retry_state)
        initial_artifacts = ()
        if new_run and (output/'00_retrieval').exists():
            from literature_retrieval.pipeline import validate_seed
            validate_seed(output, request)
            initial_artifacts = ('00_retrieval',)
        super().__init__(request,output,environment,resume,initial_artifacts=initial_artifacts)
        if retry_analysis:
            audit = retry_audit
            if guard.exists() and previous_lock.get('provider_routes_fingerprint') != lock.get('provider_routes_fingerprint'):
                audit['provider_route_changed_for_new_analysis'] = True
                audit['previous_provider_routes_fingerprint'] = previous_lock.get('provider_routes_fingerprint')
                audit['new_provider_routes_fingerprint'] = lock.get('provider_routes_fingerprint')
                write(self.out/'analysis_retry_audit.json', audit)
            self.save('running')
            print('[analysis] prior analysis archived; retaining completed stages 1–4'
                  if rebuild_handoff else
                  '[analysis] prior analysis archived; retaining completed stages 1–6', flush=True)
        if compatibility_audit:
            write(output/'update_mode_compatibility_audit.json',compatibility_audit)
        if provider_profile:
            from .provider import audit
            write(output/'provider_routes.json',audit(provider_profile))
        write(guard,lock)
        # Workers consume a frozen copy, so an edited sidecar cannot change a running job.
        snapshot=json.loads(json.dumps(settings))
        for group,field in [('graph','patch_file'),('bias','assessments_file')]:
            if settings[group].get(field):
                payload=read(settings[group][field])
                dest=output/'extension_inputs'/(field+'.json')
                if dest.exists():require(read(dest)==payload,'extension input snapshot changed: '+field)
                else:write(dest,payload)
                snapshot[group][field]=str(dest)
        write(output/'extensions.normalized.json',snapshot)
        if new_run:
            from .update import implementation_versions
            write(output/'update_dependencies.json',implementation_versions())
    def worker(self,stage,out,*extra):
        # Article-level concurrency is supplied by this external coordinator;
        # the archived workers under vendor/ are never edited.
        if stage in ('screen','extract','review'):
            return [sys.executable,str(ROOT/'extensions/parallel_worker.py'),stage,
                    '--request',str(self.out/'request.normalized.json'),'--output',str(out),*map(str,extra)]
        return super().worker(stage,out,*extra)
    def _recover_pre_partition_screening_rob(self):
        """Re-run only the historical all-or-nothing RoB placeholder batch.

        Older Graph runs could finish screening-time RoB in zero seconds after one
        uncovered route raised before provider calls. The nonblocking wrapper
        then wrote `needs_assessment` for every included article and marked the
        stage completed. This narrowly detects that obsolete result, archives
        it, and lets the new per-article route partition run on `--resume`.
        It never resets a real partial RoB batch or a completed draft.
        """
        if not self._resume:
            return False
        robdir=self.out/'01_screening'
        failures_path=robdir/'risk_of_bias_failures.json'
        assessments_path=robdir/'risk_of_bias_assessments.json'
        stage=self.state.get('stages',{}).get('screening_rob',{})
        if stage.get('status')!='completed' or not (failures_path.is_file() and assessments_path.is_file()):
            return False
        try:
            failures=read(failures_path).get('failures',[])
            assessments=read(assessments_path).get('assessments',[])
        except Exception:
            return False
        stagewide_route_failure=any(
            item.get('scope')=='stage' and 'No defensible RoB tool route for included article:' in str(item.get('message',''))
            for item in failures if isinstance(item,dict))
        placeholders=bool(assessments) and all(
            item.get('status')=='needs_assessment' and item.get('overall_judgment')=='unknown'
            for item in assessments if isinstance(item,dict))
        if not (stagewide_route_failure and placeholders):
            return False
        archive=robdir/'route_partition_recovery'/str(time.time_ns())
        archive.mkdir(parents=True,exist_ok=False)
        artifact_names=(
            'risk_of_bias_assessments.json','risk_of_bias_failures.json','risk_of_bias_complete.json',
            'risk_of_bias_retry.md','risk_of_bias_effective_routes.json','risk_of_bias.input.json',
            'risk_of_bias.progress.json','risk_of_bias_input_legacy.json','risk_of_bias_input_migration.json',
        )
        archived=[]
        for name in artifact_names:
            source=robdir/name
            if source.is_file():
                shutil.copy2(source,archive/name)
                source.unlink()
                archived.append(name)
        write(robdir/'risk_of_bias_route_partition_recovery.json',{
            'reason':'pre_partition_stagewide_uncovered_route_placeholder',
            'recovery':'rerun only routable article/target RoB assessments; retain uncovered articles as unknown/manual',
            'archive':str(archive.relative_to(robdir)),
            'archived_artifacts':archived,
        })
        self.state['stages'].pop('screening_rob',None)
        self.state.setdefault('recovery_history',[]).append({
            'event':'screening_rob_route_partition_recovery',
            'archive':str(archive.relative_to(self.out)),
        })
        self.save('running')
        print('[screening-rob] recovering prior all-unknown batch: rerunning only articles with a supported route',flush=True)
        return True
    def stage(self,name,command,outputs):
        if name=='extraction':
            robdir=self.out/'01_screening'
            self._recover_pre_partition_screening_rob()
            self.stage('screening_rob',[sys.executable,str(ROOT/'extensions/screening_bias.py'),'--request',str(self.out/'request.normalized.json'),'--screening-dir',str(robdir)],
                       [robdir/'risk_of_bias_assessments.json',robdir/'risk_of_bias_complete.json'])
        if name in ('analysis_check','analysis'):
            command=[sys.executable,str(ROOT/'extensions/analysis_worker.py'),'--request',str(self.out/'request.normalized.json'),'--input-dir',str(self.out/'04_analysis_input'),'--output-dir',str(self.out/'05_analysis'),'--extensions',str(self.out/'extensions.normalized.json'),'--recommendations',str(self.out/'01_screening/risk_of_bias_recommendations.json'),'--screening-rob',str(self.out/'01_screening/risk_of_bias_assessments.json')]
            if name=='analysis_check':command.append('--check')
            else:
                outputs=[*outputs,self.out/'05_analysis/extension_summary.json',self.out/'05_analysis/extension_files.json']
        result=super().stage(name,command,outputs)
        if name=='analysis' and (self.out/'05_analysis/extension_files.json').exists():
            for item in read(self.out/'05_analysis/extension_files.json')['files']:
                root=(self.out/'05_analysis').resolve();path=(root/item['path']).resolve()
                require(path.is_relative_to(root),'extension checkpoint path leaves its output directory')
                require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],'extension output checkpoint changed: '+item['path'])
        return result
    def run(self):
        try:
            # Keep the frozen stage implementations, but do not treat an
            # included study with no eligible *quantitative* row as a failed
            # extraction.  Such studies remain part of the review record and
            # are reported as qualitative-only evidence; only successfully
            # reviewed numeric rows reach the calculator.
            from trace_runtime.contracts import (
                accepted_articles, prepare_extraction,
                write_analysis_input,
            )
            from .handoff import evidence_handoff
            screening_dir=self.out/'01_screening'
            extract_dir=self.out/'02_extraction'
            review_dir=self.out/'03_review'
            inputs=self.out/'04_analysis_input'
            analysis=self.out/'05_analysis'

            self.stage('screening',self.worker('screen',screening_dir),[
                screening_dir/'screening_results.json',
                screening_dir/'risk_of_bias_recommendations.json',
                screening_dir/'screening_complete.json',
            ])
            screening=read(screening_dir/'screening_results.json')
            accepted_articles(self.r,screening)
            resolution_path=self.out/'screening_resolutions.json'
            if resolution_path.exists():
                resolutions=read(resolution_path)
                require(isinstance(resolutions,list),'screening_resolutions must be an array')
                by_id={x['candidate_id']:x for x in screening['decisions']};seen=set()
                for resolution in resolutions:
                    aid=resolution.get('article_id')
                    require(aid not in seen and aid in by_id,'unknown/duplicate screening resolution')
                    seen.add(aid)
                    require(by_id[aid]['final_status']=='Unsure','resolutions only apply to Unsure articles')
                    require(resolution.get('decision') in ('Include','Exclude') and resolution.get('reason') and resolution.get('reviewer'),
                            'resolution needs decision, reason, reviewer')
                    by_id[aid]['final_status']=resolution['decision'];by_id[aid]['human_resolution']=resolution
            queue=[row for row in screening['decisions'] if row['final_status']=='Unsure']
            write(self.out/'manual_review_queue.json',queue)
            if queue:
                self.save('needs_screening_review');return self.state
            included=accepted_articles(self.r,screening)
            write(screening_dir/'effective_screening_results.json',screening)
            selection_hash=digest(screening)
            if self.state.get('effective_screening_hash'):
                require(self.state['effective_screening_hash']==selection_hash,
                        'screening resolutions changed after downstream work started; use a new output directory')
            self.state['effective_screening_hash']=selection_hash
            if not included:
                self.save('no_included_articles');return self.state
            config=prepare_extraction(self.r,included,self.out/'extraction_config')
            self.stage('extraction',self.worker('extract',extract_dir,'--config',config),[
                extract_dir/'case_extraction_submission.json',
                extract_dir/'case_run_manifest.json',extract_dir/'extraction_complete.json',
            ])
            submission=read(extract_dir/'case_extraction_submission.json')
            if not submission['predictions']:
                self.save('no_extracted_evidence');return self.state
            extraction_manifest=read(extract_dir/'case_run_manifest.json')
            require(not extraction_manifest.get('failed_article_ids'),'one or more article extractions failed')
            predicted={row.get('article_id') for row in submission['predictions']}
            missing=sorted(set(included)-predicted)
            article_runs={row.get('article_id'):row for row in extraction_manifest.get('article_runs',[])}
            write(self.out/'extraction_review_queue.json',{
                'articles_without_predictions':missing,
                'article_runs':extraction_manifest.get('article_runs',[]),
            })
            write(self.out/'qualitative_only_evidence.json',{
                'schema_version':'qualitative_only_evidence/1.0',
                'case_id':self.r['case_id'],
                'policy':'Included studies without a source-supported quantitative row remain reported as qualitative-only and are excluded from quantitative synthesis.',
                'articles':[
                    {
                        'article_id':article_id,
                        'quantitative_status':'not_extractable_for_requested_analysis',
                        'issues':article_runs.get(article_id,{}).get('issues',[]),
                    }
                    for article_id in missing
                ],
            })
            self.stage('review',self.worker('review',review_dir,'--config',config,
                                             '--submission',extract_dir/'case_extraction_submission.json'),[
                review_dir/'candidate_evidence_review_structured.json',review_dir/'review_complete.json',
            ])
            review=read(review_dir/'candidate_evidence_review_structured.json')
            candidates,audit=evidence_handoff(self.r,screening,submission,review)
            audit['qualitative_only_evidence_path']='qualitative_only_evidence.json'
            audit['qualitative_only_article_ids']=missing
            write(self.out/'handoff_audit.json',audit)
            if not candidates['candidates']:
                self.save('no_retained_evidence');return self.state
            write_analysis_input(self.r,candidates,inputs)
            self.stage('analysis_check',[],[])
            self.stage('analysis',[],[analysis/'analysis_summary.json',analysis/'analysis_result.json'])
            summary=read(analysis/'analysis_summary.json')
            warnings=[row for row in audit['records'] if row['warnings'] or row['disposition']=='needs_further_review' or row['verification_status'] not in ('accepted',None)]
            write(self.out/'brain_result.json',{
                'schema_version':'brain_result/1.0','case_id':self.r['case_id'],
                'analysis_summary':summary,'analysis_result_path':'05_analysis/analysis_result.json',
                'handoff_audit_path':'handoff_audit.json',
                'qualitative_only_evidence_path':'qualitative_only_evidence.json',
                'upstream_warnings':warnings,
                'extraction_issues':[row for row in extraction_manifest.get('article_runs',[]) if row.get('issues')],
            })
            self.state['analysis_status']=summary.get('status')
            self.save('completed')
            state=self.state
            if state['status']=='completed':
                # A recovered run must not retain the transient failure which
                # triggered the recovery. Preserve it as history instead.
                previous_error=self.state.pop('error',None)
                if previous_error is not None:
                    self.state.setdefault('recovery_history',[]).append({
                        'event':'completed_after_recovery',
                        'previous_error':previous_error,
                    })
                    self.save('completed')
                summary=read(self.out/'05_analysis/extension_summary.json')
                brain=read(self.out/'brain_result.json')
                brain['extensions']=summary
                brain['extension_summary_path']='05_analysis/extension_summary.json'
                write(self.out/'brain_result.json',brain)
            return state
        except KeyboardInterrupt:
            # Match the frozen runner's resumable interruption contract.  The
            # external qualitative-evidence branch must never strand a live
            # run in ``running`` after the parent process is interrupted.
            from trace_runtime.pipeline import now
            for stage in self.state['stages'].values():
                if stage.get('status')=='running':
                    stage['status']='paused';stage['paused_at']=now()
            self.save('paused')
            return self.state
        except Exception as exc:
            # Preserve the frozen failure manifest so a user can distinguish a
            # stopped process from a running calculation and resume only when
            # the completed stage checkpoints remain valid.
            for stage in self.state['stages'].values():
                if stage.get('status')=='running':
                    stage['status']='failed'
            self.state['error']={'type':type(exc).__name__,'message':str(exc)}
            self.save('failed')
            raise
        finally:
            self.close_dashboard_monitor()


def run(request,output,settings,env_file=None,resume=False,preparation_environment=None,
        workers=4,retry_deferred=False,retry_analysis=False,rebuild_handoff=False,reuse_screening_from=None,
        reuse_legacy_from=None):
    from .provider import runtime
    from .document_preparation import prepare_request
    # Operational concurrency may change without changing scientific inputs.
    active_workers = workers
    prior_request=Path(output).resolve()/'request.normalized.json'
    if retry_analysis:
        require(resume, '--retry-analysis requires --resume')
        request = frozen_analysis_request_for_retry(request, output)
    require(not rebuild_handoff or retry_analysis,
            '--rebuild-handoff requires --retry-analysis')
    if resume and prior_request.is_file():
        workers=read(prior_request).get('options',{}).get('max_workers',workers)
    require(settings['bias']['mode']=='draft' and settings['bias'].get('auto_sensitivity'), 'Pipeline requires screening RoB and high-risk sensitivity')
    request['options']['route_rob']=True
    request['options']['max_workers']=workers
    preparation_environment=dict(preparation_environment or {})
    preparation_environment['TRACE_WORKERS']=str(active_workers)
    if retry_analysis:
        # Retain the request exactly as the finished screening/extraction/review
        # stages saw it. runtime() is used only to construct the child
        # environment and current provider profile; its derived request would
        # otherwise alter the frozen input fingerprint.
        preparation = (read(Path(output).resolve()/'document_preparation.json')
                       if (Path(output).resolve()/'document_preparation.json').is_file()
                       else {'status': 'reused_frozen_document_preparation'})
        _, environment, profile = runtime(request, env_file)
    else:
        request, preparation = prepare_request(request, preparation_environment)
        request,environment,profile=runtime(request,env_file)
    environment['TRACE_ACTIVE_WORKERS']=str(active_workers)
    environment['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    if retry_deferred:
        environment['TRACE_RETRY_DEFERRED_524']='1'
    # COMPATIBLE has shown provider-side timeouts when several full-text RoB calls are
    # in flight.  Keep all ordinary stages at the requested worker count while
    # retaining two concurrent RoB assessments for a reliable resumable stage.
    environment['TRACE_ROB_WORKERS']=str(max(1,min(int(active_workers),2)))
    print('[preflight] Checking extraction, analysis components and R/metafor runtime', flush=True)
    preflight_runtime(request,environment)
    print('[preflight] Runtime checks completed', flush=True)
    if retry_analysis:
        rebind_runner_input_for_analysis_retry(request, output)
    legacy_inspection = None
    if reuse_legacy_from is not None:
        require(not resume and reuse_screening_from is None,
                '--reuse-legacy-from requires a fresh output and cannot combine with --reuse-screening-from')
        from .legacy_run_reuse import inspect_legacy_run
        legacy_inspection = inspect_legacy_run(request, reuse_legacy_from)
    runner=ExtensionRunner(request,output,environment,settings,resume,profile,
                           retry_analysis=retry_analysis,rebuild_handoff=rebuild_handoff)
    from .common import write
    write(runner.out/'document_preparation.json',preparation)
    if legacy_inspection is not None:
        from .legacy_run_reuse import import_legacy_run
        import_legacy_run(request, runner, legacy_inspection)
    if reuse_screening_from is not None:
        if resume:
            require(False, '--reuse-screening-from is only valid with a new output directory')
        source=Path(reuse_screening_from).resolve()
        from .stage_inputs import validate_screening_reuse
        validate_screening_reuse(request, source)
        require(source.is_dir() and (source/'pipeline_manifest.json').is_file(),
                'reuse screening source must be a completed Graph run directory')
        source_manifest=read(source/'pipeline_manifest.json')
        require(source_manifest.get('case_id')==request['case_id'],
                'reuse screening source case_id does not match request')
        stages=source_manifest.get('stages',{})
        require(stages.get('screening',{}).get('status')=='completed'
                and stages.get('screening_rob',{}).get('status')=='completed',
                'reuse screening source must have completed screening and screening-RoB stages')
        source_screening=source/'01_screening'
        require((source_screening/'screening_results.json').is_file()
                and (source_screening/'effective_screening_results.json').is_file()
                and (source_screening/'risk_of_bias_assessments.json').is_file(),
                'reuse screening source is missing effective screening/RoB artifacts')
        shutil.copytree(source_screening,runner.out/'01_screening',dirs_exist_ok=True)
        if (source/'screening_resolutions.json').is_file():
            shutil.copy2(source/'screening_resolutions.json',runner.out/'screening_resolutions.json')
        for name, outputs in {
            'screening': ('01_screening/screening_results.json','01_screening/risk_of_bias_recommendations.json','01_screening/screening_complete.json'),
            'screening_rob': ('01_screening/risk_of_bias_assessments.json','01_screening/risk_of_bias_complete.json'),
        }.items():
            checks={}
            for rel in outputs:
                path=runner.out/rel
                require(path.is_file(), 'copied screening artifact missing: '+rel)
                checks[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
            runner.state['stages'][name]={'status':'completed','completed_at':source_manifest['stages'][name].get('completed_at'),
                                          'outputs':checks,'reused_from':str(source)}
        runner.state['effective_screening_hash']=digest(read(runner.out/'01_screening/effective_screening_results.json'))
        runner.save('running')
        print(f'[pipeline] reused screening and screening-RoB from {source}',flush=True)
    return runner.run()
