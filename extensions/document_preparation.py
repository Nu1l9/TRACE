"""Prepare PDF-only inputs without changing the frozen pipeline modules.

The request contract is deliberately simple: an already supplied Markdown file
is authoritative.  Only a PDF-only document is sent to MinerU, and its cached
Markdown becomes the representation used by screening and extraction.
"""
from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
import time

from requests.exceptions import RequestException

from .common import ContractError, require


def _cache_root(environment: dict[str, str]) -> Path:
    configured = environment.get("META_PIPELINE_DOCUMENT_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / "Library/Application Support/meta_pipeline_sources/prepared_mineru").resolve()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shared_index(environment):
    root = Path(environment.get('META_PIPELINE_SHARED_MINERU_ROOT') or
                Path.home() / 'Library/Application Support/meta_pipeline_shared/mineru_22_cases').expanduser().resolve()
    manifest = root / 'dataset_manifest.json'
    if not manifest.is_file():
        return root, {}, {}
    payload = json.loads(manifest.read_text(encoding='utf-8'))
    checks = json.loads((root / 'file_checksums.json').read_text(encoding='utf-8'))
    by_hash = {}
    for row in payload['documents']:
        by_hash.setdefault(row['source_sha256'], []).append(row)
    return root, by_hash, {r['path']: r['sha256'] for r in checks['files']}


def _primary_paths(row, checksums):
    root = Path(row['paths']['pdf']).parent / 'mineru'
    markdowns = row.get('markdown_files', [row['paths']['markdown']])
    primary = [p for p in markdowns if Path(p).parent == root]
    if not primary:
        primary = [row['paths']['markdown']]
    full = [p for p in primary if Path(p).name == 'full.md']
    require(len(primary) == 1 or len(full) == 1, 'shared MinerU: ambiguous primary article')
    markdown = full[0] if full else primary[0]
    parent = str(Path(markdown).parent)
    content = [p for p in checksums if p.endswith('_content_list.json')
               and p.rsplit('/', 1)[0] == parent]
    require(len(content) == 1, 'shared MinerU: primary article requires one matching content_list')
    return {**row['paths'], 'markdown': markdown, 'content_list': content[0]}


def _paired_markdown(row):
    """Pair the primary content list with Markdown in its own output folder.

    Supplementary outputs can contain their own full.md in nested folders.
    Never prefer those over a primary article with a publisher-specific name.
    """
    content_parent = Path(row['paths']['content_list']).parent
    candidates = sorted({p for p in row.get('markdown_files', [row['paths']['markdown']])
                         if Path(p).parent == content_parent})
    require(bool(candidates), 'shared MinerU: no Markdown paired with primary content_list')
    full = [p for p in candidates if Path(p).name == 'full.md']
    require(len(candidates) == 1 or len(full) == 1,
            'shared MinerU: ambiguous primary Markdown; inspect corpus index')
    return full[0] if full else candidates[0]


def _retry_network_conversion(convert, *, attempts: int, retry_seconds: float, status):
    """Retry transient network failures without changing the frozen MinerU client."""
    for attempt in range(1, attempts + 1):
        try:
            return convert()
        except (RequestException, TimeoutError) as exc:
            if attempt == attempts:
                raise
            delay = retry_seconds * attempt
            status(
                f"temporary MinerU network error ({type(exc).__name__}); "
                f"retrying upload/conversion {attempt + 1}/{attempts} in {delay:g}s"
            )
            time.sleep(delay)


def _convert_with_existing_mineru(pdf: Path, article_id: str, environment: dict[str, str]) -> str:
    """Use the project's established MinerU Precision API wrapper.

    The wrapper reads MINERU_TOKEN only at call time, uploads the source PDF to
    MinerU, polls the result, and keeps its own digest-keyed Markdown cache.  It
    never writes that token into the request, cache provenance, or run output.
    """
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from document_io.mineru import MinerUError, convert_document_to_markdown
        attempts = max(1, int(environment.get("MINERU_CONVERSION_ATTEMPTS", "4")))
        retry_seconds = max(0.0, float(environment.get("MINERU_RETRY_SECONDS", "3")))
        status = lambda message: print(f"[document-preparation] {article_id}: {message}", flush=True)
        return _retry_network_conversion(
            lambda: convert_document_to_markdown(
                pdf,
                source_id=article_id,
                status=status,
                timeout_seconds=float(environment.get("MINERU_TIMEOUT_SECONDS", "600")),
            ),
            attempts=attempts,
            retry_seconds=retry_seconds,
            status=status,
        )
    except ImportError as exc:
        raise ContractError("The project MinerU conversion wrapper is unavailable") from exc
    except MinerUError as exc:
        raise ContractError(f"{article_id}: MinerU conversion failed: {exc}") from exc


def prepare_request(request: dict, environment: dict[str, str] | None = None, *, allow_conversion: bool = True, converter=None) -> tuple[dict, dict]:
    """Return a request where every PDF-only document has MinerU Markdown.

    This function intentionally never runs MinerU when a request already has
    Markdown, even if it also registers the source PDF.  Cache entries are
    keyed by the PDF digest, so a changed PDF cannot silently reuse a prior
    conversion.
    """
    environment = dict(os.environ if environment is None else environment)
    result = deepcopy(request)
    cache_root = _cache_root(environment)
    report = {"schema_version": "document_preparation/1.0", "policy": "prefer_registered_markdown_else_mineru_pdf", "documents": []}
    documents = result.get("documents", [])
    workers = min(max(1, int(environment.get('META_PIPELINE_WORKERS', '4'))), max(1, len(documents)))
    @lru_cache(maxsize=1)
    def shared_index():
        return _shared_index(environment)
    if allow_conversion:
        print(f"\n[progress 1/7] Document preparation: checking {len(documents)} articles", flush=True)
        print(f"[progress 1/7] Document preparation: parallel workers {workers}", flush=True)

    def prepare_one(index, document):
        document = deepcopy(document)
        article_id = document.get("article_id", "unknown")
        paths = document.setdefault("paths", {})
        markdown = Path(paths["markdown"]).resolve() if paths.get("markdown") else None
        pdf = Path(paths["pdf"]).resolve() if paths.get("pdf") else None
        if markdown:
            require(markdown.is_file(), f"{article_id}: missing markdown: {markdown}")
            paths["markdown"] = str(markdown)
            if allow_conversion:
                print(f"[progress 1/7] Document {index}/{len(documents)} {article_id}: using existing Markdown", flush=True)
            return document, {"article_id": article_id, "status": "used_registered_markdown", "markdown": str(markdown)}
        require(pdf and pdf.is_file(), f"{article_id}: needs Markdown or a readable PDF")
        pdf_sha = _digest(pdf)
        cache_dir = cache_root / article_id / pdf_sha
        provenance_path = cache_dir / "provenance.json"
        provenance = {}
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        cached = Path(provenance["markdown"]).resolve() if provenance.get("pdf_sha256") == pdf_sha and provenance.get("markdown") else None
        cache_compatible = True
        if cached and cached.is_file() and provenance.get('shared_content_list'):
            _, index_rows, index_checks = shared_index()
            matched = index_rows.get(pdf_sha, [])
            if matched:
                item = next((r for r in matched if r['case_id'] == request.get('case_id')), matched[0])
                primary = _primary_paths(item, index_checks)
                cache_compatible = (_digest(cached) == index_checks[primary['markdown']]
                                    and Path(provenance['shared_content_list']).is_file()
                                    and _digest(Path(provenance['shared_content_list'])) == index_checks[primary['content_list']])
        if cached and cached.is_file() and cache_compatible:
            paths["markdown"] = str(cached)
            if provenance.get('shared_content_list'):
                content = Path(provenance['shared_content_list'])
                require(content.is_file(), f'{article_id}: shared cache content_list missing')
                paths.setdefault('content_list', str(content))
            if allow_conversion:
                print(f"[progress 1/7] Document {index}/{len(documents)} {article_id}: using cached MinerU output", flush=True)
            return document, {"article_id": article_id, "status": "used_cached_mineru_markdown", "pdf": str(pdf), "markdown": str(cached)}
        shared_root, shared, checksums = shared_index()
        matches = shared.get(pdf_sha, [])
        if matches:
            # Match PDF bytes, never sample labels or filenames. Prefer the
            # registered case when one PDF occurs in multiple benchmark cases.
            row = next((r for r in matches if r['case_id'] == request.get('case_id')), matches[0])
            primary_paths = _primary_paths(row, checksums)
            sources = {}
            for kind in ('pdf', 'markdown', 'content_list'):
                relative = primary_paths[kind]
                source = (shared_root / relative).resolve()
                require(source.is_relative_to(shared_root) and source.is_file(), f'{article_id}: invalid shared MinerU {kind}')
                expected = checksums.get(relative)
                require(expected and _digest(source) == expected, f'{article_id}: shared MinerU {kind} checksum mismatch')
                sources[kind] = source
            require(_digest(sources['pdf']) == pdf_sha, f'{article_id}: shared PDF mismatch')
            if allow_conversion:
                # Neutral paths prevent positive/negative folder labels from
                # entering downstream model inputs. Keep relative image links.
                source_dir = sources['markdown'].parent
                require(sources['content_list'].is_relative_to(source_dir), f'{article_id}: unsupported shared layout')
                destination = cache_dir / 'shared_primary_structured'
                shutil.copytree(source_dir, destination, dirs_exist_ok=True)
                markdown = destination / sources['markdown'].relative_to(source_dir)
                content = destination / sources['content_list'].relative_to(source_dir)
                provenance = {'schema_version':'mineru_conversion/1.0', 'article_id':article_id,
                              'pdf':str(pdf), 'pdf_sha256':pdf_sha, 'markdown':str(markdown),
                              'shared_content_list':str(content), 'provider':'verified shared MinerU corpus'}
                provenance_path.write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
                print(f'[progress 1/7] Document {index}/{len(documents)} {article_id}: shared MinerU cache hit; conversion skipped', flush=True)
            else:
                markdown, content = sources['markdown'], sources['content_list']
            paths['markdown'] = str(markdown)
            paths.setdefault('content_list', str(content))
            return document, {'article_id':article_id, 'status':'used_shared_mineru_markdown',
                              'pdf':str(pdf), 'markdown':str(markdown), 'pdf_sha256':pdf_sha}
        if not allow_conversion:
            return document, {"article_id": article_id, "status": "needs_mineru_conversion", "pdf": str(pdf)}
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[progress 1/7] Document {index}/{len(documents)} {article_id}: PDF → MinerU Markdown", flush=True)
        markdown_text = (converter or _convert_with_existing_mineru)(pdf, article_id, environment)
        converted = cache_dir / "full.md"
        converted.write_text(markdown_text, encoding="utf-8")
        provenance = {"schema_version": "mineru_conversion/1.0", "article_id": article_id, "pdf": str(pdf), "pdf_sha256": pdf_sha,
                      "markdown": str(converted), "provider": "MinerU precision extraction API"}
        provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths["markdown"] = str(converted)
        print(f"[progress 1/7] Document {index}/{len(documents)} {article_id}: conversion completed", flush=True)
        return document, {"article_id": article_id, "status": "converted_with_mineru", "pdf": str(pdf), "markdown": str(converted)}

    # MinerU reads MINERU_TOKEN from process environment.  Set it once for this
    # child run so independent uploads can progress concurrently, then restore it.
    old_token = os.environ.get('MINERU_TOKEN')
    supplied_token = environment.get('MINERU_TOKEN')
    if supplied_token:
        os.environ['MINERU_TOKEN'] = supplied_token
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='prepare') as executor:
            prepared = list(executor.map(lambda pair: prepare_one(*pair), enumerate(documents, start=1)))
    finally:
        if supplied_token:
            if old_token is None:
                os.environ.pop('MINERU_TOKEN', None)
            else:
                os.environ['MINERU_TOKEN'] = old_token
    result['documents'] = [document for document, _ in prepared]
    report['documents'] = [item for _, item in prepared]
    if allow_conversion:
        print(f"[progress 1/7] Document preparation: completed", flush=True)
    return result, report
