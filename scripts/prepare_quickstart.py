"""Prepare a small, source-bound influenza example from public PMC JATS XML.

Standard library only. Does not call a model, extract outcome values, or read Gold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples' / 'quickstart_influenza'
MAX_BYTES = 20 * 1024 * 1024


def text(node):
    return re.sub(r'\s+', ' ', ''.join(node.itertext())).strip() if node is not None else ''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_checked(path, data):
    """A rerun may reuse identical files; never overwrite a user's changed input."""
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Existing file differs: {path}; choose a new output directory.')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def verify_xml(data, source):
    if digest(data) != source['xml_sha256']:
        raise ValueError(f"{source['pmcid']}: source SHA256 differs from the verified version")
    root = ET.fromstring(data)
    # Normalize JATS namespaces for deterministic section/table handling.
    for node in root.iter():
        node.tag = node.tag.rsplit('}', 1)[-1]
    ids = {node.attrib.get('pub-id-type'): text(node)
           for node in root.findall('./front/article-meta/article-id')}
    if ids.get('pmcid') != source['pmcid'] or ids.get('doi', '').lower() != source['doi'].lower():
        raise ValueError(f"{source['pmcid']}: article identity does not match the source manifest")
    if root.find('body') is None:
        raise ValueError(f"{source['pmcid']}: full article body is missing")
    return root


def render_article(root, source):
    """Keep complete main-text paragraphs, table structure, captions and notes.

    JATS has no PDF page numbers: blocks intentionally do not invent page indices.
    Bibliography is omitted from the extraction representation; original XML stays.
    """
    blocks = []

    def add(value, level=None):
        if value:
            b = {'type': 'text', 'text': value}
            if level is not None:
                b['text_level'] = level
            blocks.append(b)

    add(source['title'], 1)
    add(f"DOI: {source['doi']}; PMCID: {source['pmcid']}; source: {source['article_url']}")
    meta = root.find('./front/article-meta')
    for abstract in meta.findall('abstract'):
        add('Abstract', 2)
        add(text(abstract))

    exported_tables, exported_figures = set(), set()

    def table(node):
        if id(node) in exported_tables:
            return
        exported_tables.add(id(node))
        body = node.find('.//table')
        if body is None:
            raise ValueError('Image-only table encountered; use the original source for manual preparation.')
        blocks.append({'type': 'table', 'table_caption': [text(node.find('label')), text(node.find('caption'))],
                       'table_body': ET.tostring(body, encoding='unicode'),
                       'table_footnote': [text(node.find('table-wrap-foot'))]})

    def figure(node):
        if id(node) not in exported_figures:
            exported_figures.add(id(node))
            add(' '.join(filter(None, [text(node.find('label')), text(node.find('caption'))])))

    def walk(node):
        if node.tag == 'table-wrap':
            table(node)
        elif node.tag == 'fig':
            figure(node)
        elif node.tag == 'title':
            add(text(node), 2)
        elif node.tag in ('p', 'list-item', 'disp-quote', 'boxed-text'):
            add(text(node))
            for nested in node.iter('table-wrap'):
                table(nested)
            for nested in node.iter('fig'):
                figure(nested)
        else:
            for child in node:
                walk(child)

    walk(root.find('body'))
    # Some JATS papers place tables/figure captions outside the main body.
    for node in root.iter('table-wrap'):
        table(node)
    for node in root.iter('fig'):
        figure(node)
    add('Source license', 2)
    add(' '.join(source['license_text']))
    parts = []
    for block in blocks:
        if block['type'] == 'table':
            parts.extend([*block['table_caption'], block['table_body'], *block['table_footnote']])
        else:
            parts.append(('#' * block['text_level'] + ' ' if block.get('text_level') else '') + block['text'])
    abstract = ' '.join(text(a) for a in meta.findall('abstract'))
    return ('\n\n'.join(p for p in parts if p) + '\n').encode(), blocks, abstract


def prepare(output, *, cache=None, offline=False, include_author_manuscript=False):
    manifest = json.loads((EXAMPLE / 'sources.json').read_text())
    request = json.loads((EXAMPLE / 'request.template.json').read_text())
    output = Path(output).resolve()
    cache = Path(cache).resolve() if cache else output / 'sources'
    selected = [s for s in manifest['sources'] if s['default'] or include_author_manuscript]
    prepared = []
    for source in selected:
        cached = cache / (source['pmcid'] + '.xml')
        if cached.is_file():
            data = cached.read_bytes()
        elif offline:
            raise ValueError(f'Missing cached full text: {cached}')
        else:
            req = Request(source['xml_url'], headers={'User-Agent': 'TRACE-Quickstart/1.0'})
            with urlopen(req, timeout=45) as response:
                data = response.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ValueError('Full text exceeds the download size limit')
        root = verify_xml(data, source)
        markdown, blocks, abstract = render_article(root, source)
        stem = 'documents/' + source['pmcid']
        prepared.append((source, data, markdown, blocks))
        request['documents'].append({
            'article_id': source['pmcid'], 'document_id': source['pmcid'], 'study_id': source['study_id'],
            'role': 'candidate_article', 'title': source['title'], 'abstract': abstract,
            'citation': {'doi': source['doi'], 'pmid': str(source['pmid']), 'pmcid': source['pmcid']},
            'source_representation': 'pmc_jats_xml',
            'paths': {'markdown': stem + '.md', 'content_list': stem + '.content_list.json'},
        })
    provenance = {**manifest, 'sources': selected, 'representation': 'JATS main text, structured tables and figure captions; no PDF pagination; no external supplementary files',
                  'generated_files': {}}
    files = {}
    for source, xml, markdown, blocks in prepared:
        pmcid = source['pmcid']
        files['sources/' + pmcid + '.xml'] = xml
        files['documents/' + pmcid + '.md'] = markdown
        files['documents/' + pmcid + '.content_list.json'] = json_bytes(blocks)
    files['request.json'] = json_bytes(request)
    if include_author_manuscript:
        added = {source['pmcid'] for source in selected if not source['default']}
        files['update.delta.json'] = json_bytes({
            'schema_version': 'trace_update/1.0',
            'documents': [doc for doc in request['documents'] if doc['article_id'] in added],
        })
    provenance['generated_files'] = {name: digest(data) for name, data in files.items()}
    files['SOURCE_MANIFEST.json'] = json_bytes(provenance)
    # Validate every destination before writing, including switching between 2/3 studies.
    for name, data in files.items():
        p = output / name
        if p.exists() and p.read_bytes() != data:
            raise ValueError(f'Existing file differs: {p}; choose a new output directory.')
    for name, data in files.items():
        write_checked(output / name, data)
    return {'case_id': request['case_id'], 'articles': len(selected),
            'request': str(output / 'request.json'), 'statistical_method_policy': 'fixed',
            'scope': manifest['scope'], 'model_calls': 0}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True, help='A new writable input directory')
    p.add_argument('--cache', type=Path, help='Existing PMCID.xml files; each SHA256 is verified')
    p.add_argument('--offline', action='store_true', help='Require cached XML, with no network requests')
    p.add_argument('--include-author-manuscript', action='store_true', help='Add the publicly accessible 2019 phase-3 author manuscript (not marked OA by PMC)')
    args = p.parse_args(argv)
    try:
        result = prepare(args.output, cache=args.cache, offline=args.offline,
                         include_author_manuscript=args.include_author_manuscript)
    except (OSError, ValueError, ET.ParseError) as exc:
        p.exit(1, f'Quickstart preparation failed: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
