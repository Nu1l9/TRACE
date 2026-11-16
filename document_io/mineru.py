"""Small, secret-free client for MinerU's precision extraction API.

Document or image bytes are sent to MinerU when conversion is requested. The API token is read at
runtime from ``MINERU_TOKEN`` and is deliberately never stored in this repository,
the MinerU cache, or a run result.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Callable

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


API_ROOT = "https://mineru.net/api/v4"
CACHE_ROOT = PROJECT_ROOT / ".cache" / "mineru"


class MinerUError(RuntimeError):
    """Raised when MinerU cannot produce a usable Markdown result."""


def _mineru_token() -> str:
    """Read a JWT safely, tolerating quotation marks pasted around the value."""
    token = os.getenv("MINERU_TOKEN", "").strip()
    # Chat applications often wrap a pasted secret in typographic quotation marks.
    # Those are not valid in an HTTP Authorization header.
    token = token.strip("'\"“”‘’")
    if not token:
        raise MinerUError("Set MINERU_TOKEN before running the MinerU document parser.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", token):
        raise MinerUError(
            "MINERU_TOKEN contains a non-ASCII or invalid character. Paste the raw token "
            "only, without quotation marks or a Bearer prefix."
        )
    return token


def _api_json(response: requests.Response, operation: str) -> dict:
    try:
        payload = response.json()
    except ValueError as exc:
        raise MinerUError(f"MinerU {operation} returned non-JSON HTTP {response.status_code}.") from exc
    if response.status_code != 200 or payload.get("code") != 0:
        raise MinerUError(
            f"MinerU {operation} failed (HTTP {response.status_code}, "
            f"code={payload.get('code')}): {payload.get('msg', 'unknown error')}"
        )
    return payload


def _normalise_png(image_bytes: bytes) -> bytes:
    """Use a stable, extension-matching upload format for the signed upload API."""
    with Image.open(BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB")
        output = BytesIO()
        rgb.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _load_cached_markdown(digest: str) -> str | None:
    path = CACHE_ROOT / f"{digest}.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _write_cache(digest: str, markdown: str) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    (CACHE_ROOT / f"{digest}.md").write_text(markdown, encoding="utf-8")
    (CACHE_ROOT / f"{digest}.json").write_text(
        json.dumps(
            {
                "provider": "MinerU precision API",
                "model_version": "vlm",
                "enable_table": True,
                "language": "en",
                "source_sha256": digest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _download_archive(url: str, *, attempts: int = 4) -> bytes:
    """Fetch MinerU's CDN archive, tolerating direct-DNS and proxy differences."""
    error: requests.RequestException | None = None
    for attempt in range(attempts):
        for trust_env in (False, True):
            session = requests.Session()
            session.trust_env = trust_env
            try:
                response = session.get(url, timeout=90)
                response.raise_for_status()
                return response.content
            except requests.RequestException as exc:
                error = exc
        # On some macOS/Python combinations the MinerU CDN closes TLS sessions
        # in a way that requests/OpenSSL rejects.  The system curl stack handles
        # that CDN reliably, so use it as a transport fallback for the *same*
        # signed result URL rather than submitting and billing a new parse task.
        with tempfile.TemporaryDirectory(prefix="mineru-download-") as directory:
            archive_path = Path(directory) / "result.zip"
            curl = subprocess.run(
                [
                    "curl", "--fail", "--location", "--silent", "--show-error",
                    "--retry", "4", "--retry-all-errors", "--connect-timeout", "30",
                    "--max-time", "300", "--output", str(archive_path), url,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if curl.returncode == 0 and archive_path.is_file():
                return archive_path.read_bytes()
        if attempt + 1 < attempts:
            time.sleep(2 * (attempt + 1))
    raise MinerUError(
        f"Could not download MinerU result archive after {attempts} attempts "
        f"with requests or curl: {type(error).__name__ if error else 'download error'}"
    )


def _request_document_archive(
    source_bytes: bytes,
    *,
    source_id: str,
    suffix: str,
    source_label: str,
    status: Callable[[str], None] | None = None,
    poll_interval_seconds: float = 2.0,
    timeout_seconds: float = 600.0,
) -> tuple[bytes, str]:
    """Run one MinerU task and return its complete result ZIP and batch ID."""
    digest = hashlib.sha256(source_bytes).hexdigest()
    token = _mineru_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    data_id = f"manalyzer-{source_id}-{digest[:12]}"[:128]

    if status:
        status("requesting signed MinerU upload URL")
    request = requests.post(
        f"{API_ROOT}/file-urls/batch",
        headers=headers,
        json={
            "files": [{"name": f"{source_id}{suffix}", "data_id": data_id, "is_ocr": True}],
            "model_version": "vlm",
            "enable_table": True,
            "enable_formula": False,
            "language": "en",
        },
        timeout=30,
    )
    payload = _api_json(request, "signed-upload request")
    batch_id = payload["data"].get("batch_id")
    urls = payload["data"].get("file_urls") or []
    if not batch_id or len(urls) != 1:
        raise MinerUError("MinerU signed-upload response did not include one batch ID and one upload URL.")

    if status:
        status(f"uploading {source_label}")
    upload = requests.put(urls[0], data=source_bytes, timeout=120)
    if upload.status_code not in {200, 201, 204}:
        raise MinerUError(f"MinerU upload failed with HTTP {upload.status_code}.")

    deadline = time.monotonic() + timeout_seconds
    last_state = None
    while time.monotonic() < deadline:
        result = requests.get(
            f"{API_ROOT}/extract-results/batch/{batch_id}",
            headers=headers,
            timeout=30,
        )
        extract_result = _api_json(result, "result polling")["data"].get("extract_result") or []
        if len(extract_result) != 1:
            raise MinerUError("MinerU result did not contain the uploaded document task.")
        task = extract_result[0]
        task_state = task.get("state")
        if task_state != last_state and status:
            status(f"MinerU task state={task_state}")
            last_state = task_state
        if task_state == "done":
            zip_url = task.get("full_zip_url")
            if not zip_url:
                raise MinerUError("MinerU completed without a result archive URL.")
            if status:
                status("downloading complete MinerU result archive")
            return _download_archive(zip_url), str(batch_id)
        if task_state == "failed":
            raise MinerUError(f"MinerU parsing failed: {task.get('err_msg') or 'unknown error'}")
        time.sleep(poll_interval_seconds)
    raise MinerUError(f"MinerU did not finish within {timeout_seconds:.0f} seconds.")


def _safe_extract_archive(archive_bytes: bytes, destination: Path) -> list[Path]:
    """Extract a MinerU ZIP without allowing members to escape destination."""
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
            members = archive.infolist()
            for member in members:
                relative = Path(member.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise MinerUError(f"Unsafe path in MinerU result archive: {member.filename}")
            archive.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise MinerUError("MinerU returned an unreadable result archive.") from exc
    return sorted(path for path in destination.rglob("*") if path.is_file())


def convert_document_to_bundle(
    path: Path,
    output_dir: Path,
    *,
    source_id: str,
    status: Callable[[str], None] | None = None,
    poll_interval_seconds: float = 2.0,
    timeout_seconds: float = 600.0,
) -> dict:
    """Persist MinerU's complete native result for one document.

    The bundle keeps the downloaded ZIP and its unmodified extracted tree.  A
    manifest records the exact Markdown, content-list, layout and image paths so
    downstream systems can add adapters without parsing provider-specific names.
    """
    path = path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix not in {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}:
        raise MinerUError(f"Unsupported original-document extension: {suffix or '[none]'}")
    if not path.is_file():
        raise MinerUError(f"Original document does not exist: {path}")

    source_bytes = path.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    manifest_path = output_dir / "manifest.json"
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        markdown = output_dir / existing["primary_markdown"]
        content_list = existing.get("primary_content_list")
        if (
            existing.get("source_sha256") == digest
            and markdown.is_file()
            and (not content_list or (output_dir / content_list).is_file())
            and (output_dir / "result.zip").is_file()
        ):
            if status:
                status(f"complete bundle cache hit ({digest[:10]})")
            return existing
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError):
        pass

    archive_bytes, batch_id = _request_document_archive(
        source_bytes,
        source_id=source_id,
        suffix=suffix,
        source_label=f"original document ({path.name})",
        status=status,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    artifact_root = temporary / "artifact"
    try:
        files = _safe_extract_archive(archive_bytes, artifact_root)
        markdowns = [item for item in files if item.name == "full.md"]
        if not markdowns:
            raise MinerUError("MinerU result archive did not contain full.md.")
        content_lists = [item for item in files if item.name.endswith("content_list.json")]
        layouts = [item for item in files if "layout" in item.name.lower() and item.suffix.lower() == ".json"]
        image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
        images = [item for item in files if item.suffix.lower() in image_suffixes]
        temporary.mkdir(parents=True, exist_ok=True)
        (temporary / "result.zip").write_bytes(archive_bytes)

        def relative(item: Path) -> str:
            return str(item.relative_to(temporary))

        inventory = []
        for item in files:
            inventory.append({
                "path": relative(item),
                "bytes": item.stat().st_size,
                "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
            })
        manifest = {
            "schema_version": "mineru_native_bundle/1.0",
            "provider": "MinerU precision API",
            "model_version": "vlm",
            "source_id": source_id,
            "source_path": str(path),
            "source_sha256": digest,
            "batch_id": batch_id,
            "primary_markdown": relative(markdowns[0]),
            "primary_content_list": relative(content_lists[0]) if content_lists else None,
            "layout_files": [relative(item) for item in layouts],
            "image_files": [relative(item) for item in images],
            "counts": {
                "files": len(files),
                "content_lists": len(content_lists),
                "layouts": len(layouts),
                "images": len(images),
            },
            "files": inventory,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        markdown_text = markdowns[0].read_text(encoding="utf-8")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary.replace(output_dir)
        _write_cache(digest, markdown_text)
        if status:
            status(
                f"saved complete bundle: files={len(files)} images={len(images)} "
                f"content_lists={len(content_lists)}"
            )
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _convert_bytes_to_markdown(
    source_bytes: bytes,
    *,
    source_id: str,
    suffix: str,
    source_label: str,
    status: Callable[[str], None] | None = None,
    poll_interval_seconds: float = 2.0,
    timeout_seconds: float = 300.0,
) -> str:
    """Upload one supported document and return MinerU's ``full.md`` result.

    MinerU's documented local-file flow is: request a signed URL, PUT the bytes,
    then poll the batch result endpoint until its state is ``done``.
    """
    digest = hashlib.sha256(source_bytes).hexdigest()
    cached = _load_cached_markdown(digest)
    if cached is not None:
        if status:
            status(f"MinerU cache hit ({digest[:10]})")
        return cached

    token = _mineru_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    data_id = f"manalyzer-{source_id}-{digest[:12]}"[:128]

    if status:
        status("requesting signed MinerU upload URL")
    request = requests.post(
        f"{API_ROOT}/file-urls/batch",
        headers=headers,
        json={
            "files": [{"name": f"{source_id}{suffix}", "data_id": data_id, "is_ocr": True}],
            "model_version": "vlm",
            "enable_table": True,
            "enable_formula": False,
            "language": "en",
        },
        timeout=30,
    )
    payload = _api_json(request, "signed-upload request")
    batch_id = payload["data"].get("batch_id")
    urls = payload["data"].get("file_urls") or []
    if not batch_id or len(urls) != 1:
        raise MinerUError("MinerU signed-upload response did not include one batch ID and one upload URL.")

    if status:
        status(f"uploading {source_label}")
    upload = requests.put(urls[0], data=source_bytes, timeout=60)
    if upload.status_code not in {200, 201, 204}:
        raise MinerUError(f"MinerU upload failed with HTTP {upload.status_code}.")

    deadline = time.monotonic() + timeout_seconds
    last_state = None
    while time.monotonic() < deadline:
        result = requests.get(
            f"{API_ROOT}/extract-results/batch/{batch_id}",
            headers=headers,
            timeout=30,
        )
        extract_result = _api_json(result, "result polling")["data"].get("extract_result") or []
        if len(extract_result) != 1:
            raise MinerUError("MinerU result did not contain the uploaded image task.")
        task = extract_result[0]
        task_state = task.get("state")
        if task_state != last_state and status:
            status(f"MinerU task state={task_state}")
            last_state = task_state
        if task_state == "done":
            zip_url = task.get("full_zip_url")
            if not zip_url:
                raise MinerUError("MinerU completed without a result archive URL.")
            archive_bytes = _download_archive(zip_url)
            try:
                with zipfile.ZipFile(BytesIO(archive_bytes)) as zip_file:
                    markdown_name = next(
                        name for name in zip_file.namelist() if name.endswith("full.md")
                    )
                    markdown = zip_file.read(markdown_name).decode("utf-8")
            except (zipfile.BadZipFile, StopIteration, UnicodeDecodeError) as exc:
                raise MinerUError("MinerU result archive did not contain a readable full.md.") from exc
            _write_cache(digest, markdown)
            return markdown
        if task_state == "failed":
            raise MinerUError(f"MinerU parsing failed: {task.get('err_msg') or 'unknown error'}")
        time.sleep(poll_interval_seconds)
    raise MinerUError(f"MinerU did not finish within {timeout_seconds:.0f} seconds.")


def convert_image_to_markdown(
    image_bytes: bytes,
    *,
    source_id: str,
    status: Callable[[str], None] | None = None,
    poll_interval_seconds: float = 2.0,
    timeout_seconds: float = 300.0,
) -> str:
    """Convert one image after normalising it to a stable PNG upload."""
    return _convert_bytes_to_markdown(
        _normalise_png(image_bytes),
        source_id=source_id,
        suffix=".png",
        source_label="lossless table image",
        status=status,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )


def convert_document_to_markdown(
    path: Path,
    *,
    source_id: str,
    status: Callable[[str], None] | None = None,
    poll_interval_seconds: float = 2.0,
    timeout_seconds: float = 600.0,
) -> str:
    """Convert an original PDF (or other documented MinerU file type) once."""
    suffix = path.suffix.lower()
    if suffix not in {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}:
        raise MinerUError(f"Unsupported original-document extension: {suffix or '[none]'}")
    if not path.is_file():
        raise MinerUError(f"Original document does not exist: {path}")
    return _convert_bytes_to_markdown(
        path.read_bytes(),
        source_id=source_id,
        suffix=suffix,
        source_label=f"original document ({path.name})",
        status=status,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )


# Kept as an explicit alias for callers that operate on tables only.
transcribe_table_image = convert_image_to_markdown
