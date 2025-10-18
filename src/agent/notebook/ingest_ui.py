"""Notebook helpers for uploading PDFs and invoking ingestion via Docker CLI."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Sequence
from uuid import uuid4

import ipywidgets as widgets
from IPython.display import display

from .context import NotebookContext, init_notebook_environment

LOGGER = logging.getLogger(__name__)

DEFAULT_CONTAINER = os.getenv("SDA_INGEST_CONTAINER", "doc_app")
DEFAULT_DOCKER_BINARY = os.getenv("SDA_DOCKER_BIN", "docker")
DEFAULT_REMOTE_PARENT = os.getenv("SDA_REMOTE_PARENT", "/tmp/notebook_ingest")


class CommandError(RuntimeError):
    """Raised when a shell command returns a non-zero exit code."""

    def __init__(self, command: Sequence[str], result: subprocess.CompletedProcess[str]):
        self.command = list(command)
        self.result = result
        message = (
            f"Command failed with exit code {result.returncode}: {' '.join(command)}"
        )
        super().__init__(message)


def display_ingest_ui(
    *,
    context: Optional[NotebookContext] = None,
    source: Optional[str] = None,
    workers: int = 4,
    docker_binary: Optional[str] = None,
    container_name: Optional[str] = None,
    remote_parent: Optional[str] = None,
) -> None:
    """
    Render a drag-and-drop UI that stages PDFs locally and invokes the ingest CLI inside Docker.
    """

    ctx = context or init_notebook_environment()
    staging_dir = ctx.staging_dir
    docker_cmd = docker_binary or DEFAULT_DOCKER_BINARY
    container = container_name or DEFAULT_CONTAINER
    remote_dir_parent = remote_parent or DEFAULT_REMOTE_PARENT

    uploader = widgets.FileUpload(
        description="Drag PDFs",
        multiple=True,
        accept=".pdf",
        layout=widgets.Layout(width="auto"),
    )
    queue_button = widgets.Button(
        description="Queue PDFs",
        button_style="primary",
        icon="cloud-upload",
        disabled=False,
    )
    source_text = widgets.Text(
        value=source or "",
        placeholder="Optional source name",
        description="Source",
        layout=widgets.Layout(width="400px"),
    )
    worker_spinner = widgets.BoundedIntText(
        value=max(1, workers),
        min=1,
        max=16,
        step=1,
        description="Workers",
        layout=widgets.Layout(width="200px"),
    )
    status_output = widgets.Output(layout=widgets.Layout(width="100%"))

    def _reset_uploader() -> None:
        uploader.value.clear()
        uploader._counter = 0  # type: ignore[attr-defined]  # ipywidgets internals

    def _on_queue_clicked(_: widgets.Button) -> None:
        with status_output:
            status_output.clear_output()
            if not uploader.value:
                print("⚠️  Please upload at least one PDF before queuing.")
                return

            try:
                files = _persist_uploads(uploader.value.values(), staging_dir)
            except Exception as exc:  # pragma: no cover - filesystem errors surface directly
                LOGGER.exception("Failed to persist uploads")
                print(f"❌ Upload failed: {exc}")
                return

            try:
                result = _enqueue_with_cli(
                    files,
                    source_text.value or None,
                    worker_spinner.value,
                    docker_cmd=docker_cmd,
                    container=container,
                    remote_parent_dir=remote_dir_parent,
                )
            except FileNotFoundError as exc:
                LOGGER.exception("Docker executable not found")
                print(f"❌ Docker command not found: {exc}")
                return
            except CommandError as exc:
                LOGGER.exception("Failed to enqueue PDFs via Docker CLI")
                print(f"❌ Ingestion failed (exit {exc.result.returncode}).")
                if exc.result.stdout:
                    print("--- stdout ---")
                    print(exc.result.stdout.rstrip())
                if exc.result.stderr:
                    print("--- stderr ---")
                    print(exc.result.stderr.rstrip())
                return
            except Exception as exc:  # pragma: no cover - unexpected errors get surfaced
                LOGGER.exception("Unexpected error during ingestion")
                print(f"❌ Unexpected error: {exc}")
                return
            finally:
                for path in files:
                    try:
                        path.unlink(missing_ok=True)
                    except Exception as cleanup_error:  # pragma: no cover - best effort cleanup
                        LOGGER.warning("Failed to remove temp upload %s: %s", path, cleanup_error)

            print(result.stdout.rstrip() if result.stdout else "✅ Ingestion command completed.")
            if result.stderr:
                print("--- stderr ---")
                print(result.stderr.rstrip())
            _reset_uploader()

    queue_button.on_click(_on_queue_clicked)

    header = widgets.HTML(
        """
        <h3 style='margin-bottom:4px;'>PDF Ingestion</h3>
        <p style='margin-top:0;'>Upload PDF files and queue them for background processing.</p>
        """
    )

    controls = widgets.HBox([source_text, worker_spinner, queue_button])
    layout = widgets.VBox([header, uploader, controls, status_output])

    display(layout)


def _persist_uploads(files: Iterable[dict], staging_dir: Path) -> list[Path]:
    staging_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for payload in files:
        filename: str = payload.get("metadata", {}).get("name") or payload.get("name")
        if not filename:
            filename = f"upload-{uuid4().hex}.pdf"
        content: bytes = payload["content"]
        safe_name = filename.replace("/", "_").replace("\\", "_")
        path = staging_dir / f"{uuid4().hex}_{safe_name}"
        path.write_bytes(content)
        saved_paths.append(path)
    return saved_paths


def _enqueue_with_cli(
    paths: Iterable[Path],
    source: Optional[str],
    workers: int,
    *,
    docker_cmd: str,
    container: str,
    remote_parent_dir: str,
) -> subprocess.CompletedProcess[str]:
    pdf_paths = list(paths)
    if not pdf_paths:
        raise ValueError("No PDF files to enqueue")

    session_dir = f"{remote_parent_dir.rstrip('/')}/{uuid4().hex}"
    _run_command([docker_cmd, "exec", container, "mkdir", "-p", session_dir])

    try:
        for path in pdf_paths:
            remote_path = f"{session_dir}/{path.name}"
            _run_command([docker_cmd, "cp", str(path), f"{container}:{remote_path}"])

        cli_cmd = [
            docker_cmd,
            "exec",
            container,
            "python",
            "-m",
            "agent.cli",
            "ingest",
            session_dir,
            "--workers",
            str(max(1, workers)),
        ]
        if source:
            cli_cmd.extend(["--source", source])

        result = _run_command(cli_cmd)
    finally:
        _run_command(
            [docker_cmd, "exec", container, "rm", "-rf", session_dir],
            check=False,
        )

    return result


def _run_command(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    LOGGER.debug("Executing command: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise CommandError(command, result)
    return result
