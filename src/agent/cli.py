"""CLI interface for smart-doc-agent."""

import cProfile
import io
import logging
import os
import pstats
import sys
from contextlib import suppress
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from agent.config import settings
from agent.ingestion.processor import ParallelIngestSummary, processor
from agent.retrieval.search import search_chunks

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer(
    name="smart-doc-agent",
    help="AI-powered research paper processing and analysis agent"
)

console = Console()

PROFILE_DEFAULT_PATH = Path("/tmp/ingest.prof")


def _emit_profile_summary(profiler: Optional[cProfile.Profile]) -> None:
    """Print a concise cProfile summary if profiling data exists."""
    stats_path = Path(os.environ.get("SMART_DOC_AGENT_PROFILE_PATH", str(PROFILE_DEFAULT_PATH)))

    stats_obj: Optional[pstats.Stats] = None

    if profiler is not None:
        try:
            profiler.dump_stats(str(stats_path))
        except Exception as exc:  # pragma: no cover - best effort persistence
            logger.debug("Unable to write profile stats: %s", exc)
        stats_obj = pstats.Stats(profiler)
    elif stats_path.exists():
        try:
            stats_obj = pstats.Stats(str(stats_path))
        except Exception as exc:  # pragma: no cover - best effort parsing
            logger.debug("Skipping profile summary: %s", exc)
            return
    else:
        return

    if stats_obj is None:
        return

    try:
        top_n = int(os.environ.get("SMART_DOC_AGENT_PROFILE_TOP", "25"))
    except ValueError:
        top_n = 25

    sort_key = os.environ.get("SMART_DOC_AGENT_PROFILE_SORT", "cumulative")

    stats_obj.strip_dirs().sort_stats(sort_key)
    buffer = io.StringIO()
    stats_obj.stream = buffer
    stats_obj.print_stats(top_n)

    logger.debug(
        "Profiler summary ready (source: %s)",
        "in-memory" if profiler is not None else str(stats_path),
    )

    console.print("\nProfiler Summary\n", style="bold magenta")
    console.print(buffer.getvalue(), markup=False, highlight=False)

    if os.environ.get("SMART_DOC_AGENT_PROFILE_DELETE", "0") == "1":
        with suppress(OSError):
            stats_path.unlink()


def _start_profiler_if_enabled() -> Optional[cProfile.Profile]:
    if os.environ.get("SMART_DOC_AGENT_PROFILE", "0") != "1":
        return None

    profiler = cProfile.Profile()
    profiler.enable()
    logger.debug(
        "Profiling enabled (output path: %s)",
        os.environ.get("SMART_DOC_AGENT_PROFILE_PATH", str(PROFILE_DEFAULT_PATH)),
    )
    return profiler


def _set_verbose_logging(verbose: bool) -> Optional[int]:
    if not verbose:
        return None
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)
    return previous_level


def _collect_pdf_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            console.print(f"[red]Error: {path} is not a PDF file[/red]")
            raise typer.Exit(1)
        return [path]

    if path.is_dir():
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = sorted(path.glob(pattern))
        if not pdf_files:
            console.print(f"[yellow]No PDF files found in {path}[/yellow]")
            raise typer.Exit(0)
        return pdf_files

    console.print(f"[red]Unsupported path type: {path}[/red]")
    raise typer.Exit(1)


def _process_pdfs(
    pdf_files: list[Path],
    source: Optional[str],
    verbose: bool,
    workers: int,
) -> dict[str, list[tuple[str, str]]]:
    results = {"success": [], "failed": [], "skipped": [], "queued": []}

    if len(pdf_files) > 1 or workers > 1:
        console.print(
            "Enqueuing PDFs for background ingestion via Redis queue\n"
        )
        summary = processor.process_pdfs_parallel(
            pdf_files,
            source_name=source,
            max_workers=workers,
        )
        _log_parallel_results(pdf_files, summary, results, verbose)
        return results

    for idx, pdf_file in enumerate(pdf_files, start=1):
        console.print(f"[{idx}/{len(pdf_files)}] Processing: [cyan]{pdf_file.name}[/cyan]")
        bucket, payload = _handle_pdf_ingest(pdf_file, source, verbose)
        results[bucket].append(payload)

    return results


def _log_parallel_results(
    pdf_files: list[Path],
    summary: ParallelIngestSummary,
    results: dict[str, list[tuple[str, str]]],
    verbose: bool,
) -> None:
    total = len(pdf_files)
    queue_name = summary.queue_name or settings.redis_queue_ingest
    console.print(f"Queue name: [cyan]{queue_name}[/cyan]")

    for idx, outcome in enumerate(summary.items, start=1):
        pdf_name = pdf_files[idx - 1].name
        console.print(f"[{idx}/{total}] Processing: [cyan]{pdf_name}[/cyan]")

        if outcome.status == "queued" and outcome.job_id:
            console.print(f"  ↻ Enqueued job → [cyan]{outcome.job_id}[/cyan]\n")
            results["queued"].append((pdf_name, outcome.job_id))
        elif outcome.status == "skipped":
            reason = outcome.message or "Skipped"
            console.print(f"  ⊘ Skipped: [yellow]{reason}[/yellow]\n")
            results["skipped"].append((pdf_name, reason))
        else:
            reason = outcome.message or "Unknown error"
            console.print(f"  ✗ Failed: [red]{reason}[/red]\n")
            if verbose:
                logger.error("Processing failed for %s: %s", pdf_name, reason)
            results["failed"].append((pdf_name, reason))


def _handle_pdf_ingest(pdf_file: Path, source: Optional[str], verbose: bool) -> tuple[str, tuple[str, str]]:
    try:
        doc_id = processor.process_pdf(pdf_file, source_name=source)
        console.print(f"  ✓ Success → [green]{doc_id}[/green]\n")
        return "success", (pdf_file.name, doc_id)
    except ValueError as exc:
        console.print(f"  ⊘ Skipped: [yellow]{exc}[/yellow]\n")
        return "skipped", (pdf_file.name, str(exc))
    except Exception as exc:  # pragma: no cover - surfacing ingest errors
        console.print(f"  ✗ Failed: [red]{exc}[/red]\n")
        if verbose:
            logger.exception("Processing failed")
        else:
            logger.error("Processing failed for %s", pdf_file.name)
        return "failed", (pdf_file.name, str(exc))


def _print_ingest_summary(results: dict[str, list[tuple[str, str]]]) -> None:
    console.print("\n[bold]Summary[/bold]")
    summary_rows = [
        ("queued", "↻ Queued", "cyan"),
        ("success", "✓ Success", "green"),
        ("skipped", "⊘ Skipped", "yellow"),
        ("failed", "✗ Failed", "red"),
    ]

    for key, label, color in summary_rows:
        if key in results:
            console.print(f"  {label}: [{color}]{len(results[key])}[/{color}]")

    if results["failed"]:
        console.print("\n[bold red]Failed documents:[/bold red]")
        for filename, error in results["failed"]:
            console.print(f"  • {filename}: {error}")


@app.command()
def ingest(
    path: Path = typer.Argument(
        ...,
        help="Path to PDF file or directory containing PDFs",
        exists=True
    ),
    source: Optional[str] = typer.Option(
        None,
        "--source",
        "-s",
        help="Source identifier (e.g., 'arxiv', 'pubmed')"
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Recursively scan directory for PDFs"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        min=1,
        help="Number of parallel workers to use"
    )
):
    """
    Ingest PDF documents into the system.
    
    Extracts text, metadata, tables, and stores in database with vector embeddings.
    """
    profiler = _start_profiler_if_enabled()
    previous_level = _set_verbose_logging(verbose)
    try:
        _run_ingest_command(path, recursive, source, verbose, workers)
    finally:
        if profiler is not None:
            profiler.disable()
        if previous_level is not None:
            logging.getLogger().setLevel(previous_level)
        _emit_profile_summary(profiler)


def _run_ingest_command(
    path: Path,
    recursive: bool,
    source: Optional[str],
    verbose: bool,
    workers: int,
) -> None:
    console.print("\n[bold blue]📄 PDF Ingestion Pipeline[/bold blue]\n")
    pdf_files = _collect_pdf_files(path, recursive)
    console.print(f"Found [cyan]{len(pdf_files)}[/cyan] PDF file(s)\n")

    results = _process_pdfs(pdf_files, source, verbose, workers)
    _print_ingest_summary(results)


@app.command()
def list(
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Number of documents to show"
    ),
    offset: int = typer.Option(
        0,
        "--offset",
        help="Pagination offset"
    )
):
    """
    List ingested documents.
    """
    console.print("\n[bold blue]📚 Ingested Documents[/bold blue]\n")
    
    documents = processor.list_documents(limit=limit, offset=offset)
    
    if not documents:
        console.print("[yellow]No documents found[/yellow]")
        return
    
    # Create table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("ID", style="dim", width=36)
    table.add_column("Title", style="cyan", no_wrap=False)
    table.add_column("Authors", style="dim", no_wrap=False)
    table.add_column("Year", justify="center", width=6)
    table.add_column("Pages", justify="center", width=6)
    table.add_column("Status", justify="center", width=10)
    
    for doc in documents:
        authors_str = ", ".join(doc["authors"][:2]) if doc["authors"] else "-"
        if doc["authors"] and len(doc["authors"]) > 2:
            authors_str += f" +{len(doc['authors']) - 2}"
        
        year_str = str(doc["publication_year"]) if doc["publication_year"] else "-"
        
        status_style = {
            "ingested": "yellow",
            "indexed": "green",
            "failed": "red"
        }.get(doc["status"], "white")
        
        table.add_row(
            doc["id"][:8] + "...",
            doc["title"] or doc["filename"],
            authors_str,
            year_str,
            str(doc["num_pages"]),
            f"[{status_style}]{doc['status']}[/{status_style}]"
        )
    
    console.print(table)
    console.print(f"\n[dim]Showing {len(documents)} document(s) (offset: {offset})[/dim]\n")


@app.command()
def status(
    doc_id: str = typer.Argument(
        ...,
        help="Document ID (UUID)"
    )
):
    """
    Show status of a specific document.
    """
    doc_info = processor.get_document_status(doc_id)
    
    if not doc_info:
        console.print(f"[red]Document not found: {doc_id}[/red]")
        raise typer.Exit(1)
    
    console.print("\n[bold blue]📄 Document Status[/bold blue]\n")
    
    for key, value in doc_info.items():
        console.print(f"  [cyan]{key:15}[/cyan]: {value}")
    
    console.print()


@app.command()
def version():
    """Show version information."""
    console.print("[cyan]smart-doc-agent[/cyan] v0.1.0")


@app.command()
def search(
    query: str = typer.Argument(
        ...,
        help="Text to search for"
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-n",
        help="Number of chunks to return"
    ),
    include_tables: bool = typer.Option(
        False,
        "--include-tables",
        help="Include table and figure chunks"
    )
):
    """Search indexed chunks and show the closest matches."""

    console.print("\n[bold blue]Similarity Search[/bold blue]\n")

    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("agent.embedding.embedder").setLevel(logging.WARNING)

    try:
        content_types = None if include_tables else ("text",)
        results = search_chunks(
            query,
            limit=limit,
            content_types=content_types,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if not results:
        console.print("[yellow]No matching chunks found[/yellow]")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", justify="right", width=3)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Page", justify="right", width=6)
    table.add_column("Type", width=8)
    table.add_column("Doc", width=10)
    table.add_column("Chunk", justify="right", width=6)
    table.add_column("Snippet", overflow="fold")

    for rank, item in enumerate(results, start=1):
        table.add_row(
            str(rank),
            f"{item.cosine_similarity:.3f}",
            str(item.page_number),
            item.content_type,
            item.document_id[:8] + "...",
            str(item.chunk_index),
            item.snippet or "-",
        )

    console.print(table)


def main():
    """Main CLI entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("Unhandled exception")
        sys.exit(1)


if __name__ == "__main__":
    main()
