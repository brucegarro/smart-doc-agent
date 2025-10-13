"""CLI interface for smart-doc-agent."""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from agent.ingestion.processor import processor

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
    )
):
    """
    Ingest PDF documents into the system.
    
    Extracts text, metadata, tables, and stores in database with vector embeddings.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print("\n[bold blue]📄 PDF Ingestion Pipeline[/bold blue]\n")
    
    # Collect PDF files
    pdf_files = []
    
    if path.is_file():
        if path.suffix.lower() == ".pdf":
            pdf_files.append(path)
        else:
            console.print(f"[red]Error: {path} is not a PDF file[/red]")
            raise typer.Exit(1)
    
    elif path.is_dir():
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(path.glob(pattern))
        
        if not pdf_files:
            console.print(f"[yellow]No PDF files found in {path}[/yellow]")
            raise typer.Exit(0)
    
    console.print(f"Found [cyan]{len(pdf_files)}[/cyan] PDF file(s)\n")
    
    # Process each PDF
    results = {
        "success": [],
        "failed": [],
        "skipped": []
    }
    
    for idx, pdf_file in enumerate(pdf_files, start=1):
        console.print(f"[{idx}/{len(pdf_files)}] Processing: [cyan]{pdf_file.name}[/cyan]")
        
        try:
            doc_id = processor.process_pdf(pdf_file, source_name=source)
            results["success"].append((pdf_file.name, doc_id))
            console.print(f"  ✓ Success → [green]{doc_id}[/green]\n")
        
        except ValueError as e:
            # Document already exists
            results["skipped"].append((pdf_file.name, str(e)))
            console.print(f"  ⊘ Skipped: [yellow]{e}[/yellow]\n")
        
        except Exception as e:
            results["failed"].append((pdf_file.name, str(e)))
            console.print(f"  ✗ Failed: [red]{e}[/red]\n")
            if verbose:
                logger.exception("Processing failed")
    
    # Summary
    console.print("\n[bold]Summary[/bold]")
    console.print(f"  ✓ Success: [green]{len(results['success'])}[/green]")
    console.print(f"  ⊘ Skipped: [yellow]{len(results['skipped'])}[/yellow]")
    console.print(f"  ✗ Failed:  [red]{len(results['failed'])}[/red]")
    
    if results["failed"]:
        console.print("\n[bold red]Failed documents:[/bold red]")
        for filename, error in results["failed"]:
            console.print(f"  • {filename}: {error}")


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
