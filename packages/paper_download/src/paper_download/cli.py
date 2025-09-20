#!/usr/bin/env python3
"""Command-line interface for paper-download tool - Computing papers downloader"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.progress import track
from rich.table import Table

from paper_download.collectors.arxiv_collector import ArxivCollector
from paper_download.mappers import MapperRegistry

console = Console()
logger = logging.getLogger(__name__)

# Default computing categories for random selection
DEFAULT_CATEGORIES = [
    "cs.LG",  # Machine Learning
    "cs.AI",  # Artificial Intelligence
    "cs.SE",  # Software Engineering
    "cs.CV",  # Computer Vision
    "cs.CL",  # Computation and Language
    "cs.HC",  # Human-Computer Interaction
    "cs.DB",  # Databases
    "cs.NI",  # Networking
    "cs.CR",  # Cryptography and Security
    "cs.PL",  # Programming Languages
    "cs.DS",  # Data Structures and Algorithms
    "cs.IR",  # Information Retrieval
]


@click.group()
@click.version_option(version="1.1.0", prog_name="paper-download")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def cli(verbose):
    """Paper Download - Download computing research papers with hierarchical organization"""
    if verbose:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
    else:
        logging.basicConfig(level=logging.WARNING)


@cli.command()
@click.option(
    "--mapper", 
    type=click.Choice(MapperRegistry.list_mappers()), 
    default="cc2020",
    help="Organization scheme to use (default: cc2020)"
)
@click.option(
    "--filter", "-f", 
    help="Smart filter: CC2020 discipline (CS/SE/DS) or ArXiv category (cs.LG)"
)
@click.option("--max", "-m", default=10, help="Maximum papers to download")
@click.option(
    "--output-dir", "-o", 
    type=click.Path(), 
    default="./papers", 
    help="Output directory"
)
@click.option("--show-stats", is_flag=True, help="Show classification statistics")
def download(mapper, filter, max, output_dir, show_stats):
    """Download papers with intelligent filtering and organization
    
    Examples:
        # Download Data Science papers (CC2020)
        paper-download download --filter DS --max 20
        
        # Download specific ArXiv category with CC2020 organization
        paper-download download --filter cs.LG --max 15
        
        # Download with ArXiv organization
        paper-download download --mapper arxiv --filter cs.LG --max 10
        
        # Random download with CC2020 organization
        paper-download download --max 10
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Display mapper information
    console.print(f"[bold blue]Using mapper: {mapper.upper()}[/bold blue]")
    console.print(f"[dim]Output directory: {output_path / mapper.upper()}[/dim]")
    
    # Configure collector
    config = {
        "mapper": mapper,
        "rate_limit_delay": 0.5
    }
    
    collector = ArxivCollector(output_path, config)
    
    # Build query based on filter
    query = _build_query(mapper, filter)
    
    if query:
        console.print(f"[cyan]Filter: {filter}[/cyan]")
        if query.startswith("cc2020:"):
            console.print(f"[dim]Searching multiple categories for {filter} discipline[/dim]")
    else:
        console.print("[yellow]No filter specified - selecting random papers[/yellow]")
        query = _get_random_query()
    
    # Download papers with progress
    with console.status(f"[bold green]Downloading papers..."):
        results = collector.collect_papers(query, max)
    
    # Display results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    console.print(f"\n[bold green]✓ Downloaded {len(successful)}/{max} papers[/bold green]")
    
    if failed:
        console.print(f"[bold red]✗ Failed: {len(failed)} papers[/bold red]")
        for r in failed[:3]:  # Show first 3 errors
            console.print(f"  - {r.paper_id}: {r.error_message}")
    
    # Show organization statistics
    if successful and show_stats:
        _show_organization_stats(successful, mapper)
    
    # Show mapper statistics if available
    if hasattr(collector.mapper, 'get_stats'):
        stats = collector.mapper.get_stats()
        if stats and show_stats:
            console.print("\n[bold]Classification Statistics:[/bold]")
            for key, value in stats.items():
                if key != "categories_seen":  # Skip long lists
                    console.print(f"  {key}: {value}")


def _build_query(mapper: str, filter: Optional[str]) -> Optional[str]:
    """
    Build search query based on mapper type and filter value.
    
    For CC2020 mapper:
    - If filter is CC2020 discipline (CS, SE, DS, etc.) → search its mapped categories
    - If filter is ArXiv category (cs.LG, etc.) → search that category, organize by CC2020
    
    For ArXiv mapper:
    - Filter is always interpreted as ArXiv category
    """
    if not filter:
        return None
    
    if mapper == "cc2020":
        # CC2020 discipline codes
        CC2020_DISCIPLINES = ["CS", "SE", "CE", "IS", "IT", "CSEC", "DS"]
        
        if filter.upper() in CC2020_DISCIPLINES:
            # It's a CC2020 discipline - use special query format
            return f"cc2020:{filter.upper()}"
        elif "." in filter:  # Likely an ArXiv category
            # Search specific ArXiv category, will be organized by CC2020
            return f"cat:{filter}"
        else:
            console.print(f"[red]Invalid filter for CC2020: {filter}[/red]")
            console.print(f"[dim]Use discipline code {CC2020_DISCIPLINES} or ArXiv category (e.g., cs.LG)[/dim]")
            sys.exit(1)
    
    elif mapper == "arxiv":
        # For ArXiv mapper, interpret as category
        if "." in filter:
            return f"cat:{filter}"
        else:
            # Try to be helpful
            console.print(f"[yellow]Warning: '{filter}' doesn't look like an ArXiv category.[/yellow]")
            console.print("[dim]ArXiv categories have format like: cs.LG, cs.AI, stat.ML[/dim]")
            return f"cat:{filter}"  # Try anyway
    
    return None


def _get_random_query() -> str:
    """Get a random category for download"""
    category = random.choice(DEFAULT_CATEGORIES)
    console.print(f"[dim]Selected random category: {category}[/dim]")
    return f"cat:{category}"


def _show_organization_stats(results, mapper_name):
    """Show how papers were organized"""
    from collections import Counter
    
    # This would need access to the actual paths, simplified for now
    console.print("\n[bold]Organization Summary:[/bold]")
    console.print(f"  Papers organized using {mapper_name.upper()} structure")
    console.print(f"  Check {Path('./papers') / mapper_name.upper()} for downloaded papers")


@cli.command(name="download-ids")
@click.argument("paper_ids", nargs=-1, required=True)
@click.option(
    "--mapper", 
    type=click.Choice(MapperRegistry.list_mappers()), 
    default="cc2020",
    help="Organization scheme to use"
)
@click.option(
    "--output-dir", "-o", 
    type=click.Path(), 
    default="./papers", 
    help="Output directory"
)
def download_ids(paper_ids, mapper, output_dir):
    """Download specific papers by their ArXiv IDs
    
    Examples:
        paper-download download-ids 2301.12345 2302.54321
        paper-download download-ids 2301.12345 --mapper arxiv
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold blue]Downloading {len(paper_ids)} specific papers[/bold blue]")
    console.print(f"[dim]Using {mapper} organization[/dim]")
    
    config = {
        "mapper": mapper,
        "rate_limit_delay": 0.5
    }
    
    collector = ArxivCollector(output_path, config)
    
    results = []
    for paper_id in track(paper_ids, description="Downloading papers"):
        # Search for specific paper
        papers = collector.search(paper_id, max_results=1)
        
        if papers:
            result = collector.download_paper(papers[0])
            results.append(result)
            
            if result.success:
                console.print(f"[green]✓[/green] {paper_id}")
            else:
                console.print(f"[red]✗[/red] {paper_id}: {result.error_message}")
        else:
            console.print(f"[yellow]![/yellow] {paper_id}: Not found")
    
    # Summary
    successful = sum(1 for r in results if r.success)
    console.print(f"\n[bold]Downloaded {successful}/{len(paper_ids)} papers[/bold]")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--mapper", 
    type=click.Choice(MapperRegistry.list_mappers()), 
    default="cc2020",
    help="Organization scheme to use"
)
@click.option(
    "--output-dir", "-o", 
    type=click.Path(), 
    default="./papers", 
    help="Output directory"
)
@click.option("--report", "-r", is_flag=True, help="Generate download report")
def batch(input_file, mapper, output_dir, report):
    """Batch download papers from a file
    
    File format: One ArXiv ID or category per line
    Lines starting with # are ignored
    
    Example file:
        # Machine Learning papers
        2301.12345
        2302.54321
        cat:cs.LG  # Download 10 from category
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read input file
    with open(input_file, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    if not lines:
        console.print("[red]No valid entries found in input file[/red]")
        return
    
    console.print(f"[bold blue]Batch downloading from {input_file}[/bold blue]")
    console.print(f"[dim]Found {len(lines)} entries[/dim]")
    
    config = {
        "mapper": mapper,
        "rate_limit_delay": 0.5
    }
    
    collector = ArxivCollector(output_path, config)
    
    all_results = []
    for line in track(lines, description="Processing entries"):
        if line.startswith("cat:"):
            # Category download
            results = collector.collect_papers(line, max_papers=10)
            all_results.extend(results)
        else:
            # Assume it's a paper ID
            papers = collector.search(line, max_results=1)
            if papers:
                result = collector.download_paper(papers[0])
                all_results.append(result)
    
    # Summary
    successful = [r for r in all_results if r.success]
    failed = [r for r in all_results if not r.success]
    
    console.print(f"\n[bold green]✓ Downloaded {len(successful)} papers[/bold green]")
    if failed:
        console.print(f"[bold red]✗ Failed: {len(failed)} papers[/bold red]")
    
    # Generate report if requested
    if report:
        report_file = Path(input_file).stem + "_report.json"
        report_data = {
            "input_file": str(input_file),
            "mapper": mapper,
            "total_entries": len(lines),
            "successful_downloads": len(successful),
            "failed_downloads": len(failed),
            "results": [
                {
                    "paper_id": r.paper_id,
                    "success": r.success,
                    "error": r.error_message if not r.success else None,
                    "file_path": str(r.file_path) if r.file_path else None,
                }
                for r in all_results
            ],
        }
        
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)
        
        console.print(f"[green]Report saved to {report_file}[/green]")


@cli.command()
@click.option(
    "--mapper", 
    type=click.Choice(MapperRegistry.list_mappers()),
    help="Show categories for specific mapper"
)
def categories(mapper):
    """List available categories and organization schemes"""
    
    if mapper:
        # Show categories for specific mapper
        console.print(f"[bold blue]{mapper.upper()} Categories[/bold blue]\n")
        
        try:
            mapper_info = MapperRegistry.get_mapper_info(mapper)
            categories = mapper_info.get("categories", {})
            
            if mapper == "cc2020":
                # Show CC2020 disciplines and their ArXiv categories
                if "discipline_mapping" in categories:
                    table = Table(title=f"CC2020 Disciplines → ArXiv Categories")
                    table.add_column("Discipline", style="cyan", no_wrap=True)
                    table.add_column("Name", style="green")
                    table.add_column("ArXiv Categories", style="white")
                    
                    discipline_names = {
                        "CS": "Computer Science",
                        "SE": "Software Engineering",
                        "CE": "Computer Engineering",
                        "IS": "Information Systems",
                        "IT": "Information Technology",
                        "CSEC": "Cybersecurity",
                        "DS": "Data Science"
                    }
                    
                    for disc, cats in categories["discipline_mapping"].items():
                        name = discipline_names.get(disc, disc)
                        cats_str = ", ".join(cats[:5])
                        if len(cats) > 5:
                            cats_str += f" ... (+{len(cats)-5} more)"
                        table.add_row(disc, name, cats_str)
                    
                    console.print(table)
                    console.print("\n[dim]Use: paper-download download --filter [DISCIPLINE][/dim]")
            
            elif mapper == "arxiv":
                # Show ArXiv categories by domain
                for domain, cats in categories.items():
                    if cats:
                        console.print(f"[cyan]{domain.replace('_', ' ').title()}:[/cyan]")
                        for cat in cats[:10]:  # Show first 10
                            console.print(f"  • {cat}")
                        if len(cats) > 10:
                            console.print(f"  ... and {len(cats)-10} more")
                        console.print()
                
                console.print("[dim]Use: paper-download download --mapper arxiv --filter [CATEGORY][/dim]")
            
        except Exception as e:
            console.print(f"[red]Error getting categories: {e}[/red]")
    
    else:
        # Show all available mappers
        console.print("[bold blue]Available Organization Schemes[/bold blue]\n")
        
        table = Table(title="Mappers")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Description", style="green")
        table.add_column("Usage", style="white")
        
        mapper_descriptions = {
            "cc2020": ("CC2020 Computing Curriculum", 
                      "Organizes by academic disciplines", 
                      "--mapper cc2020 --filter DS"),
            "arxiv": ("ArXiv Categories", 
                     "Organizes by research areas", 
                     "--mapper arxiv --filter cs.LG")
        }
        
        for name in MapperRegistry.list_mappers():
            if name in mapper_descriptions:
                desc, org, usage = mapper_descriptions[name]
                table.add_row(name, f"{desc}\n{org}", f"paper-download download\n{usage}")
        
        console.print(table)
        console.print("\n[dim]Use --mapper [name] to see specific categories[/dim]")


@cli.command()
def stats():
    """Show download statistics from existing papers directory"""
    papers_dir = Path("./papers")
    
    if not papers_dir.exists():
        console.print("[red]No papers directory found[/red]")
        return
    
    console.print("[bold blue]Download Statistics[/bold blue]\n")
    
    # Count papers by mapper
    for mapper_dir in papers_dir.iterdir():
        if mapper_dir.is_dir():
            pdf_count = len(list(mapper_dir.rglob("*.pdf")))
            json_count = len(list(mapper_dir.rglob("*metadata.json")))
            
            console.print(f"[cyan]{mapper_dir.name}:[/cyan]")
            console.print(f"  Papers: {pdf_count}")
            console.print(f"  Metadata files: {json_count}")
            
            # Count by subdirectory
            subdirs = [d for d in mapper_dir.iterdir() if d.is_dir()]
            if subdirs:
                console.print("  Distribution:")
                for subdir in sorted(subdirs)[:10]:  # Show top 10
                    count = len(list(subdir.rglob("*.pdf")))
                    if count > 0:
                        console.print(f"    {subdir.name}: {count}")


def main():
    """Main entry point"""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()