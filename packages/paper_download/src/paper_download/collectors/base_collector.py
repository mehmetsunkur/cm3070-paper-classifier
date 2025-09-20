"""Base collector class for all paper download sources"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Metadata for a downloaded paper"""

    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    source: str  # API source (arxiv, semantic_scholar, etc.)
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    categories: list[str] = field(default_factory=list)
    discipline: Optional[str] = None  # CS, IS, IT (kept for backward compatibility)
    research_field: Optional[str] = None  # ML, SE, HCI, etc.
    extra_data: dict[str, Any] = field(default_factory=dict)
    
    # CC2020 fields
    cc2020_discipline: Optional[str] = None  # CS, SE, CE, IS, IT, CSEC, DS
    discipline_confidence: float = 0.0  # Confidence score 0.0-1.0
    secondary_disciplines: list[str] = field(default_factory=list)  # Secondary CC2020 disciplines
    classification_method: str = ""  # "category", "keyword", "hybrid"


@dataclass
class DownloadResult:
    """Result of a paper download attempt"""

    success: bool
    paper_id: str
    source: str
    file_path: Optional[Path] = None
    metadata: Optional[PaperMetadata] = None
    error_message: Optional[str] = None
    download_time: Optional[float] = None  # seconds
    file_size: Optional[int] = None  # bytes
    timestamp: datetime = field(default_factory=datetime.now)


class BaseCollector(ABC):
    """Abstract base class for paper collectors"""

    def __init__(self, output_dir: Path, config: dict[str, Any]):
        """
        Initialize collector

        Args:
            output_dir: Directory to save downloaded papers
            config: Configuration dictionary
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.results: list[DownloadResult] = []
        self.rate_limit_delay = config.get("rate_limit_delay", 0.5)

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> list[PaperMetadata]:
        """
        Search for papers based on query

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of paper metadata
        """
        pass

    @abstractmethod
    def download_paper(self, metadata: PaperMetadata) -> DownloadResult:
        """
        Download a single paper

        Args:
            metadata: Paper metadata

        Returns:
            Download result
        """
        pass

    def collect_papers(self, query: str, max_papers: int = 10) -> list[DownloadResult]:
        """
        Search and download papers

        Args:
            query: Search query
            max_papers: Maximum number of papers to download

        Returns:
            List of download results
        """
        logger.info(f"Collecting papers for query: {query}")

        # Search for papers
        papers = self.search(query, max_papers)
        logger.info(f"Found {len(papers)} papers")

        # Download each paper
        results = []
        for i, paper in enumerate(papers):
            logger.info(f"Downloading paper {i+1}/{len(papers)}: {paper.title[:50]}...")

            # Download with timing
            start_time = time.time()
            result = self.download_paper(paper)
            result.download_time = time.time() - start_time

            results.append(result)
            self.results.append(result)

            if result.success:
                logger.info(f"✓ Downloaded successfully in {result.download_time:.2f}s")
            else:
                logger.warning(f"✗ Failed: {result.error_message}")

            # Rate limiting
            if i < len(papers) - 1:
                time.sleep(self.rate_limit_delay)

        return results

    def get_statistics(self) -> dict[str, Any]:
        """Get download statistics"""
        total = len(self.results)
        if total == 0:
            return {"total": 0, "success_rate": 0}

        successful = sum(1 for r in self.results if r.success)
        avg_time = sum(r.download_time or 0 for r in self.results if r.success) / max(
            successful, 1
        )
        total_size = sum(r.file_size or 0 for r in self.results if r.success)

        return {
            "source": self.__class__.__name__,
            "total_attempts": total,
            "successful_downloads": successful,
            "success_rate": successful / total * 100,
            "average_download_time": avg_time,
            "total_size_mb": total_size / (1024 * 1024),
            "errors": [r.error_message for r in self.results if not r.success],
        }

    def save_results(self, output_file: Path):
        """Save results to file"""
        import json

        import pandas as pd

        stats = self.get_statistics()

        # Save as JSON
        json_file = output_file.with_suffix(".json")
        with open(json_file, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        # Save as CSV
        if self.results:
            df_data = []
            for r in self.results:
                df_data.append(
                    {
                        "paper_id": r.paper_id,
                        "source": r.source,
                        "success": r.success,
                        "download_time": r.download_time,
                        "file_size": r.file_size,
                        "error": r.error_message,
                        "timestamp": r.timestamp,
                    }
                )
            df = pd.DataFrame(df_data)
            csv_file = output_file.with_suffix(".csv")
            df.to_csv(csv_file, index=False)

        logger.info(f"Results saved to {output_file}")
