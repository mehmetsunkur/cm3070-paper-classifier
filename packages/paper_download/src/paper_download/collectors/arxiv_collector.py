"""ArXiv paper collector implementation"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import arxiv

from .base_collector import BaseCollector, DownloadResult, PaperMetadata
from ..mappers import MapperRegistry

logger = logging.getLogger(__name__)


class ArxivCollector(BaseCollector):
    """Collector for ArXiv papers"""

    def __init__(self, output_dir: Path, config: dict[str, Any]):
        super().__init__(output_dir, config)
        self.api_name = "arxiv"
        
        # Get mapper from config
        mapper_name = config.get("mapper", "cc2020")  # Default to CC2020
        mapper_config = config.get("mapper_config", {})
        self.mapper = MapperRegistry.get_mapper(mapper_name, mapper_config)
        
        logger.info(f"Using {mapper_name} mapper for paper organization")

    def search(self, query: str, max_results: int = 10) -> list[PaperMetadata]:
        """
        Search ArXiv for papers

        Args:
            query: Can be a category (e.g., "cs.LG") or search terms
            max_results: Maximum number of results

        Returns:
            List of paper metadata
        """
        papers = []

        try:
            # Determine if query is a category or search terms
            if query.startswith("cat:") or "." in query:
                # Category search
                if not query.startswith("cat:"):
                    query = f"cat:{query}"

            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )

            for result in search.results():
                metadata = PaperMetadata(
                    paper_id=result.get_short_id(),
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    source=self.api_name,
                    url=result.entry_id,
                    pdf_url=result.pdf_url,
                    year=result.published.year if result.published else None,
                    categories=result.categories,
                    extra_data={
                        "arxiv_id": result.get_short_id(),
                        "primary_category": result.primary_category,
                        "comment": result.comment,
                        "journal_ref": result.journal_ref,
                    },
                )
                
                papers.append(metadata)

        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")

        return papers

    def download_paper(self, metadata: PaperMetadata) -> DownloadResult:
        """
        Download a paper from ArXiv with hierarchical organization

        Args:
            metadata: Paper metadata

        Returns:
            Download result
        """
        try:
            # Get paper ID
            paper_id = metadata.extra_data.get("arxiv_id", metadata.paper_id)
            
            # Get mapping for this paper
            mapping = self.mapper.map_paper(metadata)
            
            # Build hierarchical path
            path_components = [self.mapper.get_mapper_name()] + mapping.path_components
            paper_dir = self.output_dir.joinpath(*path_components)
            paper_dir.mkdir(parents=True, exist_ok=True)
            
            # Create safe filename
            safe_title = self._sanitize_filename(metadata.title)
            # Replace slashes in paper_id to avoid path issues
            safe_paper_id = paper_id.replace('/', '_')
            filename = f"arxiv_{safe_paper_id}_{safe_title}.pdf"
            filepath = paper_dir / filename

            # Check if file already exists
            if filepath.exists() and filepath.stat().st_size > 1000:
                logger.info(f"Paper already exists: {filepath}")
                return DownloadResult(
                    success=True,
                    paper_id=paper_id,
                    source=self.api_name,
                    file_path=filepath,
                    metadata=metadata,
                    file_size=filepath.stat().st_size,
                )

            # Search for the specific paper
            search = arxiv.Search(id_list=[paper_id])
            paper = next(search.results())

            # Download PDF to the hierarchical location
            paper.download_pdf(dirpath=str(paper_dir), filename=filename)

            # Get file size
            file_size = filepath.stat().st_size if filepath.exists() else None
            
            # Save metadata alongside the PDF
            self._save_metadata(paper_dir, paper_id, metadata, mapping)

            return DownloadResult(
                success=True,
                paper_id=paper_id,
                source=self.api_name,
                file_path=filepath,
                metadata=metadata,
                file_size=file_size,
            )

        except Exception as e:
            logger.error(f"Error downloading paper {metadata.paper_id}: {e}")
            return DownloadResult(
                success=False,
                paper_id=metadata.paper_id,
                source=self.api_name,
                error_message=str(e),
            )

    def _sanitize_filename(self, title: str, max_length: int = 50) -> str:
        """Create safe filename from paper title"""
        import re
        # Remove special characters except spaces, hyphens, underscores
        safe = re.sub(r'[^\w\s-]', '', title)
        # Replace spaces with underscores
        safe = safe.replace(' ', '_')
        # Truncate to max length
        safe = safe[:max_length]
        # Remove trailing underscores
        return safe.rstrip('_')
    
    def _save_metadata(self, paper_dir: Path, paper_id: str, metadata: PaperMetadata, mapping) -> None:
        """Save metadata JSON alongside the PDF"""
        # Replace slashes in paper_id to avoid path issues
        safe_paper_id = paper_id.replace('/', '_')
        metadata_file = paper_dir / f"arxiv_{safe_paper_id}_metadata.json"
        
        metadata_dict = {
            "version": "2.0",
            "download_info": {
                "timestamp": datetime.now().isoformat(),
                "tool_version": "1.1.0",
                "mapper_used": self.mapper.get_mapper_name(),
                "organization_path": mapping.path_components
            },
            "paper_info": {
                "paper_id": metadata.paper_id,
                "title": metadata.title,
                "authors": metadata.authors,
                "abstract": metadata.abstract,
                "year": metadata.year,
                "venue": metadata.venue,
                "url": metadata.url,
                "pdf_url": metadata.pdf_url
            },
            "classification": {
                "arxiv_categories": metadata.categories,
                "mapping_confidence": mapping.confidence,
                "mapping_method": mapping.method,
                "mapping_metadata": mapping.metadata
            },
            "extra_data": metadata.extra_data
        }
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved metadata to {metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def collect_by_category(
        self, category: str, max_papers: int = 10
    ) -> list[DownloadResult]:
        """
        Collect papers from a specific ArXiv category

        Args:
            category: ArXiv category (e.g., "cs.LG", "cs.AI")
            max_papers: Maximum papers to download

        Returns:
            List of download results
        """
        return self.collect_papers(f"cat:{category}", max_papers)
    
    def collect_papers(self, query: str, max_papers: int) -> list[DownloadResult]:
        """
        Collect papers based on query.
        
        Special query formats:
        - "cat:cs.LG" - Standard ArXiv category search
        - "cc2020:DS" - Search all categories for a CC2020 discipline
        """
        if query.startswith("cc2020:"):
            # Special handling for CC2020 discipline search
            discipline = query.split(":")[1]
            return self._collect_cc2020_discipline(discipline, max_papers)
        else:
            # Standard search
            return super().collect_papers(query, max_papers)
    
    def _collect_cc2020_discipline(self, discipline: str, max_papers: int) -> list[DownloadResult]:
        """Collect papers from all ArXiv categories mapped to a CC2020 discipline"""
        # Only works if using CC2020 mapper
        if self.mapper.get_mapper_name() != "CC2020":
            logger.warning(f"CC2020 discipline search requires CC2020 mapper, using {self.mapper.get_mapper_name()}")
            return []
        
        # Get categories for this discipline from the mapper
        from ..mappers.cc2020_mapper import CC2020Mapper
        discipline_categories = CC2020Mapper.DISCIPLINE_CATEGORIES.get(discipline, [])
        
        if not discipline_categories:
            logger.error(f"Unknown CC2020 discipline: {discipline}")
            return []
        
        logger.info(f"Searching {len(discipline_categories)} categories for {discipline} papers")
        
        # Distribute max_papers across categories
        papers_per_category = max(1, max_papers // len(discipline_categories))
        remaining = max_papers - (papers_per_category * len(discipline_categories))
        
        all_results = []
        for i, category in enumerate(discipline_categories):
            # Add extra papers to first categories if there's a remainder
            category_max = papers_per_category + (1 if i < remaining else 0)
            
            logger.info(f"Searching {category} for {discipline} papers (max: {category_max})")
            papers = self.search(f"cat:{category}", category_max)
            
            # Download each paper (mapper will organize them)
            for paper in papers:
                if len(all_results) >= max_papers:
                    break
                result = self.download_paper(paper)
                if result.success:
                    all_results.append(result)
            
            if len(all_results) >= max_papers:
                break
        
        logger.info(f"Downloaded {len(all_results)}/{max_papers} {discipline} papers")
        return all_results[:max_papers]
