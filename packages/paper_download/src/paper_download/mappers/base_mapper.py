"""Base mapper interface for all category mappers"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from ..collectors.base_collector import PaperMetadata


@dataclass
class MappingResult:
    """Result of mapping a paper to a category structure"""
    path_components: List[str]  # Folder hierarchy components
    confidence: float           # Classification confidence (0.0 to 1.0)
    method: str                # Classification method used
    metadata: Dict[str, Any]   # Additional mapping metadata


class BaseMapper(ABC):
    """Abstract interface for all category mappers"""
    
    @abstractmethod
    def get_mapper_name(self) -> str:
        """
        Return the root directory name for this mapper.
        
        Examples: 'CC2020', 'ArXiv', 'ACM'
        
        Returns:
            Root directory name for organizing papers
        """
        pass
    
    @abstractmethod
    def map_paper(self, metadata: PaperMetadata) -> MappingResult:
        """
        Map a paper to the appropriate folder structure.
        
        Args:
            metadata: Paper metadata including categories, title, abstract
            
        Returns:
            MappingResult with path components and classification details
        """
        pass
    
    @abstractmethod
    def get_category_description(self, category: str) -> str:
        """
        Get human-readable description for a category.
        
        Args:
            category: Category code (e.g., 'CS', 'cs.AI')
            
        Returns:
            Description string for the category
        """
        pass
    
    @abstractmethod
    def list_categories(self) -> Dict[str, List[str]]:
        """
        List all available categories in this mapping scheme.
        
        Returns:
            Dictionary of categories organized by level/type
        """
        pass
    
    def post_download_hook(self, paper_path: str, metadata: Dict[str, Any]) -> None:
        """
        Optional hook called after paper download.
        Override to add custom post-processing.
        
        Args:
            paper_path: Path where paper was saved
            metadata: Paper metadata and mapping information
        """
        pass
    
    def pre_classify_hook(self, metadata: PaperMetadata) -> PaperMetadata:
        """
        Optional hook called before classification.
        Override to modify metadata before classification.
        
        Args:
            metadata: Original paper metadata
            
        Returns:
            Modified paper metadata
        """
        return metadata