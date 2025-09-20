"""CC2020 computing curriculum mapper implementation"""

import logging
from typing import Dict, List, Any

from .base_mapper import BaseMapper, MappingResult
from ..collectors.base_collector import PaperMetadata
from ..utils.cc2020_mapper import CC2020Mapper as CC2020Classifier, CC2020Classification

logger = logging.getLogger(__name__)


class CC2020Mapper(BaseMapper):
    """Maps papers to CC2020 computing curriculum disciplines"""
    
    # Discipline full names for folder creation
    DISCIPLINE_NAMES = {
        "CS": "Computer_Science",
        "SE": "Software_Engineering", 
        "CE": "Computer_Engineering",
        "IS": "Information_Systems",
        "IT": "Information_Technology",
        "CSEC": "Cybersecurity",
        "DS": "Data_Science"
    }
    
    # ArXiv categories mapped to each CC2020 discipline
    DISCIPLINE_CATEGORIES = {
        "CS": ["cs.AI", "cs.CV", "cs.CL", "cs.CC", "cs.DS", "cs.LO", "cs.DM", 
               "cs.FL", "cs.GT", "cs.CG", "cs.SC", "cs.MS", "cs.NA", "cs.RO",
               "cs.NE", "cs.MA", "cs.GR", "cs.HC", "cs.MM", "cs.SD", "cs.IT"],
        "SE": ["cs.SE", "cs.PL", "cs.GL"],
        "CE": ["cs.AR", "cs.ET", "cs.OH"],
        "IS": ["cs.CY", "cs.DL", "cs.SI"],
        "IT": ["cs.NI", "cs.DC", "cs.OS", "cs.SY", "cs.PF", "eess.SY"],
        "CSEC": ["cs.CR"],
        "DS": ["cs.LG", "stat.ML", "cs.DB", "cs.IR", "stat.CO", "stat.AP"]
    }
    
    def __init__(self, strict_mode: bool = False, min_confidence: float = 0.3):
        """
        Initialize CC2020 Mapper.
        
        Args:
            strict_mode: If True, require higher confidence for classification
            min_confidence: Minimum confidence threshold for classification
        """
        self.classifier = CC2020Classifier(strict_mode=strict_mode)
        self.min_confidence = min_confidence
        self.stats = {
            "total_mapped": 0,
            "low_confidence": 0,
            "unclassified": 0
        }
    
    def get_mapper_name(self) -> str:
        """Return the root directory name for CC2020 organization"""
        return "CC2020"
    
    def map_paper(self, metadata: PaperMetadata) -> MappingResult:
        """
        Map a paper to CC2020 discipline hierarchy.
        
        Args:
            metadata: Paper metadata
            
        Returns:
            MappingResult with hierarchical path components
        """
        self.stats["total_mapped"] += 1
        
        # Use the existing CC2020 classification logic
        classification = self.classifier.classify(
            categories=metadata.categories,
            title=metadata.title,
            abstract=metadata.abstract,
            keywords=metadata.extra_data.get("keywords", [])
        )
        
        # Check confidence threshold
        if classification.confidence < self.min_confidence:
            self.stats["unclassified"] += 1
            logger.warning(
                f"Low confidence classification for {metadata.paper_id}: "
                f"{classification.primary_discipline} (confidence: {classification.confidence:.2f})"
            )
            # Put in Unclassified folder
            path_components = ["Unclassified", "low_confidence"]
        elif classification.confidence < 0.7:
            self.stats["low_confidence"] += 1
            # Normal classification but log warning
            logger.info(
                f"Medium confidence classification for {metadata.paper_id}: "
                f"{classification.primary_discipline} (confidence: {classification.confidence:.2f})"
            )
            path_components = self._build_path(classification, metadata)
        else:
            # High confidence classification
            path_components = self._build_path(classification, metadata)
        
        return MappingResult(
            path_components=path_components,
            confidence=classification.confidence,
            method=classification.classification_method,
            metadata={
                "primary_discipline": classification.primary_discipline,
                "secondary_disciplines": classification.secondary_disciplines,
                "evidence": classification.evidence,
                "stats": self.get_stats()
            }
        )
    
    def _build_path(self, classification: CC2020Classification, metadata: PaperMetadata) -> List[str]:
        """Build hierarchical path from classification"""
        discipline = classification.primary_discipline
        discipline_name = self.DISCIPLINE_NAMES.get(discipline, discipline)
        discipline_folder = f"{discipline}_{discipline_name}"
        
        # Get primary ArXiv category if available
        if metadata.categories:
            primary_category = metadata.categories[0]
        else:
            primary_category = "unknown"
        
        return [discipline_folder, primary_category]
    
    def get_category_description(self, category: str) -> str:
        """Get description for a CC2020 discipline or ArXiv category"""
        # CC2020 discipline descriptions
        discipline_descriptions = {
            "CS": "Computer Science - Theory, algorithms, AI, graphics, HCI",
            "SE": "Software Engineering - Development, testing, methodologies",
            "CE": "Computer Engineering - Hardware/software interface, embedded systems",
            "IS": "Information Systems - Business computing, enterprise systems",
            "IT": "Information Technology - Infrastructure, operations, networking",
            "CSEC": "Cybersecurity - Security, privacy, cryptography",
            "DS": "Data Science - Analytics, ML applications, big data"
        }
        
        if category in discipline_descriptions:
            return discipline_descriptions[category]
        
        # ArXiv category descriptions
        arxiv_descriptions = {
            "cs.AI": "Artificial Intelligence",
            "cs.LG": "Machine Learning",
            "cs.SE": "Software Engineering",
            "cs.CV": "Computer Vision and Pattern Recognition",
            "cs.CL": "Computation and Language",
            "cs.CR": "Cryptography and Security",
            "cs.DB": "Databases",
            "cs.DS": "Data Structures and Algorithms",
            "cs.HC": "Human-Computer Interaction",
            "cs.NI": "Networking and Internet Architecture",
            "cs.PL": "Programming Languages",
            "cs.RO": "Robotics",
            "stat.ML": "Statistics - Machine Learning",
        }
        
        return arxiv_descriptions.get(category, category)
    
    def list_categories(self) -> Dict[str, List[str]]:
        """List CC2020 disciplines and their mapped ArXiv categories"""
        return {
            "disciplines": list(self.DISCIPLINE_NAMES.keys()),
            "discipline_mapping": self.DISCIPLINE_CATEGORIES,
            "all_categories": self._get_all_arxiv_categories()
        }
    
    def _get_all_arxiv_categories(self) -> List[str]:
        """Get all unique ArXiv categories across disciplines"""
        all_cats = set()
        for cats in self.DISCIPLINE_CATEGORIES.values():
            all_cats.update(cats)
        return sorted(list(all_cats))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mapping statistics"""
        return {
            "total_mapped": self.stats["total_mapped"],
            "low_confidence": self.stats["low_confidence"],
            "unclassified": self.stats["unclassified"],
            "confidence_distribution": self._get_confidence_distribution()
        }
    
    def _get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence levels"""
        total = self.stats["total_mapped"]
        if total == 0:
            return {}
        
        high = total - self.stats["low_confidence"] - self.stats["unclassified"]
        return {
            "high (>0.7)": high,
            "medium (0.3-0.7)": self.stats["low_confidence"],
            "low (<0.3)": self.stats["unclassified"]
        }