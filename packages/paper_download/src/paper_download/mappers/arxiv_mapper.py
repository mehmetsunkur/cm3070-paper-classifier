"""ArXiv category-based mapper implementation"""

import logging
from typing import Dict, List, Any

from .base_mapper import BaseMapper, MappingResult
from ..collectors.base_collector import PaperMetadata

logger = logging.getLogger(__name__)


class ArXivMapper(BaseMapper):
    """Simple ArXiv category-based organization"""
    
    # ArXiv category descriptions
    CATEGORY_DESCRIPTIONS = {
        # Computer Science
        "cs.AI": "Artificial Intelligence",
        "cs.AR": "Hardware Architecture",
        "cs.CC": "Computational Complexity",
        "cs.CE": "Computational Engineering",
        "cs.CG": "Computational Geometry",
        "cs.CL": "Computation and Language",
        "cs.CR": "Cryptography and Security",
        "cs.CV": "Computer Vision and Pattern Recognition",
        "cs.CY": "Computers and Society",
        "cs.DB": "Databases",
        "cs.DC": "Distributed, Parallel, and Cluster Computing",
        "cs.DL": "Digital Libraries",
        "cs.DM": "Discrete Mathematics",
        "cs.DS": "Data Structures and Algorithms",
        "cs.ET": "Emerging Technologies",
        "cs.FL": "Formal Languages and Automata Theory",
        "cs.GL": "General Literature",
        "cs.GR": "Graphics",
        "cs.GT": "Computer Science and Game Theory",
        "cs.HC": "Human-Computer Interaction",
        "cs.IR": "Information Retrieval",
        "cs.IT": "Information Theory",
        "cs.LG": "Machine Learning",
        "cs.LO": "Logic in Computer Science",
        "cs.MA": "Multiagent Systems",
        "cs.MM": "Multimedia",
        "cs.MS": "Mathematical Software",
        "cs.NA": "Numerical Analysis",
        "cs.NE": "Neural and Evolutionary Computing",
        "cs.NI": "Networking and Internet Architecture",
        "cs.OH": "Other Computer Science",
        "cs.OS": "Operating Systems",
        "cs.PF": "Performance",
        "cs.PL": "Programming Languages",
        "cs.RO": "Robotics",
        "cs.SC": "Symbolic Computation",
        "cs.SD": "Sound",
        "cs.SE": "Software Engineering",
        "cs.SI": "Social and Information Networks",
        "cs.SY": "Systems and Control",
        # Statistics
        "stat.AP": "Applications",
        "stat.CO": "Computation",
        "stat.ME": "Methodology",
        "stat.ML": "Machine Learning",
        "stat.OT": "Other Statistics",
        "stat.TH": "Theory",
        # Mathematics
        "math.NA": "Numerical Analysis",
        "math.OC": "Optimization and Control",
        # Electrical Engineering
        "eess.AS": "Audio and Speech Processing",
        "eess.IV": "Image and Video Processing",
        "eess.SP": "Signal Processing",
        "eess.SY": "Systems and Control",
    }
    
    def __init__(self, include_subcategories: bool = True):
        """
        Initialize ArXiv Mapper.
        
        Args:
            include_subcategories: If True, papers with multiple categories 
                                  get subdirectory for secondary categories
        """
        self.include_subcategories = include_subcategories
        self.stats = {
            "total_mapped": 0,
            "unknown_categories": 0,
            "categories_seen": set()
        }
    
    def get_mapper_name(self) -> str:
        """Return the root directory name for ArXiv organization"""
        return "ArXiv"
    
    def map_paper(self, metadata: PaperMetadata) -> MappingResult:
        """
        Map a paper to ArXiv category structure.
        
        Args:
            metadata: Paper metadata
            
        Returns:
            MappingResult with category path
        """
        self.stats["total_mapped"] += 1
        
        # Get primary category
        if metadata.categories and len(metadata.categories) > 0:
            primary_category = metadata.categories[0]
            self.stats["categories_seen"].add(primary_category)
            
            # Check if category is known
            if primary_category not in self.CATEGORY_DESCRIPTIONS:
                self.stats["unknown_categories"] += 1
                logger.info(f"Unknown ArXiv category: {primary_category}")
            
            path_components = [primary_category]
            
            # Optionally add subcategory folder for papers with multiple categories
            if self.include_subcategories and len(metadata.categories) > 1:
                secondary_categories = "_".join(metadata.categories[1:3])  # Max 2 secondary
                if secondary_categories:
                    path_components.append(f"also_{secondary_categories}")
            
            confidence = 1.0  # ArXiv categories are explicit, not inferred
        else:
            # No category information
            path_components = ["unknown"]
            confidence = 0.0
            logger.warning(f"Paper {metadata.paper_id} has no category information")
        
        return MappingResult(
            path_components=path_components,
            confidence=confidence,
            method="category",
            metadata={
                "all_categories": metadata.categories if metadata.categories else [],
                "primary_category": path_components[0],
                "stats": self.get_stats()
            }
        )
    
    def get_category_description(self, category: str) -> str:
        """Get description for an ArXiv category"""
        return self.CATEGORY_DESCRIPTIONS.get(category, f"Unknown category: {category}")
    
    def list_categories(self) -> Dict[str, List[str]]:
        """List all ArXiv categories by domain"""
        categories_by_domain = {
            "computer_science": [],
            "statistics": [],
            "mathematics": [],
            "electrical_engineering": [],
            "other": []
        }
        
        for cat in sorted(self.CATEGORY_DESCRIPTIONS.keys()):
            if cat.startswith("cs."):
                categories_by_domain["computer_science"].append(cat)
            elif cat.startswith("stat."):
                categories_by_domain["statistics"].append(cat)
            elif cat.startswith("math."):
                categories_by_domain["mathematics"].append(cat)
            elif cat.startswith("eess."):
                categories_by_domain["electrical_engineering"].append(cat)
            else:
                categories_by_domain["other"].append(cat)
        
        return categories_by_domain
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mapping statistics"""
        return {
            "total_mapped": self.stats["total_mapped"],
            "unknown_categories": self.stats["unknown_categories"],
            "unique_categories": len(self.stats["categories_seen"]),
            "categories_seen": sorted(list(self.stats["categories_seen"]))
        }