"""CC2020 Taxonomy Mapper for ArXiv Categories

Maps ArXiv categories to CC2020 computing disciplines:
- CS: Computer Science
- SE: Software Engineering  
- CE: Computer Engineering
- IS: Information Systems
- IT: Information Technology
- CSEC: Cybersecurity
- DS: Data Science
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CC2020Classification:
    """Result of CC2020 classification"""
    primary_discipline: str
    confidence: float
    secondary_disciplines: List[str]
    classification_method: str  # "category", "keyword", "hybrid"
    evidence: Dict[str, any]


class CC2020Mapper:
    """Maps ArXiv categories and paper content to CC2020 disciplines"""
    
    # Comprehensive ArXiv to CC2020 mapping
    ARXIV_TO_CC2020 = {
        # Computer Science (Theory, Foundations, Core CS)
        "cs.CC": "CS",     # Computational Complexity
        "cs.LO": "CS",     # Logic in Computer Science
        "cs.DS": "CS",     # Data Structures and Algorithms
        "cs.DM": "CS",     # Discrete Mathematics
        "cs.FL": "CS",     # Formal Languages and Automata Theory
        "cs.GT": "CS",     # Computer Science and Game Theory
        "cs.CG": "CS",     # Computational Geometry
        "cs.SC": "CS",     # Symbolic Computation
        "cs.MS": "CS",     # Mathematical Software
        "cs.NA": "CS",     # Numerical Analysis
        
        # Computer Science (AI/ML Research - theoretical aspects)
        "cs.AI": "CS",     # Artificial Intelligence
        "cs.CV": "CS",     # Computer Vision and Pattern Recognition
        "cs.CL": "CS",     # Computation and Language (NLP)
        "cs.RO": "CS",     # Robotics
        "cs.NE": "CS",     # Neural and Evolutionary Computing
        "cs.MA": "CS",     # Multiagent Systems
        "cs.GR": "CS",     # Graphics
        "cs.HC": "CS",     # Human-Computer Interaction (research focus)
        "cs.MM": "CS",     # Multimedia
        "cs.SD": "CS",     # Sound
        "cs.IT": "CS",     # Information Theory
        
        # Software Engineering
        "cs.SE": "SE",     # Software Engineering
        "cs.PL": "SE",     # Programming Languages
        "cs.GL": "SE",     # General Literature (often SE-related)
        
        # Computer Engineering (Hardware/Software Interface)
        "cs.AR": "CE",     # Hardware Architecture
        "cs.ET": "CE",     # Emerging Technologies
        "cs.OH": "CE",     # Other Computer Science (often hardware)
        
        # Cybersecurity
        "cs.CR": "CSEC",   # Cryptography and Security
        
        # Data Science (Applied ML/Analytics)
        "cs.LG": "DS",     # Machine Learning
        "cs.DB": "DS",     # Databases (when data-focused)
        "cs.IR": "DS",     # Information Retrieval
        "stat.ML": "DS",   # Statistics - Machine Learning
        "stat.CO": "DS",   # Statistics - Computation
        "stat.AP": "DS",   # Statistics - Applications
        
        # Information Technology (Infrastructure/Operations)
        "cs.NI": "IT",     # Networking and Internet Architecture
        "cs.DC": "IT",     # Distributed, Parallel, and Cluster Computing
        "cs.SY": "IT",     # Systems and Control
        "cs.OS": "IT",     # Operating Systems
        "cs.PF": "IT",     # Performance
        "eess.SY": "IT",   # Electrical Engineering - Systems and Control
        
        # Information Systems (Business/Organizational)
        "cs.CY": "IS",     # Computers and Society
        "cs.DL": "IS",     # Digital Libraries
        "cs.SI": "IS",     # Social and Information Networks
        
        # Cross-disciplinary (requires additional context)
        "cs.CE": "CS",     # Computational Engineering (could be CE)
        "math.LO": "CS",   # Mathematical Logic
        "math.CO": "CS",   # Combinatorics
        "q-bio": "DS",     # Quantitative Biology (when computational)
        "physics.comp-ph": "CS",  # Computational Physics
    }
    
    # Secondary mappings for interdisciplinary papers
    SECONDARY_MAPPINGS = {
        "cs.LG": ["CS", "DS"],      # ML can be theoretical or applied
        "cs.AI": ["CS", "DS"],       # AI similar
        "cs.DB": ["DS", "IT"],       # Databases span DS and IT
        "cs.HC": ["CS", "IS"],       # HCI spans CS and IS
        "cs.SE": ["SE", "CS"],       # SE has CS foundations
        "cs.CR": ["CSEC", "CS"],     # Security has theoretical aspects
        "cs.SI": ["IS", "DS"],       # Social networks span IS and DS
    }
    
    # Keyword indicators for disciplines (especially IS/IT disambiguation)
    DISCIPLINE_KEYWORDS = {
        "IS": {
            "strong": [
                "enterprise system", "business process", "ERP", "CRM", "SCM",
                "organizational", "management information", "decision support",
                "business intelligence", "knowledge management", "stakeholder",
                "digital transformation", "change management", "IT governance",
                "strategic information", "competitive advantage", "business value"
            ],
            "moderate": [
                "information management", "business analytics", "workflow",
                "enterprise architecture", "business model", "organizational learning",
                "technology adoption", "user acceptance", "TAM", "UTAUT"
            ],
            "weak": [
                "management", "organization", "business", "enterprise", "strategic"
            ]
        },
        "IT": {
            "strong": [
                "network administration", "system administration", "infrastructure",
                "DevOps", "cloud deployment", "IT service", "ITIL", "virtualization",
                "container orchestration", "kubernetes", "docker", "monitoring",
                "backup", "disaster recovery", "server management", "data center"
            ],
            "moderate": [
                "deployment", "configuration management", "automation", "provisioning",
                "load balancing", "scalability", "availability", "reliability",
                "fault tolerance", "system maintenance", "patch management"
            ],
            "weak": [
                "network", "system", "infrastructure", "operations", "maintenance"
            ]
        },
        "SE": {
            "strong": [
                "software development", "agile", "scrum", "testing", "debugging",
                "refactoring", "code review", "version control", "git", "CI/CD",
                "software architecture", "design patterns", "requirements engineering",
                "software quality", "technical debt", "code smell", "unit test"
            ],
            "moderate": [
                "software design", "implementation", "software process", "SDLC",
                "software metrics", "code coverage", "integration testing",
                "software maintenance", "software evolution"
            ],
            "weak": [
                "development", "programming", "coding", "software", "application"
            ]
        },
        "DS": {
            "strong": [
                "data science", "machine learning", "deep learning", "neural network",
                "data mining", "predictive analytics", "feature engineering",
                "model training", "classification", "regression", "clustering",
                "dimensionality reduction", "cross-validation", "hyperparameter"
            ],
            "moderate": [
                "data analysis", "statistical modeling", "data visualization",
                "big data", "data pipeline", "ETL", "data warehouse", "data lake",
                "business analytics", "predictive modeling"
            ],
            "weak": [
                "data", "analytics", "model", "prediction", "learning"
            ]
        },
        "CSEC": {
            "strong": [
                "cybersecurity", "cryptography", "encryption", "vulnerability",
                "penetration testing", "security audit", "threat modeling",
                "authentication", "authorization", "access control", "PKI",
                "zero trust", "security operations", "incident response", "SIEM"
            ],
            "moderate": [
                "security", "privacy", "confidentiality", "integrity", "availability",
                "secure coding", "security testing", "firewall", "IDS", "IPS",
                "malware", "ransomware", "phishing"
            ],
            "weak": [
                "secure", "attack", "defense", "protection", "threat"
            ]
        }
    }
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize CC2020 Mapper
        
        Args:
            strict_mode: If True, require higher confidence for classification
        """
        self.strict_mode = strict_mode
        self.stats = {
            "total_classified": 0,
            "category_based": 0,
            "keyword_based": 0,
            "hybrid": 0,
            "low_confidence": 0
        }
    
    def classify(
        self,
        categories: List[str] = None,
        title: str = "",
        abstract: str = "",
        keywords: List[str] = None
    ) -> CC2020Classification:
        """
        Classify paper into CC2020 discipline
        
        Args:
            categories: List of ArXiv categories
            title: Paper title
            abstract: Paper abstract
            keywords: Author-provided keywords
            
        Returns:
            CC2020Classification object with results
        """
        self.stats["total_classified"] += 1
        
        # Try category-based classification first
        if categories:
            cat_result = self._classify_by_category(categories)
            if cat_result.confidence >= (0.8 if self.strict_mode else 0.6):
                self.stats["category_based"] += 1
                return cat_result
        
        # Try keyword-based classification
        text = f"{title} {abstract}".lower()
        if keywords:
            text += " " + " ".join(keywords).lower()
        
        keyword_result = self._classify_by_keywords(text)
        
        # Combine results if we have both
        if categories and keyword_result.confidence > 0.3:
            self.stats["hybrid"] += 1
            return self._combine_classifications(cat_result, keyword_result)
        elif keyword_result.confidence > 0.3:
            self.stats["keyword_based"] += 1
            return keyword_result
        elif categories:
            self.stats["category_based"] += 1
            return cat_result
        else:
            self.stats["low_confidence"] += 1
            return CC2020Classification(
                primary_discipline="CS",  # Default to CS
                confidence=0.2,
                secondary_disciplines=[],
                classification_method="default",
                evidence={"reason": "No strong indicators found"}
            )
    
    def _classify_by_category(self, categories: List[str]) -> CC2020Classification:
        """Classify based on ArXiv categories"""
        discipline_scores = {}
        evidence = {"categories": categories, "mappings": {}}
        
        for cat in categories:
            if cat in self.ARXIV_TO_CC2020:
                discipline = self.ARXIV_TO_CC2020[cat]
                discipline_scores[discipline] = discipline_scores.get(discipline, 0) + 1
                evidence["mappings"][cat] = discipline
                
                # Check for secondary disciplines
                if cat in self.SECONDARY_MAPPINGS:
                    for sec_disc in self.SECONDARY_MAPPINGS[cat]:
                        if sec_disc != discipline:
                            discipline_scores[sec_disc] = discipline_scores.get(sec_disc, 0) + 0.5
        
        if not discipline_scores:
            return CC2020Classification(
                primary_discipline="CS",
                confidence=0.3,
                secondary_disciplines=[],
                classification_method="category",
                evidence={"reason": "No matching ArXiv categories"}
            )
        
        # Sort disciplines by score
        sorted_disciplines = sorted(discipline_scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_disciplines[0][0]
        secondary = [d for d, s in sorted_disciplines[1:] if s >= 0.5]
        
        # Calculate confidence based on number of matching categories
        confidence = min(0.95, 0.7 + (0.1 * len(evidence["mappings"])))
        
        return CC2020Classification(
            primary_discipline=primary,
            confidence=confidence,
            secondary_disciplines=secondary,
            classification_method="category",
            evidence=evidence
        )
    
    def _classify_by_keywords(self, text: str) -> CC2020Classification:
        """Classify based on keyword analysis"""
        discipline_scores = {}
        evidence = {"keywords_found": {}}
        
        for discipline, keyword_dict in self.DISCIPLINE_KEYWORDS.items():
            score = 0
            found_keywords = []
            
            # Check strong indicators
            for keyword in keyword_dict["strong"]:
                if keyword in text:
                    score += 3
                    found_keywords.append(keyword)
            
            # Check moderate indicators
            for keyword in keyword_dict["moderate"]:
                if keyword in text:
                    score += 2
                    found_keywords.append(keyword)
            
            # Check weak indicators
            for keyword in keyword_dict["weak"]:
                if keyword in text:
                    score += 1
                    found_keywords.append(keyword)
            
            if score > 0:
                discipline_scores[discipline] = score
                evidence["keywords_found"][discipline] = found_keywords[:5]  # Top 5
        
        if not discipline_scores:
            return CC2020Classification(
                primary_discipline="CS",
                confidence=0.2,
                secondary_disciplines=[],
                classification_method="keyword",
                evidence={"reason": "No discipline keywords found"}
            )
        
        # Sort disciplines by score
        sorted_disciplines = sorted(discipline_scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_disciplines[0][0]
        primary_score = sorted_disciplines[0][1]
        secondary = [d for d, s in sorted_disciplines[1:] if s >= primary_score * 0.5]
        
        # Calculate confidence based on keyword matches
        confidence = min(0.9, 0.3 + (0.05 * primary_score))
        
        return CC2020Classification(
            primary_discipline=primary,
            confidence=confidence,
            secondary_disciplines=secondary,
            classification_method="keyword",
            evidence=evidence
        )
    
    def _combine_classifications(
        self,
        cat_result: CC2020Classification,
        keyword_result: CC2020Classification
    ) -> CC2020Classification:
        """Combine category and keyword classifications"""
        # If they agree, boost confidence
        if cat_result.primary_discipline == keyword_result.primary_discipline:
            confidence = min(0.98, (cat_result.confidence + keyword_result.confidence) / 1.8)
            primary = cat_result.primary_discipline
        else:
            # Use the one with higher confidence
            if cat_result.confidence > keyword_result.confidence:
                primary = cat_result.primary_discipline
                confidence = cat_result.confidence * 0.9  # Slight penalty for disagreement
            else:
                primary = keyword_result.primary_discipline
                confidence = keyword_result.confidence * 0.9
        
        # Merge secondary disciplines
        secondary = list(set(cat_result.secondary_disciplines + keyword_result.secondary_disciplines))
        if cat_result.primary_discipline != primary:
            secondary.append(cat_result.primary_discipline)
        if keyword_result.primary_discipline != primary and keyword_result.primary_discipline not in secondary:
            secondary.append(keyword_result.primary_discipline)
        
        return CC2020Classification(
            primary_discipline=primary,
            confidence=confidence,
            secondary_disciplines=secondary[:3],  # Limit to top 3
            classification_method="hybrid",
            evidence={
                "category_evidence": cat_result.evidence,
                "keyword_evidence": keyword_result.evidence
            }
        )
    
    def get_discipline_name(self, code: str) -> str:
        """Get full discipline name from code"""
        names = {
            "CS": "Computer Science",
            "SE": "Software Engineering",
            "CE": "Computer Engineering",
            "IS": "Information Systems",
            "IT": "Information Technology",
            "CSEC": "Cybersecurity",
            "DS": "Data Science"
        }
        return names.get(code, code)
    
    def get_stats(self) -> Dict[str, any]:
        """Get classification statistics"""
        total = max(self.stats["total_classified"], 1)
        return {
            "total_classified": self.stats["total_classified"],
            "category_based": self.stats["category_based"],
            "keyword_based": self.stats["keyword_based"],
            "hybrid": self.stats["hybrid"],
            "low_confidence": self.stats["low_confidence"],
            "category_rate": (self.stats["category_based"] / total) * 100,
            "keyword_rate": (self.stats["keyword_based"] / total) * 100,
            "hybrid_rate": (self.stats["hybrid"] / total) * 100
        }
    
    def validate_mapping(self) -> List[str]:
        """Validate that all mappings are to valid CC2020 disciplines"""
        valid_disciplines = {"CS", "SE", "CE", "IS", "IT", "CSEC", "DS"}
        issues = []
        
        for arxiv_cat, discipline in self.ARXIV_TO_CC2020.items():
            if discipline not in valid_disciplines:
                issues.append(f"Invalid discipline '{discipline}' for category '{arxiv_cat}'")
        
        return issues


def map_arxiv_to_cc2020(categories: List[str]) -> str:
    """
    Simple function to map ArXiv categories to CC2020 discipline
    
    Args:
        categories: List of ArXiv categories
        
    Returns:
        Primary CC2020 discipline code
    """
    mapper = CC2020Mapper()
    result = mapper.classify(categories=categories)
    return result.primary_discipline