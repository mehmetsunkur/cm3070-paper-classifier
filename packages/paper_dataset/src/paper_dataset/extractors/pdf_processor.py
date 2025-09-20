"""
PDF processing pipeline for extracting and preparing text from academic papers.
Handles various PDF formats, OCR, and metadata extraction.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

import pdfplumber
import pymupdf
from pdf2image import convert_from_path
import pytesseract
import numpy as np
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import spacy
from loguru import logger


@dataclass
class PaperMetadata:
    """Metadata extracted from academic papers."""
    paper_id: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    sections: Optional[Dict[str, str]] = None
    references: Optional[List[str]] = None
    figures_count: int = 0
    tables_count: int = 0
    equations_count: int = 0
    pages: int = 0
    word_count: int = 0
    char_count: int = 0


class PDFExtractor:
    """Extract text and metadata from PDF files."""
    
    def __init__(
        self,
        use_ocr: bool = True,
        ocr_threshold: float = 0.1,
        extract_metadata: bool = True,
        extract_structure: bool = True,
        cache_dir: Optional[Path] = None
    ):
        self.use_ocr = use_ocr
        self.ocr_threshold = ocr_threshold
        self.extract_metadata = extract_metadata
        self.extract_structure = extract_structure
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache/pdf_extracted")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load spaCy model for NER (optional)
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not available. NLP features disabled.")
            self.nlp = None
    
    def _safe_extract_metadata_field(self, value: Any) -> Optional[str]:
        """Safely convert metadata field to string, handling PSKeyword and other special types."""
        if value is None:
            return None
        try:
            # Handle PSKeyword objects and other special PDF types
            str_value = str(value).strip()
            return str_value if str_value else None
        except Exception as e:
            logger.warning(f"Failed to convert metadata field: {e}")
            return None
            
    def extract(self, pdf_path: Path) -> Tuple[str, PaperMetadata]:
        """Extract text and metadata from PDF."""
        
        # Check cache
        cache_key = self._get_cache_key(pdf_path)
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            return cached_result
            
        try:
            # Try pdfplumber first (better for tables/structure)
            text, metadata = self._extract_with_pdfplumber(pdf_path)
            
            # If text extraction is poor, try PyMuPDF
            if self._is_text_quality_poor(text):
                text_mupdf, metadata_mupdf = self._extract_with_pymupdf(pdf_path)
                if len(text_mupdf) > len(text):
                    text, metadata = text_mupdf, metadata_mupdf
                    
            # OCR if needed
            if self.use_ocr and self._needs_ocr(text, metadata):
                ocr_text = self._perform_ocr(pdf_path)
                if ocr_text:
                    text = self._merge_ocr_text(text, ocr_text)
                    
            # Extract additional metadata
            if self.extract_metadata:
                metadata = self._enhance_metadata(text, metadata)
                
            # Extract document structure
            if self.extract_structure:
                sections = self._extract_sections(text)
                metadata.sections = sections
                
            # Save to cache
            self._save_to_cache(cache_key, text, metadata)
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Failed to extract PDF {pdf_path}: {e}")
            return "", PaperMetadata(paper_id=str(pdf_path))
            
    def _extract_with_pdfplumber(self, pdf_path: Path) -> Tuple[str, PaperMetadata]:
        """Extract using pdfplumber."""
        text_parts = []
        metadata = PaperMetadata(paper_id=pdf_path.stem)
        
        with pdfplumber.open(pdf_path) as pdf:
            metadata.pages = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages):
                # Extract text
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                
                # Extract tables
                tables = page.extract_tables()
                metadata.tables_count += len(tables) if tables else 0
                
                # Count figures (approximate)
                if page.images:
                    metadata.figures_count += len(page.images)
                    
            # Extract metadata from first page
            if pdf.metadata:
                metadata.title = self._safe_extract_metadata_field(pdf.metadata.get('Title'))
                author_field = self._safe_extract_metadata_field(pdf.metadata.get('Author'))
                metadata.authors = self._parse_authors(author_field or '')
                
        text = "\n".join(text_parts)
        metadata.word_count = len(text.split())
        metadata.char_count = len(text)
        
        return text, metadata
        
    def _extract_with_pymupdf(self, pdf_path: Path) -> Tuple[str, PaperMetadata]:
        """Extract using PyMuPDF (better for some PDFs)."""
        text_parts = []
        metadata = PaperMetadata(paper_id=pdf_path.stem)
        
        doc = pymupdf.open(pdf_path)
        metadata.pages = len(doc)
        
        for page_num, page in enumerate(doc):
            # Extract text
            text_parts.append(page.get_text())
            
            # Count images
            image_list = page.get_images()
            metadata.figures_count += len(image_list)
            
        # Extract metadata
        if doc.metadata:
            metadata.title = self._safe_extract_metadata_field(doc.metadata.get('title'))
            author_field = self._safe_extract_metadata_field(doc.metadata.get('author'))
            metadata.authors = self._parse_authors(author_field or '')
            
        doc.close()
        
        text = "\n".join(text_parts)
        metadata.word_count = len(text.split())
        metadata.char_count = len(text)
        
        return text, metadata
        
    def _perform_ocr(self, pdf_path: Path) -> str:
        """Perform OCR on scanned pages."""
        try:
            images = convert_from_path(pdf_path, dpi=300)
            ocr_texts = []
            
            for img in images:
                ocr_text = pytesseract.image_to_string(img)
                ocr_texts.append(ocr_text)
                
            return "\n".join(ocr_texts)
        except Exception as e:
            logger.warning(f"OCR failed for {pdf_path}: {e}")
            return ""
            
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract paper sections (abstract, introduction, etc.)."""
        sections = {}
        
        # Common section patterns
        section_patterns = [
            r'(?i)\n(abstract|introduction|related work|methodology|methods|'
            r'experiments|results|discussion|conclusion|references)\s*\n',
        ]
        
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text))
            
            for i, match in enumerate(matches):
                section_name = match.group(1).lower()
                start = match.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                sections[section_name] = text[start:end].strip()
                
        # Extract abstract if not found
        if 'abstract' not in sections:
            abstract_match = re.search(
                r'(?i)abstract[\s\S]{0,50}?([\s\S]{100,1500})(?:introduction|keywords)',
                text
            )
            if abstract_match:
                sections['abstract'] = abstract_match.group(1).strip()
                
        return sections
        
    def _enhance_metadata(self, text: str, metadata: PaperMetadata) -> PaperMetadata:
        """Enhance metadata using NLP techniques."""
        
        # Extract title if not found
        if not metadata.title:
            # Usually title is in first few lines
            first_lines = text[:500].split('\n')
            for line in first_lines:
                if 10 < len(line) < 200 and not line.isupper():
                    metadata.title = line.strip()
                    break
                    
        # Extract abstract
        if 'abstract' in text.lower():
            abstract_match = re.search(
                r'(?i)abstract[\s\S]{0,50}?([\s\S]{100,1500})',
                text
            )
            if abstract_match:
                metadata.abstract = abstract_match.group(1).strip()
                
        # Extract keywords
        keywords_match = re.search(
            r'(?i)keywords?:?\s*([\s\S]{10,200})',
            text
        )
        if keywords_match:
            keywords_text = keywords_match.group(1)
            metadata.keywords = [k.strip() for k in re.split(r'[,;]', keywords_text)][:10]
            
        # Extract year
        year_matches = re.findall(r'\b(19|20)\d{2}\b', text[:2000])
        if year_matches:
            metadata.year = int(year_matches[0])
            
        # Extract DOI
        doi_match = re.search(r'10\.\d{4,}[/.][\S]+', text)
        if doi_match:
            metadata.doi = doi_match.group(0)
            
        # Extract arXiv ID
        arxiv_match = re.search(r'arXiv:(\d{4}\.\d{4,5})', text)
        if arxiv_match:
            metadata.arxiv_id = arxiv_match.group(1)
            
        # Count equations (LaTeX)
        equation_patterns = [r'\$\$[\s\S]+?\$\$', r'\\\[[\s\S]+?\\\]', r'\\begin\{equation']
        for pattern in equation_patterns:
            metadata.equations_count += len(re.findall(pattern, text))
            
        return metadata
        
    def _is_text_quality_poor(self, text: str) -> bool:
        """Check if extracted text quality is poor."""
        if len(text) < 1000:
            return True
        # Check for garbled text
        if len(re.findall(r'[^\x00-\x7F]+', text)) / len(text) > 0.3:
            return True
        return False
        
    def _needs_ocr(self, text: str, metadata: PaperMetadata) -> bool:
        """Determine if OCR is needed."""
        # If very little text extracted relative to pages
        avg_chars_per_page = metadata.char_count / max(metadata.pages, 1)
        return avg_chars_per_page < 500
        
    def _merge_ocr_text(self, original: str, ocr: str) -> str:
        """Merge OCR text with original extracted text."""
        if len(original) > len(ocr) * 0.8:
            return original
        return ocr
        
    def _parse_authors(self, author_string: str) -> List[str]:
        """Parse author string into list."""
        if not author_string:
            return []
        # Handle various formats
        authors = re.split(r'[,;]|(?:\band\b)', author_string)
        return [a.strip() for a in authors if a.strip()]
        
    def _get_cache_key(self, pdf_path: Path) -> str:
        """Generate cache key for PDF."""
        stat = pdf_path.stat()
        key_string = f"{pdf_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(key_string.encode()).hexdigest()
        
    def _load_from_cache(self, cache_key: str) -> Optional[Tuple[str, PaperMetadata]]:
        """Load from cache if exists."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    metadata = PaperMetadata(**data['metadata'])
                    return data['text'], metadata
            except:
                pass
        return None
        
    def _save_to_cache(self, cache_key: str, text: str, metadata: PaperMetadata):
        """Save to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'text': text,
                    'metadata': asdict(metadata)
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


class PaperDataset:
    """Dataset class for paper classification."""
    
    def __init__(
        self,
        data_dir: Path,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 100_000,
        cache_dir: Optional[Path] = None,
        num_workers: int = 4
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.num_workers = num_workers
        self.extractor = PDFExtractor(cache_dir=cache_dir)
        
        self.papers = []
        self.labels = {}
        
    def load_pdfs(self, pdf_dir: Path, labels_file: Optional[Path] = None):
        """Load PDFs and optionally labels."""
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Load labels if provided
        if labels_file and labels_file.exists():
            with open(labels_file, 'r') as f:
                self.labels = json.load(f)
                
        # Extract PDFs in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(
                executor.map(self.extractor.extract, pdf_files),
                total=len(pdf_files),
                desc="Extracting PDFs"
            ))
            
        for pdf_file, (text, metadata) in zip(pdf_files, results):
            if text:
                self.papers.append({
                    'paper_id': metadata.paper_id,
                    'text': text,
                    'metadata': metadata,
                    'labels': self.labels.get(metadata.paper_id, {})
                })
                
        logger.info(f"Successfully extracted {len(self.papers)} papers")
        
    def tokenize_papers(self):
        """Tokenize all papers."""
        logger.info("Tokenizing papers...")
        
        for paper in tqdm(self.papers, desc="Tokenizing"):
            tokens = self.tokenizer(
                paper['text'],
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            paper['input_ids'] = tokens['input_ids'].squeeze()
            paper['attention_mask'] = tokens['attention_mask'].squeeze()
            
    def save_dataset(self, output_path: Path):
        """Save processed dataset."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame for easier handling
        df_data = []
        for paper in self.papers:
            row = {
                'paper_id': paper['paper_id'],
                'text': paper['text'][:10000],  # Save truncated version
                'full_text_length': len(paper['text']),
                'title': paper['metadata'].title,
                'abstract': paper['metadata'].abstract,
                'year': paper['metadata'].year,
                'pages': paper['metadata'].pages,
                'word_count': paper['metadata'].word_count,
            }
            
            # Add labels
            for key, value in paper['labels'].items():
                row[f'label_{key}'] = value
                
            df_data.append(row)
            
        df = pd.DataFrame(df_data)
        
        # Save as parquet for efficiency
        df.to_parquet(output_path)
        
        # Save full tokenized data separately
        import torch
        torch.save(self.papers, output_path.with_suffix('.pt'))
        
        logger.info(f"Dataset saved to {output_path}")
        
    @staticmethod
    def load_dataset(dataset_path: Path):
        """Load preprocessed dataset."""
        import torch
        papers = torch.load(dataset_path.with_suffix('.pt'))
        df = pd.read_parquet(dataset_path)
        return papers, df