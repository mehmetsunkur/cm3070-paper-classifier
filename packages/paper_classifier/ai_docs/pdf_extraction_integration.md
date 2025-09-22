# PDF Text Extraction Integration

**Generated**: 2025-09-22  
**Context**: Technical specification for integrating PDF text extraction into paper_classifier package

## Current State Analysis

### Available Dependencies in Ecosystem
The `paper_dataset` package already has comprehensive PDF processing dependencies:

```toml
# From paper_dataset/pyproject.toml
dependencies = [
    # PDF processing
    "pdfplumber>=0.10.0",      # Text-based PDF extraction
    "pymupdf>=1.23.0",         # Alternative PDF library  
    "pdf2image>=1.16.0",       # Convert PDF to images
    "pytesseract>=0.3.13",     # OCR for image-based PDFs
]
```

### Gap in paper_classifier
The `paper_classifier` package focuses on ML model training/inference and lacks PDF processing:

```toml
# Current paper_classifier/pyproject.toml - no PDF libraries
dependencies = [
    "torch", "transformers", "mamba-ssm", "pandas", ...
    # Missing: PDF processing capabilities
]
```

## Integration Strategies

### Strategy 1: Cross-Package Import (Recommended)
**Approach**: Import PDF functionality from `paper_dataset` package  
**Pros**: No duplicate dependencies, reuse existing code  
**Cons**: Adds inter-package dependency  

```python
# paper_classifier/src/paper_classifier/pdf_extractor.py
try:
    from paper_dataset.pdf_processor import extract_text_from_pdf
    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    PDF_EXTRACTION_AVAILABLE = False
    
class PDFTextExtractor:
    def __init__(self):
        if not PDF_EXTRACTION_AVAILABLE:
            raise ImportError("paper_dataset package required for PDF processing")
    
    def extract_text(self, pdf_path: str) -> str:
        return extract_text_from_pdf(pdf_path)
```

### Strategy 2: Lightweight Direct Integration  
**Approach**: Add minimal PDF dependencies to `paper_classifier`  
**Pros**: Self-contained, no cross-package dependencies  
**Cons**: Duplicate dependencies, increased package size  

```python
# Add to paper_classifier/pyproject.toml
dependencies = [
    # Existing dependencies...
    "pdfplumber>=0.10.0",  # Add minimal PDF support
]
```

### Strategy 3: Optional PDF Dependencies
**Approach**: Make PDF processing optional with graceful degradation  
**Pros**: Flexible installation, clear separation of concerns  
**Cons**: More complex error handling  

```python
# paper_classifier/pyproject.toml
[project.optional-dependencies]
pdf = [
    "pdfplumber>=0.10.0",
    "pymupdf>=1.23.0", 
    "pytesseract>=0.3.13",
]

# Installation: pip install -e .[pdf]
```

## Recommended Implementation

### Multi-Strategy PDF Text Extraction

```python
# paper_classifier/src/paper_classifier/pdf_extractor.py
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class PDFTextExtractor:
    """Multi-strategy PDF text extraction with fallback methods."""
    
    def __init__(self):
        self.strategies = self._initialize_strategies()
        
    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize available extraction strategies."""
        strategies = {}
        
        # Strategy 1: pdfplumber (best for text-based PDFs)
        try:
            import pdfplumber
            strategies['pdfplumber'] = pdfplumber
            logger.info("pdfplumber strategy available")
        except ImportError:
            logger.warning("pdfplumber not available")
            
        # Strategy 2: PyMuPDF (good fallback)
        try:
            import fitz  # PyMuPDF
            strategies['pymupdf'] = fitz
            logger.info("PyMuPDF strategy available") 
        except ImportError:
            logger.warning("PyMuPDF not available")
            
        # Strategy 3: OCR with pytesseract (for image-based PDFs)
        try:
            import pytesseract
            from pdf2image import convert_from_path
            strategies['ocr'] = {'pytesseract': pytesseract, 'pdf2image': convert_from_path}
            logger.info("OCR strategy available")
        except ImportError:
            logger.warning("OCR strategy not available")
            
        if not strategies:
            raise ImportError("No PDF processing libraries available. Install with: pip install pdfplumber")
            
        return strategies
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF using multiple strategies with fallback.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            FileNotFoundError: PDF file doesn't exist
            PDFExtractionError: All extraction strategies failed
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        # Try strategies in order of preference
        for strategy_name in ['pdfplumber', 'pymupdf', 'ocr']:
            if strategy_name in self.strategies:
                try:
                    text = self._extract_with_strategy(pdf_path, strategy_name)
                    if text and len(text.strip()) > 0:
                        logger.info(f"Successfully extracted text using {strategy_name}")
                        return text
                except Exception as e:
                    logger.warning(f"{strategy_name} extraction failed: {e}")
                    continue
                    
        raise PDFExtractionError(f"All extraction strategies failed for {pdf_path}")
    
    def _extract_with_strategy(self, pdf_path: Path, strategy: str) -> str:
        """Extract text using specified strategy."""
        
        if strategy == 'pdfplumber':
            return self._extract_with_pdfplumber(pdf_path)
        elif strategy == 'pymupdf':
            return self._extract_with_pymupdf(pdf_path)  
        elif strategy == 'ocr':
            return self._extract_with_ocr(pdf_path)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber (best for text-based PDFs)."""
        pdfplumber = self.strategies['pdfplumber']
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                    
        return '\n\n'.join(text_parts)
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> str:
        """Extract text using PyMuPDF as fallback."""
        fitz = self.strategies['pymupdf']
        text_parts = []
        
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text_parts.append(page.get_text())
        doc.close()
        
        return '\n\n'.join(text_parts)
    
    def _extract_with_ocr(self, pdf_path: Path) -> str:
        """Extract text using OCR (for image-based or scanned PDFs)."""
        ocr_tools = self.strategies['ocr']
        pytesseract = ocr_tools['pytesseract']
        convert_from_path = ocr_tools['pdf2image']
        
        # Convert PDF pages to images
        images = convert_from_path(pdf_path)
        text_parts = []
        
        for i, image in enumerate(images):
            try:
                page_text = pytesseract.image_to_string(image)
                if page_text.strip():
                    text_parts.append(page_text)
                logger.debug(f"OCR processed page {i+1}/{len(images)}")
            except Exception as e:
                logger.warning(f"OCR failed for page {i+1}: {e}")
        
        return '\n\n'.join(text_parts)

class PDFExtractionError(Exception):
    """Raised when PDF text extraction fails."""
    pass
```

## Integration with CLI

```python
# paper_classifier/src/paper_classifier/cli_inference.py
from .pdf_extractor import PDFTextExtractor, PDFExtractionError

def main():
    args = parse_args()
    
    # Initialize PDF extractor
    try:
        pdf_extractor = PDFTextExtractor()
    except ImportError as e:
        print(f"Error: PDF processing not available. {e}")
        print("Install with: pip install paper_classifier[pdf]")
        return 1
    
    # Extract text from PDF
    try:
        text = pdf_extractor.extract_text(args.pdf_file)
        print(f"Extracted {len(text)} characters from PDF")
    except (FileNotFoundError, PDFExtractionError) as e:
        print(f"Error extracting PDF: {e}")
        return 1
    
    # Continue with classification...
```

## Quality Assurance

### Text Preprocessing
After extraction, text may need cleaning:

```python
def preprocess_extracted_text(text: str) -> str:
    """Clean and preprocess extracted PDF text."""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove page headers/footers (common patterns)
    # Remove URLs, email addresses if needed
    # Handle special characters
    
    # Limit text length for model context
    MAX_TOKENS = 32000  # Adjust based on model size
    if len(text.split()) > MAX_TOKENS:
        text = ' '.join(text.split()[:MAX_TOKENS])
        
    return text
```

### Extraction Validation
```python
def validate_extracted_text(text: str, pdf_path: str) -> bool:
    """Validate that extracted text is reasonable."""
    if not text or len(text.strip()) < 100:
        logger.warning(f"Extracted text too short from {pdf_path}")
        return False
        
    # Check for common OCR errors or garbage text
    garbage_ratio = sum(1 for c in text if not c.isalnum() and c not in ' .,!?') / len(text)
    if garbage_ratio > 0.3:
        logger.warning(f"High garbage character ratio in {pdf_path}")
        return False
        
    return True
```

## Installation Options

### Option 1: Include in main dependencies
```toml
# paper_classifier/pyproject.toml
dependencies = [
    # ... existing deps ...
    "pdfplumber>=0.10.0",
]
```

### Option 2: Optional dependency (Recommended)
```toml
[project.optional-dependencies]
pdf = [
    "pdfplumber>=0.10.0",
    "pymupdf>=1.23.0",
    "pdf2image>=1.16.0", 
    "pytesseract>=0.3.13",
]
```

```bash
# Installation
pip install -e .[pdf]  # With PDF support
pip install -e .       # Without PDF support
```

### Option 3: Cross-package dependency
```toml
# paper_classifier/pyproject.toml  
dependencies = [
    # ... existing deps ...
    "paper-dataset",  # Import PDF functionality
]
```

## Error Handling Strategy

```python
class PDFProcessingError(Exception):
    """Base exception for PDF processing errors."""
    pass

class PDFNotFoundError(PDFProcessingError):
    """PDF file not found."""
    pass

class PDFCorruptedError(PDFProcessingError):
    """PDF file corrupted or unreadable."""
    pass

class PDFExtractionError(PDFProcessingError):
    """Text extraction failed with all strategies."""
    pass

class PDFTooLargeError(PDFProcessingError):
    """PDF file too large to process."""
    pass

# Usage in CLI
try:
    text = pdf_extractor.extract_text(pdf_path)
except PDFNotFoundError:
    print(f"Error: PDF file not found: {pdf_path}")
    return 1
except PDFCorruptedError:
    print(f"Error: PDF file appears corrupted: {pdf_path}")
    return 1
except PDFExtractionError:
    print(f"Error: Could not extract text from PDF: {pdf_path}")
    print("This may be a scanned image PDF. Install OCR support with: pip install paper_classifier[pdf]")
    return 1
```

## Performance Considerations

### Memory Management
```python
def extract_text_memory_efficient(self, pdf_path: Path) -> str:
    """Extract text with memory constraints."""
    # Process large PDFs page by page instead of loading all at once
    # Implement streaming for very large files
    # Add progress reporting for long operations
    pass
```

### Caching
```python
import hashlib
from pathlib import Path

def get_pdf_cache_key(pdf_path: Path) -> str:
    """Generate cache key for PDF file."""
    stat = pdf_path.stat()
    content = f"{pdf_path.name}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(content.encode()).hexdigest()

def cache_extracted_text(pdf_path: Path, text: str) -> None:
    """Cache extracted text for future use."""
    cache_dir = Path.home() / '.paper_classifier_cache'
    cache_dir.mkdir(exist_ok=True)
    
    cache_key = get_pdf_cache_key(pdf_path)
    cache_file = cache_dir / f"{cache_key}.txt"
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(text)
```

## Testing Strategy

```python
# tests/test_pdf_extractor.py
import pytest
from pathlib import Path
from paper_classifier.pdf_extractor import PDFTextExtractor, PDFExtractionError

class TestPDFTextExtractor:
    def test_extract_text_based_pdf(self):
        """Test extraction from text-based PDF."""
        extractor = PDFTextExtractor()
        text = extractor.extract_text("tests/fixtures/sample_text.pdf")
        assert len(text) > 100
        assert "abstract" in text.lower() or "introduction" in text.lower()
    
    def test_extract_scanned_pdf(self):
        """Test extraction from scanned PDF (OCR)."""
        # Requires OCR dependencies
        pass
    
    def test_file_not_found(self):
        """Test handling of missing PDF files."""
        extractor = PDFTextExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract_text("nonexistent.pdf")
    
    def test_corrupted_pdf(self):
        """Test handling of corrupted PDF files."""
        # Create corrupted PDF fixture
        pass
```

## Summary

**Recommended Approach**: Optional PDF dependencies with multi-strategy extraction

**Benefits**:
- ✅ Flexible installation (with/without PDF support)
- ✅ Robust extraction with multiple fallback methods  
- ✅ Clear error handling and user guidance
- ✅ Memory efficient processing
- ✅ Caching for performance
- ✅ Comprehensive testing

**Installation**: `pip install -e .[pdf]`  
**Usage**: Transparent integration with CLI inference pipeline