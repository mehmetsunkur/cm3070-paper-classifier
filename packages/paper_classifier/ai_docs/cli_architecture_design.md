# CLI Inference Pipeline Architecture

**Generated**: 2025-09-22  
**Context**: Detailed architecture design for `classify-paper` CLI command

## System Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   PDF File      │───▶│  Text Extractor  │───▶│  Inference Pipeline │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                            │
                       ┌──────────────────┐                ▼
                       │  Model Manager   │    ┌─────────────────────┐
                       │  Model Selector  │◀───│  Classification     │
                       └──────────────────┘    │  Results            │
                                               └─────────────────────┘
```

## Component Architecture

### 1. CLI Interface Layer (`cli_inference.py`)
**Responsibility**: Command-line argument parsing and orchestration

```python
# paper_classifier/src/paper_classifier/cli_inference.py
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

from .pdf_extractor import PDFTextExtractor, PDFExtractionError
from .inference_pipeline import InferencePipeline, InferenceError
from .model_manager import ModelManager

def main() -> int:
    """Main CLI entry point for paper classification."""
    try:
        args = parse_arguments()
        
        # Validate inputs
        validate_cli_inputs(args)
        
        # Initialize components
        pipeline = InferencePipeline(
            models_dir=args.models_dir,
            verbose=args.verbose
        )
        
        # Run classification
        results = pipeline.classify_pdf(
            pdf_path=args.pdf_file,
            label_types=args.labels,
            size_preference=args.size_preference
        )
        
        # Output results
        output_results(results, args.output_format)
        return 0
        
    except (FileNotFoundError, PDFExtractionError, InferenceError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\\nOperation cancelled by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Classify academic papers using trained Mamba models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Classify for all label types with default settings
  classify-paper paper.pdf
  
  # Classify for specific label types only
  classify-paper paper.pdf --labels discipline field
  
  # Use larger model context and JSON output
  classify-paper paper.pdf --size-preference 8_16k --output-format json
  
  # Verbose output with custom models directory
  classify-paper paper.pdf --models-dir /path/to/models --verbose
        '''
    )
    
    # Required arguments
    parser.add_argument(
        'pdf_file',
        help='Path to PDF file to classify'
    )
    
    # Classification options
    parser.add_argument(
        '--labels', 
        nargs='+',
        choices=['discipline', 'field', 'method'],
        default=['discipline', 'field', 'method'],
        help='Label types to predict (default: all)'
    )
    
    parser.add_argument(
        '--size-preference',
        choices=['1_4k', '4_8k', '8_16k', '16_32k', '32_64k'],
        default='4_8k',
        help='Preferred model size/context length (default: 4_8k)'
    )
    
    # Output options
    parser.add_argument(
        '--output-format',
        choices=['text', 'json', 'csv'],
        default='text',
        help='Output format (default: text)'
    )
    
    # System options
    parser.add_argument(
        '--models-dir',
        default='trained_models',
        help='Directory containing trained models (default: trained_models)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Advanced options
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.0,
        help='Minimum confidence threshold for predictions (default: 0.0)'
    )
    
    parser.add_argument(
        '--max-text-length',
        type=int,
        default=30000,
        help='Maximum text length in tokens (default: 30000)'
    )
    
    return parser.parse_args()

def validate_cli_inputs(args: argparse.Namespace) -> None:
    """Validate CLI inputs and raise errors if invalid."""
    # Check PDF file exists
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {args.pdf_file}")
    
    if not pdf_path.suffix.lower() == '.pdf':
        raise ValueError(f"File must be a PDF: {args.pdf_file}")
    
    # Check models directory exists
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {args.models_dir}")
    
    # Validate confidence threshold
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0.0 and 1.0")

def output_results(results: Dict[str, Any], format_type: str) -> None:
    """Output classification results in specified format."""
    if format_type == 'json':
        print(json.dumps(results, indent=2))
    elif format_type == 'csv':
        output_csv_results(results)
    else:  # text format
        output_text_results(results)

def output_text_results(results: Dict[str, Any]) -> None:
    """Output results in human-readable text format."""
    print("Paper Classification Results")
    print("=" * 50)
    
    for label_type, result in results['predictions'].items():
        print(f"\\n{label_type.upper()}:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Model: {result['model_name']}")
    
    print(f"\\nProcessing Time: {results['processing_time']:.2f}s")
    print(f"Text Length: {results['text_stats']['character_count']} characters")

def output_csv_results(results: Dict[str, Any]) -> None:
    """Output results in CSV format."""
    import csv
    import sys
    
    writer = csv.writer(sys.stdout)
    writer.writerow(['label_type', 'prediction', 'confidence', 'model_name'])
    
    for label_type, result in results['predictions'].items():
        writer.writerow([
            label_type,
            result['prediction'],
            result['confidence'],
            result['model_name']
        ])

if __name__ == '__main__':
    sys.exit(main())
```

### 2. Inference Pipeline (`inference_pipeline.py`)
**Responsibility**: Orchestrate multi-label classification workflow

```python
# paper_classifier/src/paper_classifier/inference_pipeline.py
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch

from .pdf_extractor import PDFTextExtractor, PDFExtractionError
from .model_selector import ModelSelector, ModelSelectionError
from .model_manager import ModelManager
from .mamba.model import MambaTextClassification

logger = logging.getLogger(__name__)

class InferencePipeline:
    """Orchestrates PDF classification using multiple models."""
    
    def __init__(self, models_dir: str = "trained_models", verbose: bool = False):
        self.models_dir = Path(models_dir)
        self.verbose = verbose
        
        # Initialize components
        self.model_manager = ModelManager(models_dir=str(self.models_dir), verbose=verbose)
        self.model_selector = ModelSelector(self.model_manager)
        self.pdf_extractor = PDFTextExtractor()
        
        # Performance tracking
        self._timing_data = {}
        
    def classify_pdf(self, 
                    pdf_path: str,
                    label_types: List[str],
                    size_preference: str = '4_8k',
                    confidence_threshold: float = 0.0,
                    max_text_length: int = 30000) -> Dict[str, Any]:
        """
        Classify PDF for multiple label types.
        
        Args:
            pdf_path: Path to PDF file
            label_types: List of label types to predict ['discipline', 'field', 'method']
            size_preference: Preferred model size
            confidence_threshold: Minimum confidence for predictions
            max_text_length: Maximum text length in tokens
            
        Returns:
            Dictionary containing predictions and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Extract text from PDF
            if self.verbose:
                print(f"Extracting text from {pdf_path}...")
            
            extraction_start = time.time()
            text = self.pdf_extractor.extract_text(pdf_path)
            extraction_time = time.time() - extraction_start
            
            if self.verbose:
                print(f"Extracted {len(text)} characters in {extraction_time:.2f}s")
            
            # Step 2: Preprocess text
            processed_text = self._preprocess_text(text, max_text_length)
            
            # Step 3: Run inference for each label type
            predictions = {}
            
            for label_type in label_types:
                if self.verbose:
                    print(f"Classifying for {label_type}...")
                    
                label_start = time.time()
                prediction = self._classify_for_label_type(
                    processed_text, 
                    label_type, 
                    size_preference,
                    confidence_threshold
                )
                label_time = time.time() - label_start
                
                predictions[label_type] = prediction
                predictions[label_type]['processing_time'] = label_time
                
                if self.verbose:
                    print(f"  {label_type}: {prediction['prediction']} "
                         f"(confidence: {prediction['confidence']:.3f}) "
                         f"in {label_time:.2f}s")
            
            # Step 4: Compile results
            total_time = time.time() - start_time
            
            results = {
                'pdf_path': pdf_path,
                'predictions': predictions,
                'text_stats': {
                    'character_count': len(text),
                    'processed_character_count': len(processed_text),
                    'extraction_time': extraction_time
                },
                'processing_time': total_time,
                'parameters': {
                    'label_types': label_types,
                    'size_preference': size_preference,
                    'confidence_threshold': confidence_threshold,
                    'max_text_length': max_text_length
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Inference pipeline failed: {e}")
            raise InferenceError(f"Classification failed: {e}") from e
    
    def _classify_for_label_type(self,
                                text: str,
                                label_type: str,
                                size_preference: str,
                                confidence_threshold: float) -> Dict[str, Any]:
        """Classify text for a specific label type."""
        
        # Select best model
        model_info = self.model_selector.select_best_model(label_type, size_preference)
        
        # Load model and tokenizer
        model, tokenizer, id2label = self._load_model_components(model_info)
        
        try:
            # Run inference
            prediction_id = model.predict(text, tokenizer)
            prediction_label = id2label.get(int(prediction_id), f"Unknown_{prediction_id}")
            
            # Calculate confidence (requires additional forward pass)
            confidence = self._calculate_confidence(model, text, tokenizer)
            
            # Check confidence threshold
            if confidence < confidence_threshold:
                logger.warning(f"Prediction confidence {confidence:.3f} below threshold {confidence_threshold}")
            
            return {
                'prediction': prediction_label,
                'prediction_id': int(prediction_id),
                'confidence': confidence,
                'model_name': model_info.name,
                'model_path': str(model_info.final_model_path)
            }
            
        finally:
            # Memory cleanup
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _load_model_components(self, model_info) -> Tuple[Any, Any, Dict[int, str]]:
        """Load model, tokenizer, and label mappings."""
        import json
        from transformers import AutoTokenizer
        
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MambaTextClassification.from_checkpoint(
            str(model_info.final_model_path),
            device=device
        )
        model.eval()
        
        # Load tokenizer - try multiple locations
        tokenizer = None
        tokenizer_paths = [
            model_info.final_model_path / "tokenizer",
            model_info.final_model_path.parent / "tokenizer",
            "state-spaces/mamba-370m"  # fallback to base model
        ]
        
        for path in tokenizer_paths:
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(path))
                break
            except Exception:
                continue
        
        if tokenizer is None:
            raise InferenceError("Could not load tokenizer for model")
        
        # Load label mappings
        label_mappings_path = model_info.final_model_path / "label_mappings.json"
        with open(label_mappings_path) as f:
            label_mappings = json.load(f)
        
        # Extract id2label mapping for this label type
        label_key = f"label_{model_info.label_type}"
        if label_key not in label_mappings:
            raise InferenceError(f"Label mapping not found for {label_key}")
        
        id2label_raw = label_mappings[label_key]["id2label"]
        id2label = {int(k): v for k, v in id2label_raw.items()}
        
        return model, tokenizer, id2label
    
    def _preprocess_text(self, text: str, max_length: int) -> str:
        """Preprocess extracted text for model input."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate to max length (approximate tokenization)
        words = text.split()
        if len(words) > max_length:
            text = ' '.join(words[:max_length])
            logger.warning(f"Truncated text to {max_length} words")
        
        return text
    
    def _calculate_confidence(self, model, text: str, tokenizer) -> float:
        """Calculate prediction confidence using softmax probabilities."""
        import torch.nn.functional as F
        
        device = next(model.parameters()).device
        input_ids = torch.tensor(tokenizer(text)['input_ids'], device=device)[None]
        
        with torch.no_grad():
            logits = model.forward(input_ids).logits[0]
            probabilities = F.softmax(logits, dim=-1)
            confidence = float(torch.max(probabilities).cpu())
        
        return confidence

class InferenceError(Exception):
    """Raised when inference pipeline encounters an error."""
    pass
```

### 3. Model Selection (`model_selector.py`)
**Responsibility**: Select optimal models based on criteria

```python
# paper_classifier/src/paper_classifier/model_selector.py
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .model_manager import ModelManager, ModelInfo

logger = logging.getLogger(__name__)

@dataclass
class ModelSelectionCriteria:
    """Criteria for model selection."""
    label_type: str
    size_preference: str
    metric_preference: str = 'eval_loss'  # 'eval_loss' or 'eval_accuracy'
    require_metrics: bool = False

class ModelSelector:
    """Selects optimal models based on specified criteria."""
    
    # Size preference order (smaller to larger context)
    SIZE_ORDER = ['1_4k', '4_8k', '8_16k', '16_32k', '32_64k']
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def select_best_model(self, 
                         label_type: str, 
                         size_preference: str = '4_8k',
                         require_metrics: bool = True) -> ModelInfo:
        """
        Select the best model for given criteria.
        
        Args:
            label_type: Target label type ('discipline', 'field', 'method')
            size_preference: Preferred model size ('1_4k', '4_8k', etc.)
            require_metrics: Whether to require models with evaluation metrics
            
        Returns:
            Best matching ModelInfo
            
        Raises:
            ModelSelectionError: No suitable models found
        """
        logger.info(f"Selecting model for {label_type} with size preference {size_preference}")
        
        # Get all models for label type
        candidates = self.model_manager.search_models(label_type=label_type)
        
        if not candidates:
            raise ModelSelectionError(f"No models found for label type: {label_type}")
        
        # Filter models with metrics if required
        if require_metrics:
            candidates_with_metrics = [m for m in candidates if m.metrics and 'eval_loss' in m.metrics]
            if candidates_with_metrics:
                candidates = candidates_with_metrics
            else:
                logger.warning(f"No models with metrics found for {label_type}, using all available")
        
        # Group by size and rank within each group
        size_groups = self._group_models_by_size(candidates)
        
        # Try sizes in preference order
        size_order = self._get_size_preference_order(size_preference)
        
        for size in size_order:
            if size in size_groups:
                best_in_group = self._select_best_in_group(size_groups[size])
                if best_in_group:
                    logger.info(f"Selected model: {best_in_group.name} (size: {size})")
                    # Add derived attributes
                    best_in_group.label_type = label_type
                    return best_in_group
        
        # Fallback: return any available model
        if candidates:
            fallback = candidates[0]
            logger.warning(f"Using fallback model: {fallback.name}")
            fallback.label_type = label_type
            return fallback
        
        raise ModelSelectionError(f"No suitable models found for {label_type}")
    
    def _group_models_by_size(self, models: List[ModelInfo]) -> Dict[str, List[ModelInfo]]:
        """Group models by size category."""
        groups = {}
        
        for model in models:
            size = self._extract_size_from_name(model.name)
            if size:
                if size not in groups:
                    groups[size] = []
                groups[size].append(model)
        
        return groups
    
    def _extract_size_from_name(self, model_name: str) -> Optional[str]:
        """Extract size category from model name."""
        for size in self.SIZE_ORDER:
            if size in model_name:
                return size
        return None
    
    def _get_size_preference_order(self, preferred_size: str) -> List[str]:
        """Get ordered list of sizes based on preference."""
        if preferred_size not in self.SIZE_ORDER:
            logger.warning(f"Unknown size preference: {preferred_size}, using 4_8k")
            preferred_size = '4_8k'
        
        # Start with preferred size, then expand outward
        preferred_idx = self.SIZE_ORDER.index(preferred_size)
        order = [preferred_size]
        
        # Add smaller and larger sizes alternately
        left_idx = preferred_idx - 1
        right_idx = preferred_idx + 1
        
        while left_idx >= 0 or right_idx < len(self.SIZE_ORDER):
            if left_idx >= 0:
                order.append(self.SIZE_ORDER[left_idx])
                left_idx -= 1
            if right_idx < len(self.SIZE_ORDER):
                order.append(self.SIZE_ORDER[right_idx])
                right_idx += 1
        
        return order
    
    def _select_best_in_group(self, models: List[ModelInfo]) -> Optional[ModelInfo]:
        """Select best model within a size group."""
        if not models:
            return None
        
        # Prefer models with metrics
        models_with_metrics = [m for m in models if m.metrics and 'eval_loss' in m.metrics]
        
        if models_with_metrics:
            # Sort by eval_loss (ascending = better)
            return min(models_with_metrics, key=lambda m: m.metrics['eval_loss'])
        else:
            # No metrics available, return first one
            return models[0]
    
    def get_available_sizes_for_label(self, label_type: str) -> List[str]:
        """Get list of available sizes for a label type."""
        models = self.model_manager.search_models(label_type=label_type)
        sizes = set()
        
        for model in models:
            size = self._extract_size_from_name(model.name)
            if size:
                sizes.add(size)
        
        # Return in standard order
        return [size for size in self.SIZE_ORDER if size in sizes]
    
    def get_model_selection_report(self, label_types: List[str], size_preference: str) -> Dict[str, Any]:
        """Generate a report of model selection for given criteria."""
        report = {
            'size_preference': size_preference,
            'label_types': label_types,
            'selections': {},
            'availability': {}
        }
        
        for label_type in label_types:
            try:
                selected = self.select_best_model(label_type, size_preference, require_metrics=False)
                report['selections'][label_type] = {
                    'model_name': selected.name,
                    'actual_size': self._extract_size_from_name(selected.name),
                    'has_metrics': selected.metrics is not None,
                    'eval_loss': selected.metrics.get('eval_loss') if selected.metrics else None
                }
            except ModelSelectionError as e:
                report['selections'][label_type] = {'error': str(e)}
            
            # Report availability
            report['availability'][label_type] = self.get_available_sizes_for_label(label_type)
        
        return report

class ModelSelectionError(Exception):
    """Raised when model selection fails."""
    pass
```

### 4. Entry Point Registration
**Location**: Already configured in `pyproject.toml`

```toml
[project.scripts]
train-paper-classifier = "paper_classifier.main:main"
list-paper-models = "paper_classifier.model_manager:main"
classify-paper = "paper_classifier.cli_inference:main"  # ← New entry point
```

## Data Flow Architecture

```
1. CLI Input Processing
   ├── Parse arguments (pdf_file, --labels, --size-preference, etc.)
   ├── Validate inputs (file exists, valid options)
   └── Initialize pipeline components

2. PDF Text Extraction  
   ├── Detect PDF type (text-based vs. image-based)
   ├── Extract text using strategy (pdfplumber → pymupdf → OCR)
   ├── Validate extraction quality
   └── Preprocess text (cleanup, truncation)

3. Model Selection (for each label type)
   ├── Query ModelManager for available models
   ├── Filter by label_type (discipline/field/method)
   ├── Apply size preference with fallback logic
   ├── Rank by evaluation metrics (eval_loss ascending)
   └── Select best candidate

4. Inference Execution (for each label type)
   ├── Load model checkpoint → GPU/CPU
   ├── Load tokenizer (multiple fallback paths)
   ├── Load label mappings (id2label)
   ├── Tokenize input text
   ├── Forward pass → logits
   ├── Convert to prediction + confidence
   └── Cleanup memory (del model, torch.cuda.empty_cache())

5. Result Compilation & Output
   ├── Aggregate predictions from all label types
   ├── Add metadata (timing, model info, text stats)
   ├── Format output (text/json/csv)
   └── Return exit code
```

## Error Handling Strategy

```python
class PaperClassifierError(Exception):
    """Base exception for paper classifier errors."""
    pass

class PDFProcessingError(PaperClassifierError):
    """PDF-related errors."""
    pass

class ModelLoadingError(PaperClassifierError):
    """Model loading errors."""
    pass

class InferenceError(PaperClassifierError):
    """Inference execution errors."""
    pass

class ModelSelectionError(PaperClassifierError):
    """Model selection errors."""
    pass

# Error handling in main()
try:
    results = pipeline.classify_pdf(...)
except PDFProcessingError as e:
    print(f"PDF Error: {e}")
    return 1
except ModelSelectionError as e:
    print(f"Model Selection Error: {e}")
    print("Use 'list-paper-models --summary' to see available models")
    return 1
except ModelLoadingError as e:
    print(f"Model Loading Error: {e}")
    return 1
except InferenceError as e:
    print(f"Inference Error: {e}")
    return 1
```

## Performance Optimization

### Memory Management
```python
# Sequential model loading to avoid memory issues
def classify_with_memory_management(self, text, label_types, size_preference):
    results = {}
    
    for label_type in label_types:
        # Load only one model at a time
        model_info = self.model_selector.select_best_model(label_type, size_preference)
        
        # Memory-efficient loading
        with self._load_model_context(model_info) as (model, tokenizer, id2label):
            prediction = model.predict(text, tokenizer)
            results[label_type] = {
                'prediction': id2label.get(prediction),
                'model': model_info.name
            }
        
        # Explicit cleanup happens in context manager
    
    return results

@contextmanager
def _load_model_context(self, model_info):
    """Context manager for model loading with guaranteed cleanup."""
    model, tokenizer, id2label = self._load_model_components(model_info)
    try:
        yield model, tokenizer, id2label
    finally:
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### Caching Strategy
```python
class CachedInferencePipeline(InferencePipeline):
    """Pipeline with model and text caching."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_cache = {}
        self._text_cache = {}
    
    def _get_cached_text(self, pdf_path: str) -> Optional[str]:
        """Get cached text extraction if available."""
        pdf_stat = Path(pdf_path).stat()
        cache_key = f"{pdf_path}_{pdf_stat.st_size}_{pdf_stat.st_mtime}"
        return self._text_cache.get(cache_key)
    
    def _cache_text(self, pdf_path: str, text: str) -> None:
        """Cache extracted text."""
        pdf_stat = Path(pdf_path).stat()
        cache_key = f"{pdf_path}_{pdf_stat.st_size}_{pdf_stat.st_mtime}"
        self._text_cache[cache_key] = text
```

## Usage Examples

### Basic Usage
```bash
# Classify for all label types
classify-paper research_paper.pdf

# Output:
# Paper Classification Results
# ==================================================
# 
# DISCIPLINE:
#   Prediction: Computer Science
#   Confidence: 0.987
#   Model: model_label_discipline_4k_s1000
# 
# FIELD:
#   Prediction: Machine Learning
#   Confidence: 0.932
#   Model: model_label_field_4_8k_s1000
# 
# METHOD:
#   Prediction: Deep Learning
#   Confidence: 0.876
#   Model: model_label_method_4_8k_s1000
```

### Advanced Usage
```bash
# Specific labels with JSON output
classify-paper paper.pdf --labels discipline field --output-format json

# Use larger context model
classify-paper paper.pdf --size-preference 8_16k --verbose

# Custom models directory
classify-paper paper.pdf --models-dir /path/to/models --confidence-threshold 0.8
```

### JSON Output Format
```json
{
  "pdf_path": "paper.pdf",
  "predictions": {
    "discipline": {
      "prediction": "Computer Science",
      "prediction_id": 0,
      "confidence": 0.987,
      "model_name": "model_label_discipline_4k_s1000",
      "model_path": "trained_models/model_label_discipline_4k_s1000/run_20250914_100827/final_model",
      "processing_time": 1.23
    },
    "field": {
      "prediction": "Machine Learning", 
      "prediction_id": 2,
      "confidence": 0.932,
      "model_name": "model_label_field_4_8k_s1000",
      "model_path": "trained_models/model_label_field_4_8k_s1000/run_20250915_122801/final_model",
      "processing_time": 1.45
    }
  },
  "text_stats": {
    "character_count": 45678,
    "processed_character_count": 30000,
    "extraction_time": 0.89
  },
  "processing_time": 4.67,
  "parameters": {
    "label_types": ["discipline", "field"],
    "size_preference": "4_8k",
    "confidence_threshold": 0.0,
    "max_text_length": 30000
  }
}
```

## Testing Strategy

```python
# tests/test_cli_inference.py
def test_cli_basic_usage(tmp_path, sample_pdf):
    """Test basic CLI functionality."""
    result = subprocess.run([
        'python', '-m', 'paper_classifier.cli_inference',
        str(sample_pdf),
        '--models-dir', 'tests/fixtures/models'
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert 'DISCIPLINE:' in result.stdout
    assert 'FIELD:' in result.stdout

def test_cli_json_output(sample_pdf):
    """Test JSON output format."""
    result = subprocess.run([
        'python', '-m', 'paper_classifier.cli_inference',
        str(sample_pdf),
        '--output-format', 'json'
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert 'predictions' in output
    assert 'processing_time' in output

def test_cli_error_handling():
    """Test error handling for missing files."""
    result = subprocess.run([
        'python', '-m', 'paper_classifier.cli_inference',
        'nonexistent.pdf'
    ], capture_output=True, text=True)
    
    assert result.returncode == 1
    assert 'Error:' in result.stderr
```

## Installation & Deployment

```bash
# Development installation
cd packages/paper_classifier
pip install -e .[pdf]

# Verify installation
classify-paper --help

# Test with sample PDF
classify-paper tests/fixtures/sample_paper.pdf --verbose
```

This architecture provides a robust, extensible CLI system for PDF paper classification with comprehensive error handling, performance optimization, and flexible output formats.