# CLI PDF Inference Implementation - Blockers & Implementation Details

**Generated**: 2025-09-22  
**Context**: Analysis for implementing `classify-paper` CLI command to classify PDF files using trained Mamba models

## Executive Summary

To implement PDF classification via CLI requires 8 major components:
1. PDF text extraction pipeline
2. CLI interface creation  
3. Model selection & ranking logic
4. Multi-label inference orchestration
5. Tokenizer integration
6. Memory management
7. Label mapping system
8. Error handling framework

**Estimated Implementation Time**: 2-3 days basic, 1 week production-ready

## Current System State

### Available Models (39 total, 8 with metrics)
```
BY LABEL TYPE:
- discipline: 18 models (best: model_label_discipline_4k_s1000, eval_loss: 0.0042)
- field: 13 models (best: model_label_field_1000_1_4k, eval_loss: 0.0040) 
- method: 8 models (best: model_label_method_1_4k_s10000, eval_loss: 0.0054)

BY SIZE RANGE:
- 1_4k: 10 models (smallest context)
- 4_8k: 9 models  
- 8_16k: 8 models
- 16_32k: 8 models
- 32_64k: 2 models (largest context)
```

### Existing Infrastructure
- ✅ **ModelManager**: Comprehensive model discovery and ranking
- ✅ **MambaTextClassification**: Model class with `predict()` method  
- ✅ **CLI Entry Point**: `classify-paper` command added to pyproject.toml
- ✅ **PDF Dependencies**: Available in `paper_dataset` package (pdfplumber, pymupdf, pytesseract)
- ❌ **Inference CLI Module**: Missing `cli_inference.py`
- ❌ **PDF Integration**: No cross-package text extraction
- ❌ **Model Selection Logic**: No automated size-based selection

## Implementation Blockers & Solutions

### BLOCKER 1: PDF Text Extraction Pipeline
**Status**: 🔴 Critical blocker  
**Issue**: No PDF→text extraction in `paper_classifier` package

**Dependencies Available**:
```toml
# In paper_dataset/pyproject.toml
"pdfplumber>=0.10.0",
"pymupdf>=1.23.0", 
"pdf2image>=1.16.0",
"pytesseract>=0.3.13",
```

**Implementation Required**:
```python
# paper_classifier/src/paper_classifier/pdf_extractor.py
class PDFTextExtractor:
    def extract_text(self, pdf_path: str) -> str:
        # Multi-strategy extraction:
        # 1. pdfplumber (text-based PDFs)
        # 2. pymupdf fallback
        # 3. pytesseract OCR (image-based PDFs)
        pass
```

**Solution**: Create lightweight PDF extraction module or import from `paper_dataset`

---

### BLOCKER 2: CLI Interface Implementation
**Status**: 🟡 Entry point exists, module missing  
**Issue**: `pyproject.toml` references non-existent `cli_inference.py`

**Current Entry Point**:
```toml
[project.scripts]
classify-paper = "paper_classifier.cli_inference:main"
```

**Implementation Required**:
```python
# paper_classifier/src/paper_classifier/cli_inference.py
import argparse
from .pdf_extractor import PDFTextExtractor  
from .model_manager import ModelManager
from .inference_pipeline import InferencePipeline

def main():
    parser = argparse.ArgumentParser(description='Classify academic papers')
    parser.add_argument('pdf_file', help='Path to PDF file')
    parser.add_argument('--labels', nargs='+', 
                       choices=['discipline', 'field', 'method'], 
                       default=['discipline', 'field', 'method'])
    parser.add_argument('--size-preference', 
                       choices=['1_4k', '4_8k', '8_16k', '16_32k', '32_64k'],
                       default='4_8k')
    parser.add_argument('--output-format', choices=['json', 'text'], default='text')
    # Implementation here
```

---

### BLOCKER 3: Model Selection & Ranking Logic  
**Status**: 🟡 Manual selection works, automation missing  
**Issue**: No "best model by size preference" algorithm

**Current Capability**:
```bash
# Manual selection works
python -m paper_classifier.model_manager --search-label discipline --best 3
python -m paper_classifier.model_manager --get-model model_label_discipline_4k_s1000
```

**Implementation Required**:
```python
# paper_classifier/src/paper_classifier/model_selector.py
class ModelSelector:
    SIZE_PREFERENCE_ORDER = ['1_4k', '4_8k', '8_16k', '16_32k', '32_64k']
    
    def select_best_model(self, label_type: str, size_preference: str = '4_8k') -> ModelInfo:
        # 1. Get all models for label_type
        # 2. Filter by size_preference (with fallback)
        # 3. Rank by eval_loss (ascending)
        # 4. Return best model
        pass
        
    def get_fallback_sizes(self, preferred_size: str) -> List[str]:
        # Return sizes in order of preference from preferred_size
        pass
```

**Ranking Logic**:
1. Filter models by `label_type` (discipline/field/method)
2. Prefer requested `size_preference` (e.g., '4_8k')  
3. Fallback to adjacent sizes: 4_8k → 1_4k → 8_16k → 16_32k → 32_64k
4. Rank by `eval_loss` (ascending = better)
5. Select model with best metrics

---

### BLOCKER 4: Multi-Label Inference Orchestration
**Status**: 🟡 Single model inference exists  
**Issue**: No pipeline for running all 3 label types sequentially

**Current Single Model**:
```python
# From mamba/model.py - works for single prediction
model = MambaTextClassification.from_checkpoint(checkpoint_path)
prediction = model.predict(text, tokenizer, id2label)
```

**Implementation Required**:
```python
# paper_classifier/src/paper_classifier/inference_pipeline.py
class InferencePipeline:
    def __init__(self, models_dir: str = "trained_models"):
        self.model_manager = ModelManager(models_dir)
        self.model_selector = ModelSelector()
        
    def classify_text(self, text: str, label_types: List[str], 
                     size_preference: str = '4_8k') -> Dict[str, Any]:
        results = {}
        
        for label_type in label_types:
            # 1. Select best model for this label_type + size_preference
            model_info = self.model_selector.select_best_model(label_type, size_preference)
            
            # 2. Load model + tokenizer
            model, tokenizer = self._load_model_and_tokenizer(model_info)
            
            # 3. Run inference
            prediction = model.predict(text, tokenizer, model_info.id2label)
            
            # 4. Store results
            results[label_type] = {
                'prediction': prediction,
                'model_name': model_info.name,
                'confidence': ...,  # Could extract from logits
            }
            
            # 5. Unload model (memory management)
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        return results
```

---

### BLOCKER 5: Tokenizer Integration
**Status**: 🟡 Model has tokenization, tokenizer discovery missing  
**Issue**: Need to discover/load correct tokenizer for each model

**Current Model Integration**:
```python
# Model expects tokenizer in predict() method
def predict(self, text, tokenizer, id2label = None):
    input_ids = torch.tensor(tokenizer(text)['input_ids'], device=device)[None]
    # ...
```

**Missing**: Tokenizer discovery from model checkpoints

**Implementation Required**:
```python
def _load_model_and_tokenizer(self, model_info: ModelInfo):
    # 1. Load model from final_model path
    model = MambaTextClassification.from_checkpoint(model_info.final_model_path)
    
    # 2. Discover tokenizer - likely stored in training_config or separate file
    tokenizer_path = model_info.final_model_path / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        # Fallback to base tokenizer used in training
        tokenizer = AutoTokenizer.from_pretrained("base-model-name")
    
    return model, tokenizer
```

---

### BLOCKER 6: Memory Management  
**Status**: 🟢 Manageable with sequential loading  
**Issue**: Loading multiple large models may exceed GPU memory

**Model Sizes**: Range from 1_4k (small) to 32_64k (large) parameters

**Solution Strategy**:
```python
# Sequential load/unload approach
for label_type in label_types:
    model, tokenizer = load_model_for_label_type(label_type)
    result = model.predict(text, tokenizer)
    results[label_type] = result
    
    # Memory cleanup
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

**Advanced**: Could implement model size estimation and memory monitoring

---

### BLOCKER 7: Label Mapping System
**Status**: 🟢 Available in models, need integration  
**Issue**: Models output numeric predictions, need human-readable labels

**Available Data**:
```json
// From each model's label_mappings.json
{
  "label_discipline": {
    "label2id": {"Computer Science": 0, "Engineering": 1, "Mathematics": 2},
    "id2label": {"0": "Computer Science", "1": "Engineering", "2": "Mathematics"}, 
    "label_names": {...},
    "num_classes": 3
  }
}
```

**Implementation**:
```python
def _load_label_mappings(self, model_info: ModelInfo) -> Dict[int, str]:
    with open(model_info.final_model_path / "label_mappings.json") as f:
        mappings = json.load(f)
    
    # Extract id2label for the relevant label type
    label_key = f"label_{model_info.label_type}"  # e.g., "label_discipline"
    return {int(k): v for k, v in mappings[label_key]["id2label"].items()}
```

---

### BLOCKER 8: Error Handling & Validation
**Status**: 🔴 Missing comprehensive error handling  

**Required Validations**:
```python
class ValidationError(Exception): pass
class ModelNotFoundError(Exception): pass
class PDFExtractionError(Exception): pass

def validate_inputs(pdf_path: str, label_types: List[str]) -> None:
    # 1. PDF file exists and readable
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # 2. Label types are valid
    valid_labels = ['discipline', 'field', 'method']
    invalid = [l for l in label_types if l not in valid_labels]
    if invalid:
        raise ValidationError(f"Invalid label types: {invalid}")
    
    # 3. Models exist for requested label types
    for label_type in label_types:
        models = model_manager.search_models(label_type=label_type)
        if not models:
            raise ModelNotFoundError(f"No trained models found for label type: {label_type}")
```

## Implementation Phases

### Phase 1: Core Infrastructure (Day 1)
1. ✅ Create `ai_docs/` directory
2. ⏳ Implement `PDFTextExtractor` class  
3. ⏳ Create basic `cli_inference.py` with argument parsing
4. ⏳ Implement `ModelSelector` with size-based ranking

### Phase 2: Inference Pipeline (Day 2)
1. ⏳ Implement `InferencePipeline` class
2. ⏳ Add tokenizer discovery logic
3. ⏳ Integrate label mapping system
4. ⏳ Add basic error handling

### Phase 3: Production Features (Days 3-7)
1. ⏳ Memory optimization and monitoring
2. ⏳ Comprehensive error handling
3. ⏳ Output formatting (JSON/text)
4. ⏳ Confidence scoring
5. ⏳ Performance benchmarking
6. ⏳ Unit tests and integration tests

## File Structure

```
paper_classifier/src/paper_classifier/
├── cli_inference.py          # Main CLI entry point
├── pdf_extractor.py          # PDF→text extraction
├── inference_pipeline.py     # Multi-label orchestration  
├── model_selector.py         # Model selection & ranking
├── ai_docs/                  # Implementation documentation
│   ├── inference_implementation_blockers.md
│   ├── pdf_extraction_integration.md
│   ├── model_selection_algorithms.md
│   └── cli_architecture_design.md
```

## Testing Strategy

```bash
# Unit tests
pytest tests/test_pdf_extractor.py
pytest tests/test_model_selector.py  
pytest tests/test_inference_pipeline.py
pytest tests/test_cli_inference.py

# Integration tests
python -m paper_classifier.cli_inference test_paper.pdf --labels discipline
python -m paper_classifier.cli_inference test_paper.pdf --labels discipline field method --output-format json
```

## Success Metrics

**MVP Success** (Day 2):
- `classify-paper paper.pdf` successfully outputs discipline/field/method predictions
- Model selection works with fallback logic
- Basic error handling prevents crashes

**Production Success** (Day 7):
- Memory efficient (handles large models)
- Comprehensive error messages
- <2 second inference time for 4_8k models
- >95% prediction accuracy on test papers
- Full test coverage

## Dependencies

**External**:
- PyTorch/CUDA for model inference
- Transformers library for tokenization
- PDF processing libraries (via paper_dataset or direct)

**Internal**:
- ModelManager (existing)
- MambaTextClassification.predict() (existing, recently enhanced)
- Model checkpoints in trained_models/ (existing)