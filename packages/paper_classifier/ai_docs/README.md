# AI Documentation - Paper Classifier CLI Inference Implementation

**Generated**: 2025-09-22  
**Context**: Implementation documentation for PDF classification CLI system

## Overview

This directory contains comprehensive technical documentation for implementing a CLI-based PDF paper classification system using trained Mamba models. The system classifies academic papers across three label types: **discipline**, **field**, and **method**.

## Documentation Structure

### 1. [Implementation Blockers & Overview](./inference_implementation_blockers.md)
**Primary document** - Comprehensive analysis of all implementation challenges and solutions.

**Key Topics:**
- Complete blocker analysis (8 major components)  
- Current system state assessment (39 models available)
- Implementation phases and timelines
- Success metrics and testing strategies
- File structure and dependencies

**Quick Start:** Read this first to understand scope and complexity.

### 2. [PDF Text Extraction Integration](./pdf_extraction_integration.md)
**Focus**: PDF processing pipeline and cross-package integration strategies.

**Key Topics:**
- Multi-strategy text extraction (pdfplumber → PyMuPDF → OCR)
- Dependency management across paper_classifier and paper_dataset packages
- Error handling for corrupted/scanned PDFs
- Performance optimization and caching
- Memory-efficient processing for large documents

**Implementation Highlights:**
```python
# Multi-fallback extraction strategy
PDFTextExtractor()
├── pdfplumber (text-based PDFs)
├── PyMuPDF (fallback)  
└── pytesseract + OCR (scanned PDFs)
```

### 3. [CLI Architecture Design](./cli_architecture_design.md)
**Focus**: End-to-end CLI system architecture and component design.

**Key Topics:**
- Complete CLI interface specification (`classify-paper` command)
- Component interaction diagrams
- Data flow architecture (PDF → Text → Models → Results)
- Error handling strategies and user experience
- Output formatting (text/JSON/CSV)
- Memory management for multi-model inference

**Usage Examples:**
```bash
# Basic usage
classify-paper paper.pdf

# Advanced usage  
classify-paper paper.pdf --labels discipline field --size-preference 8_16k --output-format json
```

### 4. [Model Selection Algorithms](./model_selection_algorithms.md)
**Focus**: Automated model selection based on size preferences and performance metrics.

**Key Topics:**
- Size preference expansion algorithm (bidirectional fallback)
- Multi-criteria model ranking within size categories
- Performance metric analysis (eval_loss, eval_accuracy)
- Advanced selection strategies (speed-optimized, confidence-based, ensemble)
- Edge case handling and validation

**Algorithm Overview:**
```
User Request: --size-preference 4_8k
    ↓
Size Expansion: ['4_8k', '1_4k', '8_16k', '16_32k', '32_64k']  
    ↓
For each size: Rank by eval_loss (ascending = better)
    ↓  
Select: First available model with best metrics
```

## Implementation Status

### ✅ Completed Components
- **ModelManager**: Comprehensive model discovery and metadata extraction
- **MambaTextClassification.predict()**: Enhanced with proper device handling and error handling
- **CLI Entry Point**: `classify-paper` command registered in pyproject.toml
- **Model Inventory**: 39 trained models across 3 label types with 8 models having performance metrics

### 🟡 Partially Implemented
- **PDF Dependencies**: Available in `paper_dataset` package, need integration strategy
- **Model Selection Logic**: Algorithm designed, implementation pending
- **Basic Infrastructure**: Core classes designed, need implementation

### 🔴 Missing Components  
- **cli_inference.py**: Main CLI module (referenced but doesn't exist)
- **pdf_extractor.py**: PDF text extraction pipeline
- **inference_pipeline.py**: Multi-label orchestration system
- **model_selector.py**: Model selection implementation

## Quick Implementation Guide

### Phase 1: Core Infrastructure (1-2 days)
1. Implement `PDFTextExtractor` with multi-strategy extraction
2. Create `cli_inference.py` with basic argument parsing
3. Implement `ModelSelector` with size-based ranking
4. Basic error handling and validation

### Phase 2: Inference Pipeline (1-2 days)  
1. Implement `InferencePipeline` for multi-label orchestration
2. Add tokenizer discovery and model loading logic
3. Integrate label mapping system from model checkpoints  
4. Memory management for sequential model loading

### Phase 3: Production Features (2-3 days)
1. Comprehensive error handling and user guidance
2. Output formatting (JSON, CSV, verbose text)
3. Performance optimization (caching, memory monitoring)
4. Unit tests and integration tests

## Architecture Summary

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   CLI Args  │───▶│  PDFExtractor    │───▶│ InferencePipeline   │
└─────────────┘    └──────────────────┘    └─────────────────────┘
                                                       │
┌─────────────┐    ┌──────────────────┐               ▼
│ JSON/Text   │◀───│  ResultFormatter │    ┌─────────────────────┐
│ Output      │    └──────────────────┘    │ ModelSelector       │
└─────────────┘                           │ + ModelManager      │
                                          └─────────────────────┘
                                                       │
                   ┌──────────────────┐               ▼  
                   │ MambaText        │    ┌─────────────────────┐
                   │ Classification   │◀───│ Model Loading &     │
                   └──────────────────┘    │ Inference           │
                                          └─────────────────────┘
```

## Key Technical Decisions

1. **PDF Processing**: Optional dependencies with multi-strategy extraction
2. **Model Selection**: Size-preference expansion with metric-based ranking  
3. **Memory Management**: Sequential model loading with explicit cleanup
4. **Error Handling**: Graceful degradation with helpful user guidance
5. **Output Formats**: Support for text, JSON, and CSV outputs
6. **Cross-Package Integration**: Leverage existing `paper_dataset` PDF capabilities

## Success Metrics

**MVP Success** (2-3 days):
- `classify-paper paper.pdf` produces predictions for all 3 label types  
- Automatic model selection works with fallback logic
- Basic error handling prevents crashes

**Production Success** (1 week):
- Memory efficient (handles multiple large models sequentially)
- <2 second inference time for 4_8k models
- Comprehensive error messages and user guidance
- Full test coverage and validation

## Dependencies

### Required Packages
```toml
# Core ML (already in paper_classifier)
torch, transformers, mamba-ssm, pandas, numpy

# PDF Processing (add to paper_classifier or import from paper_dataset)  
pdfplumber, pymupdf, pdf2image, pytesseract

# CLI and utilities (already available)
argparse, pathlib, json, logging
```

### Hardware Requirements
- **GPU**: Recommended for inference (CUDA support)
- **RAM**: 8GB+ (for loading 32_64k models)
- **Storage**: Models directory (~5GB for all trained models)

## Next Steps

1. **Review Documentation**: Start with `inference_implementation_blockers.md` for complete context
2. **Choose Implementation Strategy**: Decide on PDF dependency approach (cross-package vs. direct)  
3. **Begin Phase 1**: Implement core infrastructure components
4. **Test Incrementally**: Validate each component before moving to next phase
5. **Optimize for Production**: Add performance monitoring and comprehensive testing

## Contact & Support

This documentation was generated to support the implementation of the paper classifier CLI system. Each document contains detailed code examples, error handling strategies, and testing frameworks to facilitate rapid development.

For questions about specific implementation details, refer to the individual documentation files or examine the existing codebase components (ModelManager, MambaTextClassification, etc.).