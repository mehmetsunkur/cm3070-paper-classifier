# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CM3070 Paper Classifier is a monorepo containing modules for downloading, processing, and classifying academic papers using Mamba-based neural networks. The project consists of three main packages that work together in a data pipeline.

## Architecture

This is a Python monorepo with three interconnected packages:

- **paper_download**: Downloads papers from ArXiv and other sources using collectors pattern
- **paper_dataset**: Processes and manages paper datasets, including PDF text extraction and batch processing
- **paper_classifier**: Mamba-based text classification models for academic papers with configurable training pipelines

Each package has its own `pyproject.toml`, source directory under `src/`, and isolated virtual environments.

## Essential Commands

### Installation and Setup
```bash
# Install root package with dev dependencies
pip install -e .[dev]

# Install specific package for development (use uv for faster installs)
cd packages/paper_download && pip install -e .[dev]
cd packages/paper_classifier && pip install -e .
cd packages/paper_dataset && pip install -e .[dev]

# Alternative using uv (recommended)
cd packages/paper_download && uv pip install -e .[dev]
```

### Development Commands
```bash
# Code formatting (applies to entire monorepo)
black .
isort .

# Type checking
mypy packages/*/src

# Linting
flake8 packages/*/src
ruff check packages/*/src  # For paper_download package

# Testing
pytest  # Runs all package tests via testpaths configuration
pytest packages/paper_classifier/tests  # Test specific package
```

### Training and Execution
```bash
# Train paper classifier with configuration
packages/paper_classifier/scripts/train.sh --config configs/example_config.yaml

# Alternative training command
uv run accelerate launch --num_processes=1 -m paper_classifier.main --config configs/example_config.yaml

# Download papers using CLI
paper-download  # Available after installing paper_download package

# Dataset processing
cd packages/paper_dataset && make install-spacy
```

## Configuration System

The paper classifier uses hierarchical YAML configurations:

- **Base configs**: `packages/paper_classifier/configs/base.yaml` contains shared settings
- **Experiment configs**: Override base settings for specific experiments
- **Model configs**: Named patterns like `model_label_discipline_1_4k_s1000.yaml`

Key configuration patterns:
- `label_*`: Dataset label configuration
- `model_*`: Model-specific training configurations  
- Size indicators: `1_4k`, `8_16k`, `32_64k` represent token context lengths
- Sample counts: `s1000`, `s5000`, `s10000` indicate dataset sizes

## Package Dependencies

The project requires Python 3.8+ (3.13+ for paper_classifier, 3.10+ for paper_dataset) and uses:

- **paper_download**: arxiv, requests, click, pymupdf, beautifulsoup4, rich, scikit-learn
- **paper_classifier**: torch, transformers, mamba-ssm, wandb, accelerate, datasets, huggingface-hub
- **paper_dataset**: spacy (with en_core_web_sm model), pdfplumber, pytesseract, loguru, multiprocessing-logging

## Data Flow

1. **paper_download** → Downloads academic papers from sources
2. **paper_dataset** → Processes PDFs, extracts text, creates datasets  
3. **paper_classifier** → Trains Mamba models on processed datasets

## Key Directories

- `packages/*/src/`: Source code for each package
- `packages/paper_classifier/configs/`: Training configurations
- `packages/paper_classifier/trained_models/`: Model outputs
- `packages/paper_classifier/logs/`: Training logs
- `packages/paper_download/papers/`: Downloaded papers
- `scripts/`: Cross-package utilities

## Environment Variables

- `WANDB_PROJECT`: Weights & Biases project name
- `PAPER_CLASSIFIER_*`: Override default directories for training script
  - `PAPER_CLASSIFIER_CONFIG_DIR`: Custom config directory
  - `PAPER_CLASSIFIER_MODEL_DIR`: Custom models directory  
  - `PAPER_CLASSIFIER_LOG_DIR`: Custom log directory
  - `PAPER_CLASSIFIER_GPU_COUNT`: Number of GPUs to use
- Training uses `.env` files loaded automatically

## Testing Strategy

Each package maintains independent test suites under `packages/*/tests/`. The root `pyproject.toml` configures pytest to discover tests across all packages using `testpaths = ["packages/*/tests"]`.