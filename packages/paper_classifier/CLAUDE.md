# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Paper Classifier is a clean Mamba-based text classification system for academic papers. This is a single Python package with modular component architecture for training text classification models on research paper data.

## Architecture

This is a Python package with component-based architecture:

- **Component-based Design**: Modular components for data loading, model building, training, optimization, and logging
- **Configurable Training**: YAML-based configuration system with inheritance support
- **Flexible Data Processing**: Automated subset generation with balanced sampling
- **Multi-GPU Support**: Built-in distributed training using Accelerate
- **Experiment Tracking**: Weights & Biases integration

## Essential Commands

### Installation and Setup
```bash
# Install package for development
uv pip install -e .

# Copy environment template and configure
cp .env.example .env
# Edit .env with your settings (especially WANDB_API_KEY)
```

### Development Commands
```bash
# Code formatting
black .
isort .

# Type checking
mypy src/

# Linting
flake8 src/
ruff check src/

# Testing
pytest tests/
```

### Training and Execution
```bash
# Train with configuration using script
./scripts/train.sh --config configs/my_config.yaml --gpu-count 1

# Alternative training commands
uv run python -m paper_classifier.main --config configs/my_config.yaml
uv run accelerate launch --num_processes=1 -m paper_classifier.main --config configs/my_config.yaml

# Command line tool (after installation)
train-paper-classifier --config configs/my_config.yaml
```

## Configuration System

The system uses hierarchical YAML configurations with inheritance:

- **Base configs**: `configs/base.yaml` contains shared settings
- **Label configs**: `label_*` patterns define dataset label configurations
- **Model configs**: `model_*` patterns override for specific experiments

Key configuration patterns:
- `label_discipline_*`, `label_field_*`, `label_method_*`: Dataset label types
- Size indicators: `1_4k`, `4_8k`, `8_16k`, `16_32k`, `32_64k`, `64_128k` represent token context lengths
- Sample counts: `s1000`, `s5000`, `s10000` indicate dataset sizes

## Package Dependencies

Requires Python 3.13+ and key dependencies:

- **Core ML**: torch, transformers, mamba-ssm, datasets, scikit-learn
- **Training**: accelerate, evaluate, bitsandbytes
- **Experiment**: wandb, python-dotenv
- **Data**: pandas, numpy, pyyaml, duckdb

## Project Structure

```
paper_classifier/
├── src/paper_classifier/          # Main package source
│   ├── main.py                   # Entry point
│   ├── pipeline.py               # Training pipeline orchestrator
│   ├── subset_generator.py       # Data subset generation
│   ├── parquet_dataset.py        # Dataset handling
│   └── components/               # Modular components
│       ├── data_loader.py        # Data loading
│       ├── model_builder.py      # Model construction
│       ├── trainer_component.py  # Training logic
│       ├── optimization.py       # Optimization utilities
│       ├── logging.py            # WandB integration
│       ├── label_mapper.py       # Label mapping
│       └── validation.py         # Validation utilities
├── configs/                      # YAML configurations
├── scripts/                      # Shell scripts
├── tests/                        # Test suite
├── trained_models/               # Model outputs
├── logs/                         # Training logs
└── data/                         # Dataset storage
```

## Environment Variables

- `WANDB_API_KEY`: Required for experiment tracking
- `WANDB_PROJECT`: Weights & Biases project name
- `PAPER_CLASSIFIER_*`: Override default directories for training script
- Configuration loaded from `.env` files automatically

## Component Architecture

The system uses five main components that work together:

1. **DataLoaderComponent**: Dataset loading, tokenization, train/val/test splits
2. **ModelBuilderComponent**: Model construction, classification heads, device placement
3. **TrainerComponent**: Training process, Hugging Face Trainer, callbacks
4. **OptimizationComponent**: Memory management, seeding, learning rate scheduling  
5. **LoggingComponent**: WandB integration, directory management, metadata saving

## Development Notes

This package is a clean refactoring from a larger project (`mamba-text-classification-mvp1`), extracting only the 15-20 core files needed for training from ~200+ original files. It implements proper Python package structure with updated import paths and dependencies.