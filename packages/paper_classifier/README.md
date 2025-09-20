# Paper Classifier

A clean implementation of the Mamba-based text classification system for academic papers. This package provides a modular, component-based architecture for training text classification models on research paper data.

## Features

- **Component-based Architecture**: Modular design with separate components for data loading, model building, training, optimization, and logging
- **Configurable Training**: YAML-based configuration system with support for inheritance and CLI overrides
- **Flexible Data Processing**: Automated subset generation with balanced sampling across labels
- **Multi-GPU Support**: Built-in support for distributed training using Accelerate
- **Experiment Tracking**: Integration with Weights & Biases for experiment monitoring

## Installation

### Prerequisites

- Python 3.13+
- CUDA-compatible GPU (recommended)
- UV package manager

### Install from Source

```bash
cd packages/paper_classifier
uv pip install -e .
```

### Dependencies

The package requires several key dependencies:
- `torch` - PyTorch framework
- `transformers` - Hugging Face transformers
- `mamba-ssm` - Mamba state space models
- `accelerate` - Multi-GPU training support
- `wandb` - Experiment tracking
- `pandas` - Data processing
- `pyyaml` - Configuration files

See `pyproject.toml` for the complete list.

## Quick Start

### 1. Setup Environment

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
# Edit .env with your settings (especially WANDB_API_KEY)
```

### 2. Prepare Configuration

Copy and customize the example configuration:

```bash
cp configs/example_config.yaml configs/my_config.yaml
# Edit my_config.yaml with your dataset paths and training parameters
```

### 3. Run Training

Using the training script:

```bash
scripts/train.sh --config configs/my_config.yaml --gpu-count 1
```

Or directly with Python:

```bash
uv run python -m paper_classifier.main --config configs/my_config.yaml
```

## Configuration

The system uses YAML configuration files with support for inheritance. Key configuration sections:

### Data Configuration
```yaml
# Data settings
data_parquet_path: "path/to/your/dataset.parquet"  # Optional: direct path
data_train_ratio: 0.7
data_val_ratio: 0.15

# Subset generation (if no direct parquet_path)
subset_generation_enabled: true
subset_generation_source_file: "path/to/source.parquet"
subset_generation_size_rank: "4-8k"
subset_generation_sample_count: 1000
subset_generation_label_column: "label_discipline"
```

### Model Configuration
```yaml
# Model settings
model_name: "state-spaces/mamba-130m"
tokenizer_model_name: "EleutherAI/gpt-neox-20b"
max_seq_length: 512
```

### Training Configuration
```yaml
# Training parameters
training_batch_size_train: 4
training_batch_size_eval: 8
training_learning_rate: 1e-4
training_num_epochs: 3
training_warmup_steps: 100
```

## Project Structure

```
paper_classifier/
├── src/paper_classifier/
│   ├── __init__.py          # Package initialization
│   ├── main.py             # Main entry point
│   ├── pipeline.py         # Training pipeline orchestrator
│   ├── subset_generator.py # Data subset generation
│   ├── parquet_dataset.py  # Dataset handling
│   ├── data_set_utils.py   # Dataset utilities
│   ├── components/         # Modular components
│   │   ├── __init__.py
│   │   ├── data_loader.py  # Data loading component
│   │   ├── model_builder.py # Model construction
│   │   ├── trainer_component.py # Training logic
│   │   ├── optimization.py # Optimization utilities
│   │   ├── logging.py      # Logging and WandB integration
│   │   ├── label_mapper.py # Label mapping utilities
│   │   └── validation.py   # Validation utilities
│   └── config/
│       ├── __init__.py
│       └── loader.py       # Configuration loading
├── configs/                # Configuration files
│   └── example_config.yaml
├── scripts/
│   └── train.sh           # Training script
├── pyproject.toml         # Package configuration
├── .env.example          # Environment template
└── README.md
```

## Component Architecture

### DataLoaderComponent
Handles dataset loading, tokenization, and preprocessing:
- Loads parquet datasets with configurable train/val/test splits
- Manages tokenizer initialization and configuration
- Creates data collators for batching

### ModelBuilderComponent  
Constructs and configures models:
- Loads pre-trained Mamba models
- Configures classification heads based on dataset
- Handles device placement and optimization

### TrainerComponent
Manages the training process:
- Creates Hugging Face Trainer instances
- Handles training arguments and callbacks
- Manages model saving and evaluation

### OptimizationComponent
Optimization and performance utilities:
- Memory management and GPU optimization
- Random seed setting for reproducibility
- Learning rate scheduling

### LoggingComponent
Experiment tracking and logging:
- Weights & Biases integration
- Directory management for outputs
- Configuration and metadata saving

## Usage Examples

### Basic Training

```python
from paper_classifier import main
import sys

# Set up arguments
sys.argv = ['main.py', '--config', 'configs/my_config.yaml']
main()
```

### Advanced Usage

```python
from paper_classifier import TrainingPipeline
from paper_classifier.config import load_config

# Load configuration
config = load_config('configs/my_config.yaml')

# Create and run pipeline
pipeline = TrainingPipeline(config)
result = pipeline.run(
    parquet_path='data/my_dataset.parquet',
    model_name='my_experiment'
)

if result['success']:
    print(f"Model saved to: {result['model_path']}")
    print(f"Test results: {result['test_results']}")
```

## Development

### Project Origins

This package is a clean refactoring of the original `mamba-text-classification-mvp1` project, extracting only the essential components needed for training. The original project contained:

- ~200+ files including UI components, test files, and experimental code
- Multiple legacy trainer implementations
- Extensive logging and evaluation artifacts

This clean implementation reduces complexity by:
- Extracting only the 15-20 core files needed for training
- Implementing proper Python package structure
- Updating import paths and dependencies
- Providing clear configuration management

### Contributing

When contributing to this package:

1. Follow the existing component-based architecture
2. Update tests when adding new functionality
3. Maintain configuration compatibility
4. Document new features in this README

### Troubleshooting

**Import Errors**: Ensure all dependencies are installed and the package is installed in development mode (`uv pip install -e .`)

**CUDA Errors**: Verify CUDA compatibility and set `CUDA_VISIBLE_DEVICES` appropriately

**Configuration Errors**: Check YAML syntax and ensure all required fields are present

**Memory Issues**: Reduce batch size, sequence length, or enable gradient checkpointing in the configuration

## License

This project maintains the same license as the original mamba-text-classification-mvp1 project.