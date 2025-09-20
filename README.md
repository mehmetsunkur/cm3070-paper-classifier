# CM3070 Paper Classifier

A monorepo containing modules for downloading, processing, and classifying academic papers.

## Project Structure

```
cm3070-paper-classifier/
├── pyproject.toml              # Root configuration
├── packages/
│   ├── paper_download/         # Paper downloading functionality
│   │   ├── src/paper_download/
│   │   ├── tests/
│   │   └── pyproject.toml
│   ├── paper_dataset/          # Dataset management
│   │   ├── src/paper_dataset/
│   │   ├── tests/
│   │   └── pyproject.toml
│   └── paper_classifier/       # Classification models
│       ├── src/paper_classifier/
│       ├── tests/
│       └── pyproject.toml
├── scripts/                    # Build and utility scripts
└── tools/                      # Development tools
```

## Installation

```bash
pip install -e .
```

## Development

Install development dependencies:
```bash
pip install -e .[dev]
```

Run tests:
```bash
pytest
```

Format code:
```bash
black .
isort .
```

## Modules

- **paper_download**: Download papers from various sources (ArXiv, etc.)
- **paper_dataset**: Manage and preprocess paper datasets
- **paper_classifier**: Train and run classification models