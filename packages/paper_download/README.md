# Paper Download CLI

A powerful CLI tool for downloading and organizing computing research papers from ArXiv with hierarchical organization based on CC2020 computing curriculum or ArXiv categories.

## 🎯 Features

- **📚 Hierarchical Organization**: Automatically organize papers by CC2020 academic disciplines or ArXiv research categories
- **🎓 CC2020 Computing Curriculum**: Map papers to CS, SE, DS, CSEC, IT, IS, CE disciplines
- **🔍 Smart Filtering**: Intelligent interpretation of discipline codes and ArXiv categories
- **📊 Comprehensive Metadata**: JSON metadata saved alongside each PDF with classification details
- **⚡ Batch Processing**: Download multiple papers from files or categories
- **📈 Statistics Tracking**: View download and classification statistics

## 🚀 Installation

### Prerequisites
- Python 3.9+
- pip or uv package manager

### Install with pip
```bash
# Clone the repository
git clone <repository-url>
cd paper_download

# Install in editable mode
pip install -e .
```

### Install with UV (Recommended)
```bash
# Install UV if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the package
uv pip install -e .
```

## 📖 CLI Commands Overview

```bash
paper-download --help              # Show all commands
paper-download download --help      # Show download options
paper-download categories --help    # Show category options
```

## 🔧 Main Commands

### 1. `download` - Smart Paper Downloading

The main command for downloading papers with intelligent filtering and organization.

#### Basic Usage
```bash
# Download 10 random papers with CC2020 organization (default)
paper-download download --max 10

# Download Data Science papers (searches all DS-related categories)
paper-download download --filter DS --max 20

# Download from specific ArXiv category
paper-download download --filter cs.LG --max 15

# Use ArXiv organization instead of CC2020
paper-download download --mapper arxiv --filter cs.LG --max 10
```

#### Options
- `--mapper [cc2020|arxiv]` - Choose organization scheme (default: cc2020)
- `--filter, -f` - Smart filter for disciplines or categories
- `--max, -m` - Maximum papers to download (default: 10)
- `--output-dir, -o` - Output directory (default: ./papers)
- `--show-stats` - Show classification statistics

#### Smart Filter Examples

**CC2020 Mapper (Default)**
```bash
# Download papers from all Data Science categories
paper-download download --filter DS --max 30
# Searches: cs.LG, stat.ML, cs.DB, cs.IR

# Download Software Engineering papers
paper-download download --filter SE --max 20
# Searches: cs.SE, cs.PL

# Download Cybersecurity papers
paper-download download --filter CSEC --max 15
# Searches: cs.CR

# Download specific ArXiv category, organized by CC2020
paper-download download --filter cs.AI --max 25
# Downloads cs.AI papers, organizes under CS_Computer_Science/cs.AI/
```

**ArXiv Mapper**
```bash
# Use ArXiv's native categorization
paper-download download --mapper arxiv --filter cs.LG --max 20
# Output: papers/ArXiv/cs.LG/

paper-download download --mapper arxiv --filter stat.ML --max 15
# Output: papers/ArXiv/stat.ML/
```

### 2. `download-ids` - Download Specific Papers

Download papers by their ArXiv IDs.

```bash
# Download single paper
paper-download download-ids 2301.12345

# Download multiple papers
paper-download download-ids 2301.12345 2302.54321 2303.11111

# Use specific mapper
paper-download download-ids 2301.12345 --mapper arxiv

# Custom output directory
paper-download download-ids 2301.12345 --output-dir ./my-papers
```

### 3. `batch` - Batch Download from File

Download papers listed in a text file.

```bash
# Create input file
cat > papers.txt << EOF
# Comments are ignored
2301.12345
2302.54321
cat:cs.LG  # Download 10 from category
EOF

# Run batch download
paper-download batch papers.txt

# With options
paper-download batch papers.txt --mapper cc2020 --output-dir ./dataset

# Generate download report
paper-download batch papers.txt --report
```

### 4. `categories` - View Available Categories

Display organization schemes and their categories.

```bash
# Show all available mappers
paper-download categories

# Show CC2020 disciplines and their mapped ArXiv categories
paper-download categories --mapper cc2020

# Show ArXiv categories organized by domain
paper-download categories --mapper arxiv
```

### 5. `stats` - View Download Statistics

Show statistics from existing downloads.

```bash
# Show download statistics
paper-download stats

# Output includes:
# - Papers per mapper (CC2020, ArXiv)
# - Distribution across disciplines/categories
# - Metadata file counts
```

## 📁 Output Structure

### CC2020 Organization (Default)
```
papers/
└── CC2020/
    ├── CS_Computer_Science/
    │   ├── cs.AI/
    │   │   ├── arxiv_2301.12345_Paper_Title.pdf
    │   │   └── arxiv_2301.12345_metadata.json
    │   ├── cs.CV/
    │   └── cs.DS/
    ├── SE_Software_Engineering/
    │   ├── cs.SE/
    │   └── cs.PL/
    ├── DS_Data_Science/
    │   ├── cs.LG/
    │   └── stat.ML/
    ├── CSEC_Cybersecurity/
    │   └── cs.CR/
    └── Unclassified/
        └── low_confidence/  # Papers with confidence < 0.3
```

### ArXiv Organization
```
papers/
└── ArXiv/
    ├── cs.AI/
    │   ├── arxiv_2301.12345_Paper_Title.pdf
    │   └── arxiv_2301.12345_metadata.json
    ├── cs.LG/
    ├── cs.SE/
    └── stat.ML/
```

## 🗂️ CC2020 Disciplines

| Code | Discipline | Description | Example Categories |
|------|------------|-------------|-------------------|
| **CS** | Computer Science | Theory, algorithms, AI, graphics | cs.AI, cs.CV, cs.DS |
| **SE** | Software Engineering | Development, testing, methodologies | cs.SE, cs.PL |
| **DS** | Data Science | Analytics, ML applications, big data | cs.LG, stat.ML, cs.DB |
| **CSEC** | Cybersecurity | Security, cryptography, privacy | cs.CR |
| **CE** | Computer Engineering | Hardware/software interface | cs.AR, cs.ET |
| **IT** | Information Technology | Infrastructure, networking | cs.NI, cs.DC, cs.OS |
| **IS** | Information Systems | Business computing, enterprise | cs.CY, cs.SI |

## 📊 Common Workflows

### Research Paper Collection
```bash
# Collect papers for ML research
paper-download download --filter DS --max 50

# Collect papers across multiple disciplines
for disc in DS SE CS CSEC; do
    paper-download download --filter $disc --max 20
done
```

### Teaching Material Organization
```bash
# Organize by academic curriculum (CC2020)
paper-download download --filter SE --max 30 --output-dir ./teaching/software-eng
paper-download download --filter DS --max 30 --output-dir ./teaching/data-science
```

### Specific Research Areas
```bash
# Focus on specific ArXiv categories
paper-download download --mapper arxiv --filter cs.LG --max 100
paper-download download --mapper arxiv --filter cs.CV --max 100
```

### Mixed Collection Strategy
```bash
# Academic organization for curriculum
paper-download download --filter CS --output-dir ./academic

# Research organization for analysis
paper-download download --mapper arxiv --filter cs.AI --output-dir ./research
```

## 📝 Metadata Format

Each downloaded paper includes a metadata JSON file:

```json
{
  "version": "2.0",
  "download_info": {
    "timestamp": "2024-08-21T10:30:00Z",
    "tool_version": "1.1.0",
    "mapper_used": "CC2020",
    "organization_path": ["DS_Data_Science", "cs.LG"]
  },
  "paper_info": {
    "paper_id": "2301.12345",
    "title": "Paper Title",
    "authors": ["Author 1", "Author 2"],
    "abstract": "...",
    "year": 2023,
    "url": "http://arxiv.org/abs/2301.12345"
  },
  "classification": {
    "arxiv_categories": ["cs.LG", "stat.ML"],
    "mapping_confidence": 0.85,
    "mapping_method": "category"
  }
}
```

## 🔧 Configuration

### Default Settings
- **Default mapper**: CC2020 (academic organization)
- **Output directory**: `./papers`
- **Max papers**: 10 per search
- **Rate limiting**: 0.5s between downloads
- **Confidence threshold**: 0.3 for CC2020 classification

### Environment Variables
```bash
# Set custom output directory
export PAPER_DOWNLOAD_OUTPUT_DIR=/path/to/papers

# Enable verbose logging
export PAPER_DOWNLOAD_VERBOSE=1
```

## 🧪 Testing

```bash
# Quick test - download 1 paper
paper-download download --filter cs.SE --max 1

# Test specific mapper
paper-download download --mapper arxiv --filter cs.LG --max 1

# Verify organization
paper-download stats
```

## ⚠️ Limitations

- Currently supports ArXiv only
- Rate limited to respect ArXiv guidelines (0.5s between requests)
- Papers with CC2020 confidence < 0.3 go to Unclassified folder
- Maximum filename length: 50 characters (truncated if longer)

## 🤝 Contributing

Contributions are welcome! The mapper architecture is designed to be extensible:

```python
# Example: Add a custom mapper
from paper_download.mappers import BaseMapper, MapperRegistry

class CustomMapper(BaseMapper):
    def get_mapper_name(self):
        return "Custom"
    
    def map_paper(self, metadata):
        # Your organization logic
        return MappingResult(...)

# Register the mapper
MapperRegistry.register_mapper("custom", CustomMapper)
```

## 📄 License

MIT License - Part of the CM3070 Computer Science Final Project

## 🆘 Troubleshooting

### Common Issues

**No papers downloaded**
- Check your internet connection
- Verify the filter/category exists
- Try with a different category

**Papers in Unclassified folder**
- This happens when CC2020 confidence is low
- Try using ArXiv mapper for direct categorization

**Import errors**
```bash
# Reinstall the package
pip install -e . --force-reinstall
```

## 📚 Additional Resources

- [ArXiv API Documentation](https://arxiv.org/help/api)
- [CC2020 Computing Curricula](https://www.acm.org/education/curricula-recommendations)
- [Project Repository](https://github.com/yourusername/paper_download)