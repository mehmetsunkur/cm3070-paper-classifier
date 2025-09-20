"""Paper dataset module for managing paper datasets and processing academic papers."""

__version__ = "0.1.0"

# Import main classes and functions for easy access
from .extractors import PDFExtractor, PaperMetadata, PaperDataset
from .utils import LabelManager, get_label_manager
from .processing import process_papers_batch_parallel_pyarrow, save_results_pyarrow
from .configs import CONFIGS_DIR, LABEL_CONFIG_PATH

__all__ = [
    # Core extractors
    "PDFExtractor",
    "PaperMetadata", 
    "PaperDataset",
    
    # Label management
    "LabelManager",
    "get_label_manager",
    
    # Batch processing
    "process_papers_batch_parallel_pyarrow",
    "save_results_pyarrow",
    
    # Configuration
    "CONFIGS_DIR",
    "LABEL_CONFIG_PATH",
]