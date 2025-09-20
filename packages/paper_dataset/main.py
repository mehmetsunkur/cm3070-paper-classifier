#!/usr/bin/env python3
"""
Example usage of the paper_dataset package.
Demonstrates how to use the migrated functionality.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from src.paper_dataset import (
    PDFExtractor, 
    LabelManager,
    process_papers_batch_parallel_pyarrow,
    save_results_pyarrow
)

def batch_processing():
    """Batch processing PDFs."""
    print("\n=== Batch Processing ===")
    
    # Load environment variables
    load_dotenv()
    
    # Get PAPERS_DIR from environment
    papers_dir = os.getenv("PAPERS_DIR")
    if not papers_dir:
        print("PAPERS_DIR environment variable not found. Please set it in your .env file.")
        return
    
    pdf_dir = Path(papers_dir)
    
    if not pdf_dir.exists():
        print(f"Directory {pdf_dir} not found. Update the path in the script.")
        return
    
    # Process papers in batch
    output_file = process_papers_batch_parallel_pyarrow(
        pdf_dir=pdf_dir,
        limit=10,  # Process only 10 papers for testing
        n_workers=2,  # Use fewer workers for testing
        batch_size=5,
        worker_timeout=30
    )
    
    if output_file:
        print(f"Batch processing completed! Output: {output_file}")
        
        # Split into train/val/test
        save_results_pyarrow(output_file)
        print("Data splits created successfully!")

def main():
    batch_processing() 
if __name__ == "__main__":
    main()