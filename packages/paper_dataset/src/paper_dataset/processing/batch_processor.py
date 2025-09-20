#!/usr/bin/env python3
"""
PyArrow-only batch processing script for papers.
Memory-efficient processing without pandas dependency.
"""

import sys
import os
import json
import gc
import signal
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Set
import time
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from tqdm import tqdm

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import from the paper_dataset package
from ..extractors.pdf_processor import PDFExtractor
from ..utils.label_manager import LabelManager

DEFAULT_OUTPUT_DIR = Path.cwd() / "data" / "processed_pyarrow"

# Global variables for worker processes
pdf_extractor = None
label_manager = None
worker_start_time = None

def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("Processing timeout exceeded")

def init_worker():
    """Initialize worker with shared resources and timeout handling."""
    global pdf_extractor, label_manager, worker_start_time
    
    # Set up signal handler for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    
    # Create a modified PDFExtractor that doesn't load spaCy
    class PDFExtractorNoSpacy(PDFExtractor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.nlp = None
    
    # Load resources once per worker
    pdf_extractor = PDFExtractorNoSpacy(use_ocr=False)
    label_manager = LabelManager()
    worker_start_time = time.time()
    
    print(f"Worker {os.getpid()} initialized")


class LabelAssigner:
    """Lightweight label assigner that uses global label_manager."""
    
    def assign_labels(self, metadata: dict, text: str) -> Dict[str, int]:
        """Assign all labels using the global label manager."""
        global label_manager
        if label_manager is None:
            label_manager = LabelManager()
        return label_manager.assign_all_labels(text, metadata)


def process_single_paper_with_timeout(pdf_path: Path, timeout_seconds: int = 20) -> Optional[Dict]:
    """Process a single paper with enforced timeout."""
    global pdf_extractor, label_manager
    
    # Set alarm for timeout (Unix only)
    if hasattr(signal, 'alarm'):
        signal.alarm(timeout_seconds)
    
    try:
        # Use global extractor
        if pdf_extractor is None:
            class PDFExtractorNoSpacy(PDFExtractor):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.nlp = None
            pdf_extractor = PDFExtractorNoSpacy(use_ocr=False)
            
        label_assigner = LabelAssigner()
        
        # Extract text and metadata
        text, metadata = pdf_extractor.extract(pdf_path)
        
        # Load ArXiv metadata if available
        json_path = pdf_path.with_suffix('.json').with_name(pdf_path.stem + '_metadata.json')
        arxiv_metadata = {}
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    arxiv_data = json.load(f)
                    if 'classification' in arxiv_data:
                        arxiv_metadata['arxiv_categories'] = arxiv_data['classification'].get('arxiv_categories', [])
            except:
                pass
        
        # Assign labels
        labels = label_assigner.assign_labels(arxiv_metadata, text)
        
        # Calculate size_rank based on text length in bytes
        text_bytes = len(text.encode('utf-8'))
        kb = text_bytes / 1024  # Convert to KB
        
        if kb < 1:
            size_rank = '<1k'
        elif kb < 4:
            size_rank = '1-4k'
        elif kb < 8:
            size_rank = '4-8k'
        elif kb < 16:
            size_rank = '8-16k'
        elif kb < 32:
            size_rank = '16-32k'
        elif kb < 64:
            size_rank = '32-64k'
        elif kb < 128:
            size_rank = '64-128k'
        elif kb < 256:
            size_rank = '128-256k'
        elif kb < 512:
            size_rank = '256-512k'
        elif kb < 1024:
            size_rank = '512-1024k'
        else:
            size_rank = '>1024k'
        
        # Ensure all fields are serializable (convert to basic Python types)
        record = {
            'paper_id': str(pdf_path.stem),
            'text': str(text) if text else "",
            'title': str(metadata.title) if metadata.title else str(pdf_path.stem),
            'abstract': str(metadata.abstract) if metadata.abstract else "",
            'word_count': int(metadata.word_count) if metadata.word_count else 0,
            'char_count': int(metadata.char_count) if metadata.char_count else 0,
            'pages': int(metadata.pages) if metadata.pages else 0,
            'size_rank': str(size_rank),
            'text_size_kb': float(round(kb, 2)),  # Also store exact size for reference
            'file_path': str(pdf_path)
        }
        
        # Add all labels
        for task, label_value in labels.items():
            record[f'label_{task}'] = label_value
        
        # Cancel alarm
        if hasattr(signal, 'alarm'):
            signal.alarm(0)
        
        # Cleanup
        del text
        gc.collect()
        
        return record
        
    except TimeoutError:
        print(f"⏱️ Timeout processing {pdf_path.name}")
        return None
    except Exception as e:
        if "timeout" in str(e).lower():
            print(f"⏱️ Timeout processing {pdf_path.name}")
        else:
            print(f"❌ Error processing {pdf_path.name}: {e}")
        return None
    finally:
        # Always cancel alarm
        if hasattr(signal, 'alarm'):
            signal.alarm(0)
        gc.collect()


def list_to_pyarrow_table(records: List[Dict]) -> pa.Table:
    """Convert list of dictionaries to PyArrow table efficiently."""
    if not records:
        return None
    
    # Get all field names from first record
    field_names = list(records[0].keys())
    
    # Build columns
    arrays = []
    fields = []
    
    for field_name in field_names:
        # Extract column data
        column_data = [record.get(field_name, None) for record in records]
        
        # Determine type from first non-None value
        sample_value = next((v for v in column_data if v is not None), None)
        
        if sample_value is None:
            # All nulls - assume string
            pa_array = pa.array(column_data, type=pa.string())
        elif isinstance(sample_value, str):
            pa_array = pa.array(column_data, type=pa.string())
        elif isinstance(sample_value, int):
            pa_array = pa.array(column_data, type=pa.int64())
        elif isinstance(sample_value, float):
            pa_array = pa.array(column_data, type=pa.float64())
        else:
            # Default to string
            pa_array = pa.array([str(v) if v is not None else None for v in column_data], type=pa.string())
        
        arrays.append(pa_array)
        fields.append(pa.field(field_name, pa_array.type))
    
    # Create table
    schema = pa.schema(fields)
    table = pa.Table.from_arrays(arrays, schema=schema)
    
    return table


def save_batch_to_disk(batch: List[Dict], batch_num: int, output_dir: Path):
    """Save a batch of results to disk using PyArrow."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert to PyArrow table
        table = list_to_pyarrow_table(batch)
        if table is None:
            return None
            
        batch_file = output_dir / f"batch_{batch_num:05d}.parquet"
        pq.write_table(table, batch_file)
        print(f"  💾 Saved batch {batch_num} with {len(batch)} papers")
        
        # Save checkpoint
        checkpoint_file = output_dir / "checkpoint.json"
        checkpoint = {
            'last_batch': batch_num,
            'total_processed': batch_num * len(batch),
            'timestamp': datetime.now().isoformat()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        return batch_file
    except Exception as e:
        print(f"  ⚠️ Error saving batch: {e}")
        return None


def shuffle_pyarrow_table(table: pa.Table, seed: int = 42) -> pa.Table:
    """Shuffle a PyArrow table efficiently."""
    n = len(table)
    np.random.seed(seed)
    indices = np.random.permutation(n)
    return table.take(pa.array(indices))


def process_papers_batch_parallel_pyarrow(
    pdf_dir: Path, 
    limit: int = None,
    n_workers: int = 50,
    batch_size: int = 1000,
    chunk_size: int = 5000,
    output_dir: Path = None,
    max_retries: int = 2,
    worker_timeout: int = 20
) -> pa.Table:
    """
    Memory-efficient parallel processing using PyArrow only.
    
    Args:
        pdf_dir: Directory containing PDF files
        limit: Maximum number of papers to process
        n_workers: Number of parallel workers
        batch_size: Number of results to accumulate before saving
        chunk_size: Number of futures to process at once
        output_dir: Directory for temporary batch files
        max_retries: Maximum retries for failed PDFs
        worker_timeout: Timeout in seconds for each PDF
    """
    
    # Setup
    if output_dir is None:
        output_dir = Path.cwd() / "temp_batches"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDFs
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    if limit:
        pdf_files = pdf_files[:limit]
    
    total_files = len(pdf_files)
    print(f"🚀 Processing {total_files} PDFs with PyArrow (memory-efficient)")
    print(f"⚙️  Config: {n_workers} workers, timeout={worker_timeout}s, batch={batch_size}")
    
    # Track problematic files
    problematic_files: Set[Path] = set()
    retry_counts: Dict[Path, int] = {}
    
    # Statistics
    successful_count = 0
    failed_count = 0
    timeout_count = 0
    batch_num = 0
    current_batch = []
    
    start_time = time.time()
    actual_workers = min(n_workers, total_files, cpu_count())
    
    # Process in chunks
    for chunk_idx in range(0, total_files, chunk_size):
        chunk_end = min(chunk_idx + chunk_size, total_files)
        chunk_files = pdf_files[chunk_idx:chunk_end]
        
        # Filter out known problematic files that exceeded retries
        chunk_files = [f for f in chunk_files 
                      if f not in problematic_files or retry_counts.get(f, 0) < max_retries]
        
        if not chunk_files:
            continue
            
        print(f"\n📦 Chunk {chunk_idx//chunk_size + 1}/{(total_files-1)//chunk_size + 1}")
        
        # Create new executor for each chunk to avoid stuck workers
        with ProcessPoolExecutor(
            max_workers=actual_workers,
            initializer=init_worker
        ) as executor:
            
            # Submit tasks
            future_to_pdf = {}
            for pdf_path in chunk_files:
                future = executor.submit(process_single_paper_with_timeout, pdf_path, worker_timeout)
                future_to_pdf[future] = pdf_path
            
            # Process with timeout handling
            completed = 0
            with tqdm(total=len(chunk_files), desc=f"Chunk {chunk_idx//chunk_size + 1}") as pbar:
                
                # Use timeout for as_completed to prevent infinite waiting
                timeout_for_chunk = len(chunk_files) * (worker_timeout + 5)
                
                try:
                    for future in as_completed(future_to_pdf, timeout=timeout_for_chunk):
                        pdf_path = future_to_pdf[future]
                        
                        try:
                            # Get result with timeout
                            result = future.result(timeout=5)
                            
                            if result:
                                current_batch.append(result)
                                successful_count += 1
                                
                                # Save batch if needed
                                if len(current_batch) >= batch_size:
                                    save_batch_to_disk(current_batch, batch_num, output_dir)
                                    batch_num += 1
                                    current_batch = []
                                    gc.collect()
                            else:
                                # Result is None - likely timeout or error
                                retry_counts[pdf_path] = retry_counts.get(pdf_path, 0) + 1
                                if retry_counts[pdf_path] >= max_retries:
                                    problematic_files.add(pdf_path)
                                    failed_count += 1
                                    
                        except TimeoutError:
                            print(f"\n⏱️ Future timeout: {pdf_path.name}")
                            timeout_count += 1
                            retry_counts[pdf_path] = retry_counts.get(pdf_path, 0) + 1
                            if retry_counts[pdf_path] >= max_retries:
                                problematic_files.add(pdf_path)
                                
                        except Exception as e:
                            print(f"\n❌ Error: {pdf_path.name}: {str(e)[:50]}")
                            failed_count += 1
                            retry_counts[pdf_path] = retry_counts.get(pdf_path, 0) + 1
                            if retry_counts[pdf_path] >= max_retries:
                                problematic_files.add(pdf_path)
                        
                        finally:
                            completed += 1
                            pbar.update(1)
                            
                except TimeoutError:
                    print(f"\n⚠️ Chunk timeout - killing remaining workers")
                    # Cancel remaining futures
                    for future in future_to_pdf:
                        future.cancel()
                    
                    # Mark remaining files as problematic
                    for future, pdf_path in future_to_pdf.items():
                        if not future.done():
                            problematic_files.add(pdf_path)
                            timeout_count += 1
            
            # Force executor shutdown
            executor.shutdown(wait=False)
            
        # Cleanup between chunks
        gc.collect()
        
        # Report progress
        if chunk_idx > 0 and chunk_idx % (chunk_size * 5) == 0:
            elapsed = time.time() - start_time
            rate = successful_count / (elapsed / 60) if elapsed > 0 else 0
            print(f"\n📊 Progress: {successful_count}/{total_files} papers")
            print(f"   Rate: {rate:.1f} papers/minute")
            print(f"   Timeouts: {timeout_count}, Failures: {failed_count}")
    
    # Save remaining batch
    if current_batch:
        save_batch_to_disk(current_batch, batch_num, output_dir)
    
    # Merge batch files using PyArrow (memory-efficient)
    batch_files = sorted(output_dir.glob("batch_*.parquet"))
    
    if batch_files:
        print(f"\n📂 Merging {len(batch_files)} batch files with PyArrow...")
        tables = []

        schema = pq.ParquetFile(batch_files[0]).schema_arrow
        batches_output = DEFAULT_OUTPUT_DIR / "output.parquet"
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with pq.ParquetWriter(batches_output, schema=schema) as writer:
            for file in batch_files:
                writer.write_table(pq.read_table(file, schema=schema))
        
        # Save problematic files list
        if problematic_files:
            problem_file = output_dir / "problematic_files.txt"
            with open(problem_file, 'w') as f:
                for pf in sorted(problematic_files):
                    f.write(f"{pf}\n")
            print(f"\n⚠️ Saved {len(problematic_files)} problematic files to {problem_file}")
        
        return batches_output
    
    return None


def save_results_pyarrow(input_parquet_path, output_path: Path = None):
    table: pa.Table
    table = pq.read_table(input_parquet_path, memory_map=True)

    """Save processing results using PyArrow only."""
    if output_path is None:
        output_path = Path.cwd() / "data" / "processed_pyarrow"
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Shuffle the table
    print("🔀 Shuffling data...")
    # shuffled_table = shuffle_pyarrow_table(table, seed=42)
    shuffled_table = table

    # Calculate split sizes
    n_total = len(shuffled_table)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    # Split using PyArrow slicing (zero-copy where possible)
    train_table = shuffled_table.slice(0, n_train)
    val_table = shuffled_table.slice(n_train, n_val)
    test_table = shuffled_table.slice(n_train + n_val)
    
    # Save splits
    pq.write_table(train_table, output_path / "train.parquet")
    pq.write_table(val_table, output_path / "val.parquet")
    pq.write_table(test_table, output_path / "test.parquet")
    
    print(f"\n💾 Saved datasets to {output_path}")
    print(f"   Train: {len(train_table)} samples")
    print(f"   Val: {len(val_table)} samples")
    print(f"   Test: {len(test_table)} samples")
    
    # Memory usage stats if available
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"   Memory usage: {memory_mb:.1f} MB")


if __name__ == "__main__":
    # Use a test directory or small sample for testing
    data_dir = Path('/exdata/dev/aos-deployments/nicsar/workspaces/agent/ben.sunkur/uol/uol-final-project/paper_download/papers/CC2020')
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        sys.exit(1)
    
    # Process with PyArrow only
    output_partquet_path = process_papers_batch_parallel_pyarrow(
        data_dir, 
        limit=100000,
        n_workers=100,
        batch_size=10000,
        chunk_size=2500,
        worker_timeout=20,
        max_retries=2
    )
    
    if output_partquet_path is not None:
        save_results_pyarrow(output_partquet_path)
        print(f"\n✨ Processing complete! {len(output_partquet_path)} papers processed with PyArrow")
        
        # Show memory savings estimate
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"🎯 Final memory usage: {memory_mb:.1f} MB")