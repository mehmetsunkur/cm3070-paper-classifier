#!/usr/bin/env python3
"""Extract size_rank value from parquet file using DuckDB SQL."""

import duckdb
from pydantic import BaseModel

class DatasetInfo(BaseModel):
    """Model for dataset information."""
    record_count: int
    highest_size_rank: str
    max_word_count: int
    parquet_path: str
    dataset_slug: str

def get_largest_text(parquet_path: str) -> str:
    """
    Get the text with the maximum word_count from a parquet file.
    
    Args:
        parquet_path: Path to the parquet file
        
    Returns:
        Text string with the highest word_count
    """
    conn = duckdb.connect()
    
    try:
        result = conn.execute(f"""
            SELECT text 
            FROM read_parquet('{parquet_path}') 
            ORDER BY word_count DESC 
            LIMIT 1
        """).fetchone()
        
        if not result or not result[0]:
            raise ValueError(f"No text found in parquet file: {parquet_path}")
            
        return result[0]
            
    finally:
        conn.close()

def get_dataset_info(parquet_path: str) -> DatasetInfo:
    """
    Get the number of records and the size_rank for the highest text_size_kb.
    
    Args:
        parquet_path: Path to the parquet file
        
    Returns:
        DatasetInfo model with record_count and highest_size_rank
    """
    conn = duckdb.connect()
    
    try:
        result = conn.execute(f"""
            SELECT 
                COUNT(*) as record_count,
                (SELECT size_rank FROM read_parquet('{parquet_path}') ORDER BY text_size_kb DESC LIMIT 1) as highest_size_rank,
                MAX(word_count) as max_word_count
            FROM read_parquet('{parquet_path}')
        """).fetchone()
        
        if not result or result[0] == 0:
            raise ValueError(f"No records found in parquet file: {parquet_path}")
        
        if not result[1]:
            raise ValueError(f"No size_rank found in parquet file: {parquet_path}")
            
        dataset_slug = f"{result[0]}_{result[1]}"
        return DatasetInfo(
            record_count=result[0], 
            highest_size_rank=result[1],
            max_word_count=result[2],
            parquet_path=parquet_path,
            dataset_slug=dataset_slug
        )
            
    finally:
        conn.close()


if __name__ == "__main__":
    # Example usage
    default_path = "/exdata/dev/aos-deployments/nicsar/workspaces/agent/ben.sunkur/uol/uol-final-project/mamba-paper-classifier/data/subsets/output_label_discipline_1_4k_100_1.parquet"
    
    # Simple function to get record count and highest size_rank
    dataset_info = get_dataset_info(default_path)
    print(f"\n=== Dataset Info ===")
    print(f"Parquet path: {dataset_info.parquet_path}")
    print(f"Number of records: {dataset_info.record_count}")
    print(f"Highest size rank: {dataset_info.highest_size_rank}")
    print(f"Max word count: {dataset_info.max_word_count}")
    print(f"Dataset slug: {dataset_info.dataset_slug}")
    
    # Get the largest text
    print(f"\n=== Largest Text Sample ===")
    largest_text = get_largest_text(default_path)
    print(f"Text length: {len(largest_text)} characters")
    print(f"First 500 chars: {largest_text[:500]}...")