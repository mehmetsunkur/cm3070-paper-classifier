#!/usr/bin/env python3
"""
Main entry point for the paper classifier training pipeline.
Refactored from trainer_config.py for clean package structure.
"""

import sys
import os
from pathlib import Path
import argparse
from dotenv import load_dotenv

from .config.loader import load_config
from .subset_generator import ParquetSubsetGenerator
from .pipeline import run_pipeline


def generate_subset_if_needed(config):
    """
    Generate subset using ParquetSubsetGenerator if enabled and no direct parquet_path is provided.
    
    Args:
        config: Flat configuration dictionary
        
    Returns:
        str: Path to the parquet file to use (either generated subset or direct path)
    """
    # Check if direct parquet path is provided
    if config.get('data_parquet_path'):
        print(f"Using direct parquet path: {config['data_parquet_path']}")
        return config['data_parquet_path']
    
    # Check if subset generation is enabled
    if not config.get('subset_generation_enabled', False):
        raise ValueError("No parquet_path provided and subset generation is disabled")
    
    # Extract subset generation parameters
    source_file = config.get('subset_generation_source_file')
    if not source_file:
        raise ValueError("subset_generation.source_file is required when subset generation is enabled")
    
    size_rank = config.get('subset_generation_size_rank')
    if not size_rank:
        raise ValueError("subset_generation.size_rank is required for subset generation")
    
    # Get other parameters with defaults
    label_column = config.get('subset_generation_label_column', 'label_discipline')
    sample_count = config.get('subset_generation_sample_count', 10000)
    random_seed = config.get('subset_generation_random_seed', 42)
    allow_oversampling = config.get('subset_generation_allow_oversampling', True)
    verbose = config.get('subset_generation_verbose', False)
    
    print(f"\nGenerating subset from {source_file}")
    print(f"  Size rank: {size_rank}")
    print(f"  Label column: {label_column}")
    print(f"  Sample count: {sample_count}")
    print(f"  Random seed: {random_seed}")
    
    # Create generator instance
    generator = ParquetSubsetGenerator(
        input_file=source_file,
        label_column=label_column,
        size_rank=size_rank,
        sample_count=sample_count,
        random_seed=random_seed,
        allow_oversampling=allow_oversampling,
        verbose=verbose
    )
    
    # Get or generate subset
    subset_path = generator.get_subset_path()
    print(f"Using subset: {subset_path}\n")
    
    return subset_path


def main():
    """Main entry point that loads config and runs the component-based pipeline"""
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Paper classifier training pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    
    # Parse known args to get config, allow other args to pass through
    args, remaining_args = parser.parse_known_args()
    
    # Load the config with all CLI args (including overrides)
    config = load_config(args.config, cli_args=sys.argv[1:])
    
    # Generate subset if needed and update config
    try:
        parquet_path = generate_subset_if_needed(config)
        config['data_parquet_path'] = parquet_path
    except Exception as e:
        print(f"Error during subset generation: {e}")
        sys.exit(1)
    
    # Extract config name without extension and path
    config_name = Path(args.config).stem  # e.g., "model_label_discipline_1000_1_4k"
    
    print("\nUsing component-based training pipeline")
    
    try:
        # Run the component-based pipeline
        result = run_pipeline(
            config=config,
            parquet_path=parquet_path,
            model_name=config_name
        )
        
        if result['success']:
            print("\nTraining completed successfully!")
            print(f"Model saved to: {result['model_path']}")
            print(f"Test results: {result['test_results']}")
            sys.exit(0)
        else:
            print(f"\nTraining failed: {result.get('message', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nPipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()