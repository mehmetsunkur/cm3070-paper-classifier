#!/usr/bin/env python3
"""
Parquet Dataset Subset Generator Class

Creates balanced subsets of parquet datasets by sampling evenly across all unique values
of a specified label column, filtered by size_rank.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd
import numpy as np


class ParquetSubsetGenerator:
    """
    Generates balanced subsets from parquet files with consistent naming for reuse.
    """
    
    def __init__(
        self,
        input_file: str,
        label_column: str = 'label_discipline',
        size_rank: str = '16-32k',
        sample_count: int = 10000,
        random_seed: int = 42,
        allow_oversampling: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the subset generator.
        
        Args:
            input_file: Path to input parquet file
            label_column: Column to use for balanced sampling
            size_rank: Size rank value to filter by
            sample_count: Total number of samples to extract
            random_seed: Random seed for reproducibility
            allow_oversampling: Allow circular sampling for underrepresented labels (default: True)
            verbose: Enable verbose logging
        """
        self.input_file = Path(input_file).resolve()
        self.label_column = label_column
        self.size_rank = size_rank
        self.sample_count = sample_count
        self.random_seed = random_seed
        self.allow_oversampling = allow_oversampling
        self.verbose = verbose
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
        
        # Validate input file exists
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file '{self.input_file}' does not exist")
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Create output directory
        self.output_dir = Path("./data/subsets")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_subset_path(self) -> str:
        """
        Returns the path to the subset file, creating it if it doesn't exist.
        
        Returns:
            Path to the subset parquet file
        """
        # Check if subset already exists
        existing_path = self._find_existing_subset()
        if existing_path:
            self.logger.info(f"Found existing subset: {existing_path}")
            return str(existing_path)
        
        # Generate new subset
        self.logger.info("Generating new subset...")
        return self._generate_subset()
    
    def _get_subset_filename(self) -> str:
        """
        Generate deterministic filename based on parameters.
        
        Returns:
            Filename for the subset
        """
        # Get base name from input file
        input_stem = self.input_file.stem
        
        # Sanitize size_rank for filename
        size_rank_safe = self.size_rank.replace('-', '_').replace(' ', '_')
        
        # Include all parameters in filename for unique identification
        os_flag = 1 if self.allow_oversampling else 0
        
        filename = (
            f"{input_stem}_{self.label_column}_{size_rank_safe}_"
            f"{self.sample_count}_s{self.random_seed}_os{os_flag}.parquet"
        )
        
        return filename
    
    def _find_existing_subset(self) -> Optional[Path]:
        """
        Check if a subset with the same parameters already exists.
        
        Returns:
            Path to existing subset file or None if not found
        """
        expected_filename = self._get_subset_filename()
        expected_path = self.output_dir / expected_filename
        
        if expected_path.exists():
            return expected_path
        
        return None
    
    def _generate_subset(self) -> str:
        """
        Generate a new subset file and return its path.
        
        Returns:
            Path to the generated subset file
        """
        # Read the parquet file
        df = pd.read_parquet(self.input_file)
        
        if self.verbose:
            self.logger.info(f"Total rows in input: {len(df)}")
            self.logger.info(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Validate and prepare balanced sampling
        label_data, unique_labels, oversampled_labels = self._validate_and_prepare_balanced_sample(df)
        
        if self.verbose:
            self._log_sampling_info(unique_labels, label_data, oversampled_labels)
        
        # Collect samples from each label
        all_samples = []
        
        for label_value in unique_labels:
            label_info = label_data[label_value]
            df_label = label_info['df']
            needed = label_info['needed']
            
            # Sample from this label
            sampled = self._create_evenly_distributed_sample(df_label, needed)
            all_samples.append(sampled)
            
            if self.verbose:
                self.logger.info(f"Sampled {len(sampled)} from {self.label_column}={label_value}")
        
        # Combine all samples
        sampled_df = pd.concat(all_samples, ignore_index=True)
        
        # Shuffle the final dataset to mix labels
        sampled_df = sampled_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        if self.verbose:
            self._log_final_statistics(sampled_df)
        
        # Save to output file
        output_filename = self._get_subset_filename()
        output_path = self.output_dir / output_filename
        
        sampled_df.to_parquet(output_path, index=False)
        
        self.logger.info(f"Successfully created balanced subset with {len(sampled_df)} samples")
        if oversampled_labels and self.allow_oversampling:
            self.logger.info(f"Note: {len(oversampled_labels)} label(s) required oversampling")
        self.logger.info(f"Output saved to: {output_path}")
        
        return str(output_path)
    
    def _validate_and_prepare_balanced_sample(
        self, 
        df: pd.DataFrame
    ) -> Tuple[Dict[Any, Dict], List[Any], List[str]]:
        """
        Validate arguments and prepare balanced sampling across all label values.
        
        Args:
            df: Input dataframe
        
        Returns:
            Tuple of (label_data dict, unique_labels list, oversampled_labels list)
        """
        # Check if label column exists
        if self.label_column not in df.columns:
            raise ValueError(
                f"Column '{self.label_column}' not found in parquet file. "
                f"Available columns: {', '.join(df.columns.tolist())}"
            )
        
        # Check if size_rank column exists
        if 'size_rank' not in df.columns:
            raise ValueError("Column 'size_rank' not found in parquet file")
        
        # Check if size_rank value exists
        if self.size_rank not in df['size_rank'].unique():
            available_ranks = df['size_rank'].unique().tolist()
            raise ValueError(
                f"Size rank '{self.size_rank}' not found. "
                f"Available ranks: {', '.join(map(str, available_ranks))}"
            )
        
        # Filter by size_rank first
        df_filtered = df[df['size_rank'] == self.size_rank]
        
        # Get unique label values
        unique_labels = sorted(df_filtered[self.label_column].unique())
        num_labels = len(unique_labels)
        
        if num_labels == 0:
            raise ValueError(f"No data found for size_rank='{self.size_rank}'")
        
        # Calculate samples per label
        samples_per_label = self.sample_count // num_labels
        remainder = self.sample_count % num_labels
        
        if samples_per_label == 0:
            raise ValueError(
                f"Sample count ({self.sample_count}) is less than number of unique "
                f"{self.label_column} values ({num_labels}). "
                f"Need at least {num_labels} samples for balanced dataset."
            )
        
        # Check availability for each label
        label_data = {}
        insufficient_labels = []
        oversampled_labels = []
        
        for i, label_value in enumerate(unique_labels):
            label_df = df_filtered[df_filtered[self.label_column] == label_value]
            available = len(label_df)
            # Add 1 extra sample to first 'remainder' labels to match exact sample_count
            needed = samples_per_label + (1 if i < remainder else 0)
            
            if available < needed:
                if self.allow_oversampling:
                    oversampled_labels.append(
                        f"{self.label_column}={label_value}: need {needed}, "
                        f"have {available} (will oversample)"
                    )
                    label_data[label_value] = {
                        'df': label_df,
                        'available': available,
                        'needed': needed,
                        'requires_oversampling': True
                    }
                else:
                    insufficient_labels.append(
                        f"{self.label_column}={label_value}: need {needed}, have {available}"
                    )
            else:
                label_data[label_value] = {
                    'df': label_df,
                    'available': available,
                    'needed': needed,
                    'requires_oversampling': False
                }
        
        if insufficient_labels and not self.allow_oversampling:
            raise ValueError(
                f"Not enough samples for balanced dataset:\n" + 
                "\n".join(insufficient_labels) + 
                "\n\nSet allow_oversampling=True to enable circular sampling for underrepresented labels."
            )
        
        return label_data, unique_labels, oversampled_labels
    
    def _create_evenly_distributed_sample(
        self, 
        df: pd.DataFrame, 
        sample_count: int
    ) -> pd.DataFrame:
        """
        Create an evenly distributed sample from the dataframe.
        
        Args:
            df: Input dataframe
            sample_count: Number of samples to extract
        
        Returns:
            Sampled dataframe
        """
        total_rows = len(df)
        
        if sample_count <= total_rows:
            # Normal sampling when we have enough data
            if sample_count == total_rows:
                return df
            
            # Calculate step size for systematic sampling
            step = total_rows / sample_count
            
            # Generate evenly spaced indices
            indices = np.arange(0, total_rows, step).astype(int)[:sample_count]
            
            # Reset index to ensure proper indexing
            df_reset = df.reset_index(drop=True)
            
            # Select rows at calculated indices
            sampled_df = df_reset.iloc[indices]
            
            return sampled_df
        
        elif self.allow_oversampling:
            # Circular sampling when we need more samples than available
            df_reset = df.reset_index(drop=True)
            
            # Calculate how many complete passes and remaining samples we need
            complete_passes = sample_count // total_rows
            remaining_samples = sample_count % total_rows
            
            # Collect samples
            samples = []
            
            # Add complete passes
            for _ in range(complete_passes):
                samples.append(df_reset)
            
            # Add remaining samples using systematic sampling from the beginning
            if remaining_samples > 0:
                step = total_rows / remaining_samples
                indices = np.arange(0, total_rows, step).astype(int)[:remaining_samples]
                samples.append(df_reset.iloc[indices])
            
            # Combine all samples
            sampled_df = pd.concat(samples, ignore_index=True)
            
            return sampled_df
        
        else:
            raise ValueError(
                f"Cannot sample {sample_count} items from {total_rows} available without oversampling"
            )
    
    def _log_sampling_info(
        self, 
        unique_labels: List[Any], 
        label_data: Dict[Any, Dict],
        oversampled_labels: List[str]
    ) -> None:
        """Log detailed sampling information."""
        self.logger.info("\nBalanced sampling setup:")
        self.logger.info(f"  Label column: {self.label_column}")
        self.logger.info(f"  Unique values: {unique_labels}")
        self.logger.info(f"  Total samples requested: {self.sample_count}")
        self.logger.info(f"  Samples per label: ~{self.sample_count // len(unique_labels)}")
        self.logger.info(f"  Oversampling enabled: {self.allow_oversampling}")
        self.logger.info("\nAvailable samples per label:")
        
        for label, data in label_data.items():
            status = " (OVERSAMPLED)" if data.get('requires_oversampling', False) else ""
            self.logger.info(
                f"  {self.label_column}={label}: {data['available']} available, "
                f"{data['needed']} needed{status}"
            )
        
        if oversampled_labels:
            self.logger.info("\nWarning: The following labels will be oversampled (circular sampling):")
            for msg in oversampled_labels:
                self.logger.info(f"  - {msg}")
    
    def _log_final_statistics(self, sampled_df: pd.DataFrame) -> None:
        """Log final dataset statistics."""
        self.logger.info("\nFinal dataset statistics:")
        self.logger.info(f"  Total samples: {len(sampled_df)}")
        
        # Show distribution statistics
        self.logger.info(f"\nDistribution by {self.label_column}:")
        value_counts = sampled_df[self.label_column].value_counts().sort_index()
        for label, count in value_counts.items():
            self.logger.info(
                f"  {self.label_column}={label}: {count} samples "
                f"({count/len(sampled_df)*100:.1f}%)"
            )
        
        self.logger.info("\nSize rank distribution:")
        self.logger.info(f"  {sampled_df['size_rank'].value_counts().to_dict()}")
        
        if 'word_count' in sampled_df.columns:
            self.logger.info("\nWord count statistics:")
            self.logger.info(f"  Range: {sampled_df['word_count'].min()} - {sampled_df['word_count'].max()}")
            self.logger.info(f"  Mean: {sampled_df['word_count'].mean():.1f}")
            self.logger.info(f"  Median: {sampled_df['word_count'].median():.1f}")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ParquetSubsetGenerator':
        """
        Create a ParquetSubsetGenerator instance from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with subset generation parameters
            
        Returns:
            ParquetSubsetGenerator instance
        """
        # Extract parameters from config with proper key names
        return cls(
            input_file=config['source_file'],
            label_column=config.get('label_column', 'label_discipline'),
            size_rank=config['size_rank'],
            sample_count=config.get('sample_count', 10000),
            random_seed=config.get('random_seed', 42),
            allow_oversampling=config.get('allow_oversampling', True),
            verbose=config.get('verbose', False)
        )


def main():
    """Example usage of the ParquetSubsetGenerator class."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate balanced subset of a parquet dataset using the ParquetSubsetGenerator class'
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input parquet file'
    )
    
    parser.add_argument(
        '--label-column',
        type=str,
        default='label_discipline',
        choices=['label_discipline', 'label_field', 'label_method'],
        help='Label column to use for balanced sampling'
    )
    
    parser.add_argument(
        '--size-rank',
        type=str,
        default='16-32k',
        help='Size rank value to filter by (e.g., "16-32k")'
    )
    
    parser.add_argument(
        '--sample-count',
        type=int,
        default=10000,
        help='Total number of samples to extract'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--no-oversampling',
        action='store_false',
        dest='allow_oversampling',
        help='Disable circular oversampling for labels with insufficient samples (default: enabled)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(levelname)s: %(message)s'
    )
    
    try:
        # Create generator instance
        generator = ParquetSubsetGenerator(
            input_file=args.input_file,
            label_column=args.label_column,
            size_rank=args.size_rank,
            sample_count=args.sample_count,
            random_seed=args.random_seed,
            allow_oversampling=args.allow_oversampling,
            verbose=args.verbose
        )
        
        # Get subset path (creates if doesn't exist)
        subset_path = generator.get_subset_path()
        print(f"Subset file: {subset_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())