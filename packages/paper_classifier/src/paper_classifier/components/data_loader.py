"""
DataLoaderComponent: Handles dataset loading and preprocessing
"""

import os
from typing import Dict, Tuple, Optional
from transformers import AutoTokenizer, DataCollatorWithPadding
from ..parquet_dataset import ParquetDataset
from ..data_set_utils import get_dataset_info


class DataLoaderComponent:
    def __init__(self, config: Dict):
        """
        Initialize DataLoaderComponent with configuration
        
        Args:
            config: Flat configuration dictionary containing data settings
        """
        self.config = config
        self.tokenizer = None
        self.parquet_dataset = None
        self.max_seq_length = None
        self.data_collator = None
        
    def initialize_tokenizer(self) -> AutoTokenizer:
        """Initialize and configure the tokenizer"""
        tokenizer_model = self.config.get('tokenizer_model_name', 'EleutherAI/gpt-neox-20b')
        print(f"\nLoading tokenizer: {tokenizer_model}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        return self.tokenizer
    
    def load_dataset(self, parquet_path: str) -> ParquetDataset:
        """
        Load the parquet dataset
        
        Args:
            parquet_path: Path to the parquet file
            
        Returns:
            ParquetDataset instance
        """
        print(f"\nLoading dataset from: {parquet_path}")
        
        train_ratio = self.config.get('data_train_ratio', 0.7)
        val_ratio = self.config.get('data_val_ratio', 0.15)
        label_column = self.config.get('subset_generation_label_column', 'label_discipline')
        
        self.parquet_dataset = ParquetDataset(
            parquet_path,
            self.tokenizer,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            label_column=label_column
        )
        
        # Determine max sequence length
        if self.config.get('data_max_seq_length') and self.config['data_max_seq_length'] != 'auto':
            self.max_seq_length = self.config['data_max_seq_length']
            print(f"Using configured max sequence length: {self.max_seq_length}")
        else:
            self.max_seq_length = self.parquet_dataset.get_max_token_length()
            print(f"Auto-detected max sequence length: {self.max_seq_length}")
        
        return self.parquet_dataset
    
    def get_datasets(self) -> Tuple:
        """
        Get train, eval, and test datasets
        
        Returns:
            Tuple of (train_dataset, eval_dataset, test_dataset)
        """
        if not self.parquet_dataset:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        train_dataset = self.parquet_dataset.return_train_dataset()
        eval_dataset = self.parquet_dataset.return_val_dataset()
        test_dataset = self.parquet_dataset.return_test_dataset()
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_dataset):,}")
        print(f"  Eval: {len(eval_dataset):,}")
        print(f"  Test: {len(test_dataset):,}")
        
        return train_dataset, eval_dataset, test_dataset
    
    def create_data_collator(self) -> DataCollatorWithPadding:
        """
        Create data collator for batching
        
        Returns:
            DataCollatorWithPadding instance
        """
        if not self.tokenizer or not self.max_seq_length:
            raise ValueError("Tokenizer and dataset must be initialized first")
        
        use_dynamic_padding = self.config.get('training_use_dynamic_padding', False)
        
        if use_dynamic_padding:
            # Dynamic padding (may cause GPU fluctuations)
            self.data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding='longest',
                max_length=self.max_seq_length,
                pad_to_multiple_of=8,
                return_tensors='pt'
            )
            print("Using dynamic padding (may cause GPU fluctuations)")
        else:
            # Fixed padding for stable GPU usage
            self.data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding='max_length',
                max_length=self.max_seq_length,
                pad_to_multiple_of=None,
                return_tensors='pt'
            )
            print("Using fixed padding for stable GPU usage")
        
        return self.data_collator
    
    def get_dataset_info(self, parquet_path: str) -> Dict:
        """
        Get dataset information for logging and output directory creation
        
        Args:
            parquet_path: Path to the parquet file
            
        Returns:
            Dictionary with dataset information
        """
        dataset_info = get_dataset_info(parquet_path=parquet_path)
        parquet_name = os.path.basename(parquet_path).replace('.parquet', '').replace('_', '-')
        
        return {
            'dataset_info': dataset_info,
            'parquet_name': parquet_name,
            'dataset_slug': dataset_info.dataset_slug
        }
    
    def get_num_classes(self) -> int:
        """
        Get the number of unique classes from the dataset
        
        Returns:
            Number of unique classes in the dataset
        """
        if not self.parquet_dataset:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        return self.parquet_dataset.get_num_classes()