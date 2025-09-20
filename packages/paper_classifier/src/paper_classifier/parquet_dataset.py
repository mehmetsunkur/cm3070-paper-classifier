import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
import random
from .data_set_utils import get_dataset_info, get_largest_text

class ParquetDataset:
    def __init__(self, parquet_path, tokenizer, train_ratio=0.7, val_ratio=0.15, random_seed=42, label_column='label_discipline'):
        """
        Initialize the ParquetDataset for multi-class classification.
        
        Args:
            parquet_path: Path to the parquet file
            tokenizer: Tokenizer to use for text encoding
            train_ratio: Ratio of data for training (default 0.7)
            val_ratio: Ratio of data for validation (default 0.15)
            random_seed: Random seed for reproducibility
            label_column: Name of the label column in the parquet file (default 'label_discipline')
        """
        self.tokenizer = tokenizer
        self.random_seed = random_seed
        self.label_column = label_column
        
        # Label mapping attributes
        self.label2id = None  # Maps original labels to 0-indexed IDs
        self.id2label = None  # Maps 0-indexed IDs back to original labels
        self.original_num_classes = None  # Number of unique labels before remapping
        self.effective_num_classes = None  # Number of classes after remapping (same as original)
        self.needs_remapping = False  # Whether labels need to be remapped
        
        # Load dataset info using data_set_utils
        self.dataset_info = get_dataset_info(parquet_path)
        print(f"\nDataset Info:")
        print(f"  Path: {self.dataset_info.parquet_path}")
        print(f"  Records: {self.dataset_info.record_count}")
        print(f"  Highest size rank: {self.dataset_info.highest_size_rank}")
        print(f"  Max word count: {self.dataset_info.max_word_count}")
        print(f"  Dataset slug: {self.dataset_info.dataset_slug}")
        
        # Get the largest text and tokenize it to determine actual max token length
        largest_text = get_largest_text(parquet_path)
        tokens = tokenizer(largest_text, truncation=False, padding=False, return_attention_mask=False)
        self.max_token_length = len(tokens['input_ids'])
        print(f"  Max token length (from largest text): {self.max_token_length}")
        
        # Load the parquet file
        self.df = pd.read_parquet(parquet_path)
        
        # Ensure we have the required columns
        assert 'text' in self.df.columns, "Parquet file must have 'text' column"
        assert self.label_column in self.df.columns, f"Parquet file must have '{self.label_column}' column"
        
        # Set up train/val/test split ratios
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        
        # Perform the split
        self._split_data()
        
        # Check if labels need remapping and apply if necessary
        self._check_and_apply_label_remapping()
    
    def _split_data(self):
        """Split data into train, validation, and test sets with stratification."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # First split: separate out test set
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=self.test_ratio,
            stratify=self.df[self.label_column],
            random_state=self.random_seed
        )
        
        # Second split: separate train and validation from train_val
        val_size_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            stratify=train_val_df[self.label_column],
            random_state=self.random_seed
        )
        
        # Convert to HuggingFace Dataset format
        self.train_dataset = Dataset.from_pandas(train_df[['text', self.label_column]])
        self.val_dataset = Dataset.from_pandas(val_df[['text', self.label_column]])
        self.test_dataset = Dataset.from_pandas(test_df[['text', self.label_column]])
        
        # Rename label column to 'label' for compatibility
        self.train_dataset = self.train_dataset.rename_column(self.label_column, 'label')
        self.val_dataset = self.val_dataset.rename_column(self.label_column, 'label')
        self.test_dataset = self.test_dataset.rename_column(self.label_column, 'label')
        
        print(f"Dataset splits created:")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Validation: {len(self.val_dataset)} samples")
        print(f"  Test: {len(self.test_dataset)} samples")
        
        # Print label distribution
        train_labels = train_df[self.label_column].value_counts().sort_index()
        print(f"\nTrain label distribution:")
        for label, count in train_labels.items():
            print(f"  Class {label}: {count} samples ({count/len(train_df)*100:.1f}%)")
    
    def preprocess_function(self, examples):
        """Preprocess function for tokenization with memory-efficient settings."""
        # Use actual max token length calculated from largest text
        samples = self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=self.max_token_length,  # Use actual max token length
            padding=False,  # Dynamic padding will be handled by the data collator
            return_attention_mask=False  # Mamba doesn't use attention masks
        )
        # Explicitly remove attention_mask if it exists (double safety)
        samples.pop('attention_mask', None)
        # Add the labels to the tokenized samples
        samples['labels'] = examples['label']
        return samples
    
    def return_train_dataset(self):
        """Return the preprocessed training dataset."""
        return self.train_dataset.map(
            self.preprocess_function, 
            batched=True,
            remove_columns=['text']  # Remove raw text after tokenization
        )
    
    def return_val_dataset(self):
        """Return the preprocessed validation dataset."""
        return self.val_dataset.map(
            self.preprocess_function, 
            batched=True,
            remove_columns=['text']  # Remove raw text after tokenization
        )
    
    def return_test_dataset(self):
        """Return the preprocessed test dataset."""
        return self.test_dataset.map(
            self.preprocess_function, 
            batched=True,
            remove_columns=['text']  # Remove raw text after tokenization
        )
    
    def get_label_distribution(self):
        """Get the label distribution for all splits."""
        distributions = {}
        for split_name, dataset in [('train', self.train_dataset), 
                                    ('val', self.val_dataset), 
                                    ('test', self.test_dataset)]:
            labels = dataset['label']
            unique, counts = np.unique(labels, return_counts=True)
            distributions[split_name] = dict(zip(unique, counts))
        return distributions
    
    def get_max_token_length(self):
        """Get the maximum token length calculated from the largest text in the dataset."""
        return self.max_token_length
    
    def _check_and_apply_label_remapping(self):
        """Check if labels need remapping and apply if necessary."""
        # Get all unique labels across all splits
        all_labels = np.concatenate([
            self.train_dataset['label'],
            self.val_dataset['label'],
            self.test_dataset['label']
        ])
        unique_labels = np.unique(all_labels)
        unique_labels = sorted([int(l) for l in unique_labels])
        
        # Store original number of classes
        self.original_num_classes = len(unique_labels)
        self.effective_num_classes = len(unique_labels)
        
        # Check if labels are continuous starting from 0
        expected_labels = list(range(len(unique_labels)))
        self.needs_remapping = unique_labels != expected_labels
        
        if self.needs_remapping:
            print(f"\nLabel remapping required:")
            print(f"  Original labels: {unique_labels}")
            print(f"  Will be mapped to: 0-{len(unique_labels)-1}")
            
            # Create bidirectional mappings
            self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
            self.id2label = {idx: label for label, idx in self.label2id.items()}
            
            # Apply remapping to all datasets
            self.train_dataset = self._remap_labels(self.train_dataset)
            self.val_dataset = self._remap_labels(self.val_dataset)
            self.test_dataset = self._remap_labels(self.test_dataset)
            
            # Verify remapping
            remapped_labels = np.unique(np.concatenate([
                self.train_dataset['label'],
                self.val_dataset['label'],
                self.test_dataset['label']
            ]))
            print(f"  Labels after remapping: {sorted(remapped_labels.tolist())}")
        else:
            print(f"\nNo label remapping needed - labels are already continuous 0-{len(unique_labels)-1}")
            # Create identity mapping for consistency
            self.label2id = {label: label for label in unique_labels}
            self.id2label = {label: label for label in unique_labels}
    
    def _remap_labels(self, dataset):
        """Apply label remapping to a dataset."""
        def remap_label(example):
            example['label'] = self.label2id[example['label']]
            return example
        
        return dataset.map(remap_label, desc="Remapping labels")
    
    def get_num_classes(self):
        """Get the number of unique classes in the dataset."""
        if self.effective_num_classes is not None:
            return self.effective_num_classes
        
        # Fallback to detecting from data if not set
        labels = self.train_dataset['label']
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        print(f"Detected {num_classes} unique classes: {list(unique_labels)}")
        return num_classes
    
    def get_label_mappings(self):
        """Get the label mappings for this dataset."""
        return {
            'label2id': self.label2id,
            'id2label': self.id2label,
            'needs_remapping': self.needs_remapping,
            'original_num_classes': self.original_num_classes,
            'effective_num_classes': self.effective_num_classes
        }
    
    def get_label_column_name(self):
        """Get the label column name used in the dataset."""
        # After renaming, it's always 'label' in the Dataset objects
        return 'label'