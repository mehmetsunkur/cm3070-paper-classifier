"""
Validation module for checking label ranges and mappings
"""

import numpy as np
import os
import json
from typing import Dict, Any, List, Optional


class LabelValidator:
    """Validate label ranges and mappings"""
    
    @staticmethod
    def validate_labels_before_training(dataset: Any, num_classes: int) -> bool:
        """
        Validate all labels are in valid range before training
        
        Args:
            dataset: Dataset object with train/val/test splits
            num_classes: Expected number of classes
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If labels are invalid
        """
        print("\n" + "=" * 60)
        print("LABEL VALIDATION")
        print("=" * 60)
        
        # Check each split
        for split_name, split_data in [
            ('train', dataset.train_dataset),
            ('val', dataset.val_dataset),
            ('test', dataset.test_dataset)
        ]:
            labels = np.array(split_data['label'])
            
            # Check for negative labels
            if labels.min() < 0:
                raise ValueError(
                    f"{split_name} split contains negative labels: min={labels.min()}"
                )
            
            # Check if any label exceeds num_classes
            if labels.max() >= num_classes:
                raise ValueError(
                    f"{split_name} split label {labels.max()} >= num_classes {num_classes}\n"
                    f"Labels must be in range [0, {num_classes-1}]"
                )
            
            # Check for NaN or inf
            if np.any(np.isnan(labels)) or np.any(np.isinf(labels)):
                raise ValueError(f"{split_name} split contains NaN or inf labels")
            
            # Check data type
            if not np.issubdtype(labels.dtype, np.integer):
                print(f"⚠️  Warning: {split_name} split labels are not integers, will be converted")
                labels = labels.astype(int)
            
            print(f"✓ {split_name} split: {len(labels)} samples, labels in range [{labels.min()}, {labels.max()}]")
        
        # Check continuity across all splits
        all_labels = np.concatenate([
            dataset.train_dataset['label'],
            dataset.val_dataset['label'],
            dataset.test_dataset['label']
        ])
        unique_labels = np.unique(all_labels)
        
        print(f"\nLabel Statistics:")
        print(f"  Total unique labels: {len(unique_labels)}")
        print(f"  Model num_classes: {num_classes}")
        print(f"  Label range: [{unique_labels.min()}, {unique_labels.max()}]")
        
        if len(unique_labels) != num_classes:
            print(f"⚠️  Warning: Using {len(unique_labels)} of {num_classes} possible classes")
            print(f"   Active classes: {sorted(unique_labels.tolist())}")
            missing_classes = set(range(num_classes)) - set(unique_labels)
            if missing_classes:
                print(f"   Missing classes: {sorted(missing_classes)}")
        else:
            print(f"✓ All {num_classes} classes are present in the dataset")
        
        # Check if labels are continuous
        expected_continuous = list(range(len(unique_labels)))
        if not np.array_equal(sorted(unique_labels), expected_continuous):
            print(f"ℹ️  Labels have been remapped from non-continuous values")
        
        print("=" * 60)
        print("✓ VALIDATION PASSED")
        print("=" * 60 + "\n")
        
        return True
    
    @staticmethod
    def validate_mapping_consistency(label_mapper: Any, dataset: Any) -> bool:
        """
        Validate mapping consistency across splits
        
        Args:
            label_mapper: LabelMapper instance with mappings
            dataset: ParquetDataset instance
            
        Returns:
            True if mappings are consistent
            
        Raises:
            ValueError: If mappings are inconsistent
        """
        if not label_mapper.mappings:
            print("No mappings to validate")
            return True
        
        print("\n" + "=" * 60)
        print("MAPPING CONSISTENCY CHECK")
        print("=" * 60)
        
        for column, mapping in label_mapper.mappings.items():
            print(f"\nChecking mappings for '{column}':")
            
            # Verify bidirectional mapping consistency
            for orig_label, mapped_id in mapping['label2id'].items():
                reverse_mapped = mapping['id2label'].get(mapped_id)
                if reverse_mapped != orig_label:
                    raise ValueError(
                        f"Inconsistent mapping for {column}: "
                        f"{orig_label} -> {mapped_id} -> {reverse_mapped}"
                    )
            
            print(f"  ✓ Bidirectional mappings are consistent")
            
            # Check that all mapped IDs are in expected range
            mapped_ids = list(mapping['id2label'].keys())
            expected_ids = list(range(len(mapped_ids)))
            if sorted(mapped_ids) != expected_ids:
                raise ValueError(
                    f"Mapped IDs are not continuous 0-indexed: {sorted(mapped_ids)}"
                )
            
            print(f"  ✓ Mapped IDs are continuous 0-{len(mapped_ids)-1}")
            
            # Verify number of classes matches
            if mapping['num_classes'] != len(mapping['id2label']):
                raise ValueError(
                    f"num_classes ({mapping['num_classes']}) doesn't match "
                    f"number of mappings ({len(mapping['id2label'])})"
                )
            
            print(f"  ✓ Number of classes consistent: {mapping['num_classes']}")
        
        print("\n✓ All mappings are consistent")
        print("=" * 60 + "\n")
        
        return True
    
    @staticmethod
    def validate_checkpoint_compatibility(checkpoint_path: str) -> Dict:
        """
        Check if checkpoint has proper label mappings
        
        Args:
            checkpoint_path: Path to model checkpoint directory
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'has_mappings': False,
            'mapping_path': None,
            'label_columns': [],
            'warnings': [],
            'info': []
        }
        
        print("\n" + "=" * 60)
        print("CHECKPOINT COMPATIBILITY CHECK")
        print(f"Checkpoint: {checkpoint_path}")
        print("=" * 60)
        
        if not os.path.exists(checkpoint_path):
            results['warnings'].append(f"Checkpoint path does not exist: {checkpoint_path}")
            print(f"✗ Checkpoint path does not exist")
            return results
        
        # Check for label mappings file
        mapping_path = os.path.join(checkpoint_path, 'label_mappings.json')
        
        if os.path.exists(mapping_path):
            results['has_mappings'] = True
            results['mapping_path'] = mapping_path
            
            try:
                with open(mapping_path, 'r') as f:
                    mappings = json.load(f)
                    results['label_columns'] = list(mappings.keys())
                    
                print(f"✓ Found label mappings for: {', '.join(results['label_columns'])}")
                
                # Display mapping details
                for column, mapping in mappings.items():
                    num_classes = mapping.get('num_classes', 'unknown')
                    original_labels = mapping.get('original_labels', [])
                    print(f"\n  {column}:")
                    print(f"    - Number of classes: {num_classes}")
                    if len(original_labels) <= 10:
                        print(f"    - Original labels: {original_labels}")
                    else:
                        print(f"    - Original labels: {original_labels[:5]} ... {original_labels[-5:]}")
                    
                    results['info'].append(f"{column}: {num_classes} classes")
                    
            except Exception as e:
                results['warnings'].append(f"Error reading label mappings: {str(e)}")
                print(f"⚠️  Error reading label mappings: {str(e)}")
        else:
            results['warnings'].append(
                "No label_mappings.json found - will use identity mapping"
            )
            print("⚠️  No label_mappings.json found")
            print("   Model will use identity mapping (labels must be 0-indexed continuous)")
        
        # Check for model config
        config_path = os.path.join(checkpoint_path, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    num_classes = config.get('num_classes', 'not specified')
                    print(f"\n✓ Model config found: {num_classes} classes")
                    results['info'].append(f"Model configured for {num_classes} classes")
            except Exception as e:
                print(f"⚠️  Error reading model config: {str(e)}")
        
        print("=" * 60 + "\n")
        
        return results
    
    @staticmethod
    def validate_labels_for_inference(labels: np.ndarray, id2label: Dict[int, int]) -> bool:
        """
        Validate that predicted labels can be mapped back to original values
        
        Args:
            labels: Array of predicted label indices
            id2label: Mapping from model outputs to original labels
            
        Returns:
            True if all labels can be mapped
            
        Raises:
            ValueError: If any label cannot be mapped
        """
        unmappable = []
        for label in np.unique(labels):
            if int(label) not in id2label:
                unmappable.append(int(label))
        
        if unmappable:
            raise ValueError(
                f"Cannot map predicted labels to original values: {unmappable}\n"
                f"Available mappings: {list(id2label.keys())}"
            )
        
        return True
    
    @staticmethod
    def print_label_summary(dataset: Any, label_column: str = None):
        """
        Print a summary of label distributions
        
        Args:
            dataset: ParquetDataset instance
            label_column: Name of the label column
        """
        print("\n" + "=" * 60)
        print("LABEL DISTRIBUTION SUMMARY")
        if label_column:
            print(f"Label Column: {label_column}")
        print("=" * 60)
        
        # Get label distributions for each split
        for split_name, split_data in [
            ('Train', dataset.train_dataset),
            ('Val', dataset.val_dataset),
            ('Test', dataset.test_dataset)
        ]:
            labels = np.array(split_data['label'])
            unique, counts = np.unique(labels, return_counts=True)
            
            print(f"\n{split_name} Set ({len(labels)} samples):")
            print("-" * 40)
            
            # Sort by label value
            sorted_indices = np.argsort(unique)
            unique = unique[sorted_indices]
            counts = counts[sorted_indices]
            
            for label, count in zip(unique, counts):
                percentage = (count / len(labels)) * 100
                bar_length = int(percentage / 2)  # Scale to fit
                bar = '█' * bar_length
                print(f"  Class {label:2d}: {count:5d} ({percentage:5.1f}%) {bar}")
        
        print("=" * 60 + "\n")