"""
LabelMapper: Manages label mappings and integration with label_config.yaml
"""

import json
import yaml
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class LabelMapper:
    """Manages label mappings and integration with label_config.yaml"""
    
    def __init__(self, label_config_path: str = 'config/label/label_config.yaml'):
        """
        Initialize LabelMapper with optional label configuration
        
        Args:
            label_config_path: Path to label configuration file
        """
        self.label_config = None
        self.mappings = {}
        
        # Load label config if it exists
        if os.path.exists(label_config_path):
            self.label_config = self._load_config(label_config_path)
        else:
            print(f"Warning: Label config not found at {label_config_path}")
    
    def _load_config(self, path: str) -> Dict:
        """Load label configuration file"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def create_mapping(self, label_column: str, unique_labels: List[int]) -> Dict:
        """
        Create and store mapping for a label column
        
        Args:
            label_column: Name of the label column (e.g., 'label_field')
            unique_labels: List of unique label values in the dataset
            
        Returns:
            Dictionary containing the mapping information
        """
        # Ensure labels are sorted for consistent mapping
        unique_labels = sorted([int(l) for l in unique_labels])
        
        # Create bidirectional mappings
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        
        # Extract task name from label_column (e.g., 'label_field' -> 'field')
        task_name = label_column.replace('label_', '')
        
        mapping = {
            'label2id': label2id,
            'id2label': id2label,
            'label_names': self._get_label_names(task_name, unique_labels),
            'num_classes': len(unique_labels),
            'original_labels': unique_labels,
            'label_column': label_column
        }
        
        self.mappings[label_column] = mapping
        
        # Log mapping information
        print(f"\nLabel mapping created for '{label_column}':")
        print(f"  - Number of classes: {mapping['num_classes']}")
        print(f"  - Original labels: {mapping['original_labels']}")
        print(f"  - Mapped to: 0-{mapping['num_classes']-1}")
        
        return mapping
    
    def _get_label_names(self, task_name: str, unique_labels: List[int]) -> Dict[int, str]:
        """
        Get human-readable names from label_config
        
        Args:
            task_name: Name of the task (discipline, field, method)
            unique_labels: List of unique label values
            
        Returns:
            Dictionary mapping label values to human-readable names
        """
        label_names = {}
        
        if self.label_config and 'label_tasks' in self.label_config:
            if task_name in self.label_config['label_tasks']:
                task_config = self.label_config['label_tasks'][task_name]
                if 'labels' in task_config:
                    task_labels = task_config['labels']
                    for label in unique_labels:
                        if label in task_labels:
                            label_names[label] = task_labels[label]
                        else:
                            label_names[label] = f"Unknown_{task_name}_{label}"
            else:
                # Task not in config, use generic names
                for label in unique_labels:
                    label_names[label] = f"{task_name}_{label}"
        else:
            # No config available, use generic names
            for label in unique_labels:
                label_names[label] = f"Label_{label}"
        
        return label_names
    
    def save_mappings(self, path: str):
        """
        Save mappings to JSON file
        
        Args:
            path: Path where to save the mappings
        """
        # Convert mappings to JSON-serializable format
        json_mappings = {}
        for column, mapping in self.mappings.items():
            json_mappings[column] = {
                'label2id': {str(k): v for k, v in mapping['label2id'].items()},
                'id2label': {str(k): v for k, v in mapping['id2label'].items()},
                'label_names': {str(k): v for k, v in mapping['label_names'].items()},
                'num_classes': mapping['num_classes'],
                'original_labels': mapping['original_labels'],
                'label_column': mapping['label_column']
            }
        
        with open(path, 'w') as f:
            json.dump(json_mappings, f, indent=2)
        
        print(f"Label mappings saved to {path}")
    
    def load_mappings(self, path: str):
        """
        Load mappings from JSON file
        
        Args:
            path: Path to the mappings file
        """
        with open(path, 'r') as f:
            json_mappings = json.load(f)
        
        # Convert back from JSON format
        self.mappings = {}
        for column, mapping in json_mappings.items():
            self.mappings[column] = {
                'label2id': {int(k): v for k, v in mapping['label2id'].items()},
                'id2label': {int(k): v for k, v in mapping['id2label'].items()},
                'label_names': {int(k): v for k, v in mapping['label_names'].items()},
                'num_classes': mapping['num_classes'],
                'original_labels': mapping['original_labels'],
                'label_column': mapping['label_column']
            }
        
        print(f"Label mappings loaded from {path}")
    
    def get_id2label(self, label_column: str) -> Dict[int, int]:
        """
        Get id2label mapping for specific column
        
        Args:
            label_column: Name of the label column
            
        Returns:
            Dictionary mapping model outputs to original labels
        """
        if label_column in self.mappings:
            return self.mappings[label_column]['id2label']
        return {}
    
    def get_label2id(self, label_column: str) -> Dict[int, int]:
        """
        Get label2id mapping for specific column
        
        Args:
            label_column: Name of the label column
            
        Returns:
            Dictionary mapping original labels to model indices
        """
        if label_column in self.mappings:
            return self.mappings[label_column]['label2id']
        return {}
    
    def get_label_name(self, label_column: str, label_id: int) -> str:
        """
        Get human-readable name for a label
        
        Args:
            label_column: Name of the label column
            label_id: Original label value
            
        Returns:
            Human-readable label name
        """
        if label_column in self.mappings:
            label_names = self.mappings[label_column].get('label_names', {})
            return label_names.get(label_id, f"Label_{label_id}")
        return f"Label_{label_id}"
    
    def get_num_classes(self, label_column: str) -> Optional[int]:
        """
        Get number of classes for a label column
        
        Args:
            label_column: Name of the label column
            
        Returns:
            Number of classes or None if not found
        """
        if label_column in self.mappings:
            return self.mappings[label_column]['num_classes']
        return None
    
    def save_mapping_report(self, output_dir: str):
        """
        Save human-readable mapping report
        
        Args:
            output_dir: Directory where to save the report
        """
        report_path = os.path.join(output_dir, 'label_mapping_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("LABEL MAPPING REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            for column, mapping in self.mappings.items():
                f.write(f"Label Column: {column}\n")
                f.write(f"Number of Classes: {mapping['num_classes']}\n")
                f.write(f"Original Labels: {mapping['original_labels']}\n")
                f.write("\n")
                f.write("Mapping Table:\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'Model Output':<15} {'Original Label':<15} {'Label Name':<30}\n")
                f.write("-" * 60 + "\n")
                
                for model_idx in range(mapping['num_classes']):
                    orig_label = mapping['id2label'][model_idx]
                    label_name = mapping['label_names'].get(orig_label, 'Unknown')
                    f.write(f"{model_idx:<15} {orig_label:<15} {label_name:<30}\n")
                
                f.write("\n" + "=" * 70 + "\n\n")
        
        print(f"Label mapping report saved to {report_path}")