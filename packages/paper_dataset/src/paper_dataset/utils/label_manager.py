"""
Centralized label configuration manager.
Single source of truth for all label definitions and assignments.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import logging

logger = logging.getLogger(__name__)


class LabelManager:
    """Centralized label configuration manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize label manager with configuration.
        
        Args:
            config_path: Path to label_config.yaml. If None, uses default location.
        """
        if config_path is None:
            # Default path relative to this package
            config_path = Path(__file__).parent.parent / "configs" / "label_config.yaml"
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Label configuration not found at {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tasks = self.config['label_tasks']
        self.enabled_tasks = [name for name, task in self.tasks.items() if task.get('enabled', True)]
        
        logger.info(f"Loaded label configuration with {len(self.enabled_tasks)} enabled tasks: {self.enabled_tasks}")
    
    def get_task_names(self) -> List[str]:
        """Get list of enabled task names."""
        return self.enabled_tasks
    
    def get_num_classes(self, task: str) -> int:
        """Get number of classes for a task."""
        if task not in self.tasks:
            raise ValueError(f"Unknown task: {task}")
        return self.tasks[task]['num_classes']
    
    def get_weight(self, task: str) -> float:
        """Get loss weight for a task."""
        if task not in self.tasks:
            raise ValueError(f"Unknown task: {task}")
        return self.tasks[task].get('weight', 1.0)
    
    def get_labels(self, task: str) -> Dict[int, str]:
        """Get label mapping for a task."""
        if task not in self.tasks:
            raise ValueError(f"Unknown task: {task}")
        return self.tasks[task].get('labels', {})
    
    def get_all_weights(self) -> Dict[str, float]:
        """Get weights for all enabled tasks."""
        return {task: self.get_weight(task) for task in self.enabled_tasks}
    
    def get_total_weight(self) -> float:
        """Get sum of all enabled task weights."""
        return sum(self.get_all_weights().values())
    
    def assign_label(self, task: str, text: str, metadata: Optional[Dict] = None) -> int:
        """
        Assign a label for a specific task based on text and metadata.
        
        Args:
            task: Task name (e.g., 'discipline', 'field', 'method')
            text: Paper text content
            metadata: Optional metadata (e.g., arxiv categories)
            
        Returns:
            Label ID for the task
        """
        if task not in self.tasks:
            raise ValueError(f"Unknown task: {task}")
        
        task_config = self.tasks[task]
        
        # Handle discipline task with CC2020 mapping
        if task == 'discipline' and metadata and 'arxiv_mapping' in task_config:
            arxiv_categories = metadata.get('arxiv_categories', [])
            for category in arxiv_categories:
                if category in task_config['arxiv_mapping']:
                    return task_config['arxiv_mapping'][category]
        
        # Handle field task with ArXiv mapping or discipline constraints
        if task == 'field':
            # First try direct ArXiv mapping if available
            if metadata and 'arxiv_field_mapping' in task_config:
                arxiv_categories = metadata.get('arxiv_categories', [])
                for category in arxiv_categories:
                    if category in task_config['arxiv_field_mapping']:
                        return task_config['arxiv_field_mapping'][category]
            
            # If we have discipline info, constrain field selection
            if metadata and 'discipline' in metadata and 'discipline_constraints' in task_config:
                discipline_id = metadata['discipline']
                valid_fields = task_config['discipline_constraints'].get(discipline_id, [])
                # Find best field within valid constraints
                text_lower = text.lower()[:10000]
                for field_id in valid_fields:
                    if field_id in task_config['labels']:
                        field_name = task_config['labels'][field_id].lower()
                        if any(word in text_lower for word in field_name.split()):
                            return field_id
                # Return first valid field as default
                if valid_fields:
                    return valid_fields[0]
        
        # Check keywords based on task type
        text_lower = text.lower()
        
        # Different text lengths for different tasks
        if task == 'discipline':
            text_sample = text_lower[:5000]
        elif task == 'field':
            text_sample = text_lower[:10000]
        elif task == 'method':
            text_sample = text_lower[:10000]
        else:
            text_sample = text_lower[:15000]
        
        # Get keyword mapping
        if 'keywords' in task_config:
            keyword_scores = {}
            for keyword, label_id in task_config['keywords'].items():
                if keyword in text_sample:
                    keyword_scores[label_id] = keyword_scores.get(label_id, 0) + 1
            
            if keyword_scores:
                return max(keyword_scores, key=keyword_scores.get)
        
        # Fallback: keyword matching in label names
        if 'labels' in task_config:
            label_scores = {}
            for label_id, label_name in task_config['labels'].items():
                if isinstance(label_id, int):
                    keywords = label_name.lower().split()
                    score = sum(1 for keyword in keywords if keyword in text_sample)
                    if score > 0:
                        label_scores[label_id] = score
            
            if label_scores:
                return max(label_scores, key=label_scores.get)
        
        # Default labels based on task
        default_labels = {
            'discipline': 0,  # Computer Science
            'field': 0,  # Theory and Algorithms
            'method': 4,  # Experimental Evaluation
        }
        
        return default_labels.get(task, 0)
    
    def assign_all_labels(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, int]:
        """
        Assign labels for all enabled tasks.
        
        Args:
            text: Paper text content
            metadata: Optional metadata
            
        Returns:
            Dictionary mapping task names to label IDs
        """
        labels = {}
        for task in self.enabled_tasks:
            labels[task] = self.assign_label(task, text, metadata)
        return labels
    
    def create_model_config(self) -> Dict[str, int]:
        """
        Create model configuration parameters based on label config.
        
        Returns:
            Dictionary with num_<task>_classes for each enabled task
        """
        config = {}
        for task in self.enabled_tasks:
            config[f'num_{task}_classes'] = self.get_num_classes(task)
        return config
    
    def prepare_batch_labels(self, row: Dict) -> Dict[str, torch.Tensor]:
        """
        Convert dataframe row to model labels dynamically.
        
        Args:
            row: Dictionary-like row from dataframe
            
        Returns:
            Dictionary mapping labels_<task> to tensors
        """
        labels = {}
        for task in self.enabled_tasks:
            label_value = row.get(f'label_{task}', 0)
            labels[f'labels_{task}'] = torch.tensor(label_value, dtype=torch.long)
        return labels
    
    def get_label_columns(self) -> List[str]:
        """Get list of label column names for dataframe."""
        return [f'label_{task}' for task in self.enabled_tasks]
    
    def validate_labels(self, labels: Dict[str, int]) -> bool:
        """
        Validate that label values are within valid range.
        
        Args:
            labels: Dictionary mapping task names to label IDs
            
        Returns:
            True if all labels are valid
        """
        for task, label_id in labels.items():
            if task in self.enabled_tasks:
                if not (0 <= label_id < self.get_num_classes(task)):
                    logger.warning(f"Invalid label {label_id} for task {task} (max: {self.get_num_classes(task)-1})")
                    return False
        return True
    
    def get_task_info(self, task: str) -> Dict[str, Any]:
        """Get all configuration info for a specific task."""
        if task not in self.tasks:
            raise ValueError(f"Unknown task: {task}")
        return self.tasks[task]
    
    def __repr__(self) -> str:
        """String representation of label manager."""
        info = []
        for task in self.enabled_tasks:
            info.append(f"{task}: {self.get_num_classes(task)} classes, weight={self.get_weight(task)}")
        return f"LabelManager({', '.join(info)})"


# Singleton instance for easy import
_label_manager_instance = None

def get_label_manager(config_path: Optional[str] = None) -> LabelManager:
    """
    Get or create singleton label manager instance.
    
    Args:
        config_path: Optional path to label config file
        
    Returns:
        LabelManager instance
    """
    global _label_manager_instance
    if _label_manager_instance is None:
        _label_manager_instance = LabelManager(config_path)
    return _label_manager_instance