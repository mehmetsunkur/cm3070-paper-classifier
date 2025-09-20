"""
LoggingComponent: Handles WandB integration and logging
"""

import os
import json
import wandb
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path


class LoggingComponent:
    def __init__(self, config: Dict):
        """
        Initialize LoggingComponent with configuration
        
        Args:
            config: Flat configuration dictionary containing logging settings
        """
        self.config = config
        self.run = None
        self.output_dir = None
        self.run_name = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_output_directory(self, model_name: Optional[str] = None, dataset_slug: str = "") -> str:
        """
        Create output directory for model checkpoints
        
        Args:
            model_name: Optional model name for directory
            dataset_slug: Dataset slug for directory naming
            
        Returns:
            Path to output directory
        """
        if model_name:
            self.output_dir = f"trained_models/{model_name}/run_{self.timestamp}"
        else:
            project_slug = dataset_slug or "mamba_training"
            self.output_dir = f"trained_models/{project_slug}/run_{self.timestamp}"
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
        return self.output_dir
    
    def save_dataset_info(self, dataset_info: Dict):
        """
        Save dataset information to output directory
        
        Args:
            dataset_info: Dictionary containing dataset information
        """
        if not self.output_dir:
            raise ValueError("Output directory not created. Call create_output_directory() first.")
        
        dataset_info_path = os.path.join(self.output_dir, "dataset_info.json")
        with open(dataset_info_path, "w") as f:
            json.dump(dataset_info, f, indent=2)
        print(f"Dataset info saved to: {dataset_info_path}")
    
    def generate_run_name(self) -> str:
        """
        Generate a run name based on configuration
        
        Returns:
            Generated run name
        """
        if self.config.get('logging_run_name'):
            self.run_name = self.config['logging_run_name']
        else:
            # Build run name from key parameters
            run_parts = [self.timestamp]
            
            # Add early stopping info if enabled
            if self.config.get('evaluation_early_stopping_patience'):
                run_parts.append(f"es{self.config['evaluation_early_stopping_patience']}")
                threshold = self.config.get('evaluation_early_stopping_threshold', 0.0001)
                run_parts.append(f"th{int(threshold*10000)}")
            
            # Add training info
            run_parts.append(f"ep{self.config.get('training_num_epochs', 1)}")
            run_parts.append(f"lr{self.config.get('training_learning_rate', 1e-5):.0e}")
            
            self.run_name = "_".join(run_parts).lower()
        
        return self.run_name
    
    def create_wandb_config(
        self,
        parquet_path: str,
        dataset_sizes: Dict,
        model_info: Dict,
        max_seq_length: int
    ) -> Dict:
        """
        Create comprehensive WandB configuration
        
        Args:
            parquet_path: Path to the parquet file
            dataset_sizes: Dictionary with train/eval/test sizes
            model_info: Dictionary with model information
            max_seq_length: Maximum sequence length
            
        Returns:
            WandB configuration dictionary
        """
        parquet_name = os.path.basename(parquet_path).replace('.parquet', '').replace('_', '-')
        
        # Extract values from config
        batch_size_train = self.config.get('training_batch_size_train', 4)
        gradient_accumulation = self.config.get('training_gradient_accumulation_steps', 4)
        
        wandb_config = {
            # Dataset information
            "dataset_name": parquet_name,
            "parquet_path": parquet_path,
            "train_size": dataset_sizes.get('train', 0),
            "eval_size": dataset_sizes.get('eval', 0),
            "test_size": dataset_sizes.get('test', 0),
            
            # Model information
            "model_name": model_info.get('model_name', 'state-spaces/mamba-130m'),
            "tokenizer_name": self.config.get('tokenizer_model_name', 'EleutherAI/gpt-neox-20b'),
            "total_params": model_info.get('total_params', 0),
            "trainable_params": model_info.get('trainable_params', 0),
            
            # Training hyperparameters
            "learning_rate": self.config.get('training_learning_rate', 1e-5),
            "num_epochs": self.config.get('training_num_epochs', 1),
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "max_grad_norm": self.config.get('training_max_grad_norm', 0.5),
            
            # Batch size configuration
            "per_device_train_batch_size": batch_size_train,
            "per_device_eval_batch_size": self.config.get('training_batch_size_eval', 8),
            "gradient_accumulation_steps": gradient_accumulation,
            "effective_batch_size": batch_size_train * gradient_accumulation,
            
            # Sequence length
            "max_seq_length": max_seq_length,
            
            # Memory optimizations
            "fp16": self.config.get('training_fp16', True),
            "gradient_checkpointing": self.config.get('training_gradient_checkpointing', False),
            "dynamic_padding": self.config.get('training_use_dynamic_padding', False),
            
            # Data loading
            "dataloader_num_workers": self.config.get('data_dataloader_num_workers', 8),
            "dataloader_pin_memory": True,
            "prefetch_factor": self.config.get('hardware_prefetch_factor', 2),
            
            # Evaluation settings
            "eval_steps": self.config.get('evaluation_eval_steps', 50),
            "save_steps": self.config.get('evaluation_eval_steps', 50),
            "logging_steps": self.config.get('logging_logging_steps', 25),
            
            # Early stopping settings
            "early_stopping_patience": self.config.get('evaluation_early_stopping_patience'),
            "early_stopping_threshold": self.config.get('evaluation_early_stopping_threshold', 0.0001),
            "early_stopping_enabled": self.config.get('evaluation_early_stopping_patience') is not None,
            
            # System information
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "timestamp": self.timestamp,
        }
        
        return wandb_config
    
    def initialize_wandb(
        self,
        project_name: str,
        wandb_config: Dict,
        tags: Optional[List[str]] = None
    ):
        """
        Initialize WandB run
        
        Args:
            project_name: Name of the WandB project
            wandb_config: Configuration dictionary for WandB
            tags: Optional list of tags for the run
        """
        if not self.run_name:
            self.generate_run_name()
        
        # Create default tags if not provided
        if tags is None:
            tags = self._generate_tags(wandb_config)
        
        # Initialize WandB
        wandb_settings = wandb.Settings(x_label=self.timestamp)
        
        self.run = wandb.init(
            project=os.getenv("WANDB_PROJECT", project_name),
            name=self.run_name,
            mode="shared",
            config=wandb_config,
            settings=wandb_settings,
            tags=tags,
            notes=os.getenv("WANDB_NOTES", f"Component-based training on {wandb_config['dataset_name']}")
        )
        
        print(f"\nWandB Configuration:")
        print(f"  Project: {self.run.project}")
        print(f"  Run: {self.run_name}")
        print(f"  Run ID: {self.run.id}")
        print(f"  URL: {self.run.url}")
        
        return self.run
    
    def _generate_tags(self, wandb_config: Dict) -> List[str]:
        """
        Generate tags for WandB run
        
        Args:
            wandb_config: WandB configuration dictionary
            
        Returns:
            List of tags
        """
        tags = [
            wandb_config['dataset_name'],
            f"seq{wandb_config['max_seq_length']}",
            f"bs{wandb_config['effective_batch_size']}",
            f"ep{wandb_config['num_epochs']}",
            "component-based",
            "fp16" if wandb_config['fp16'] else "fp32",
            "grad-ckpt" if wandb_config['gradient_checkpointing'] else "no-grad-ckpt",
            "fixed-pad" if not wandb_config['dynamic_padding'] else "dynamic-pad",
        ]
        
        return tags
    
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """
        Log metrics to WandB
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if self.run:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
    
    def finish(self):
        """Finish WandB run"""
        if self.run:
            wandb.finish()
            print("WandB run finished")
    
    def save_config(self, config: Dict):
        """
        Save configuration to output directory
        
        Args:
            config: Configuration dictionary to save
        """
        if not self.output_dir:
            raise ValueError("Output directory not created")
        
        config_path = os.path.join(self.output_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to: {config_path}")


# Import torch for GPU detection in wandb_config
import torch