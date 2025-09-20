"""
OptimizationComponent: Handles memory and GPU optimizations
"""

import os
import random
import numpy as np
import torch
from typing import Dict, Optional


class OptimizationComponent:
    def __init__(self, config: Dict):
        """
        Initialize OptimizationComponent with configuration
        
        Args:
            config: Flat configuration dictionary containing optimization settings
        """
        self.config = config
        self.has_bitsandbytes = self._check_bitsandbytes()
        
    def _check_bitsandbytes(self) -> bool:
        """Check if bitsandbytes is available for 8-bit optimization"""
        try:
            import bitsandbytes as bnb
            print("bitsandbytes available - 8-bit Adam optimizer can be used")
            return True
        except ImportError:
            print("bitsandbytes not available - using standard AdamW")
            return False
    
    def set_seed(self, seed: Optional[int] = None):
        """
        Set random seeds for reproducibility
        
        Args:
            seed: Random seed value (uses config value if not provided)
        """
        if seed is None:
            seed = self.config.get('training_seed', 42)
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # Enable for stable performance
        
        print(f"Random seed set to: {seed}")
    
    def print_memory_stats(self, stage: str = "current"):
        """
        Print current GPU memory statistics
        
        Args:
            stage: Description of the current stage
        """
        if not torch.cuda.is_available():
            return
        
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.synchronize()
        
        print(f"\nMemory stats ({stage}) - GPU {local_rank}:")
        print(f"  Allocated: {torch.cuda.memory_allocated(local_rank) / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(local_rank) / 1024**3:.2f} GB")
        print(f"  Max reserved: {torch.cuda.max_memory_reserved(local_rank) / 1024**3:.2f} GB")
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("GPU cache cleared")
    
    def get_optimization_config(self) -> Dict:
        """
        Get optimization configuration for logging
        
        Returns:
            Dictionary with optimization settings
        """
        return {
            'fp16': self.config.get('training_fp16', True),
            'gradient_checkpointing': self.config.get('training_gradient_checkpointing', False),
            'gradient_accumulation_steps': self.config.get('training_gradient_accumulation_steps', 4),
            'max_grad_norm': self.config.get('training_max_grad_norm', 0.5),
            'optimizer': 'adamw_bnb_8bit' if self.has_bitsandbytes else 'adamw_torch',
            'dynamic_padding': self.config.get('training_use_dynamic_padding', False),
            'dataloader_num_workers': self.config.get('data_dataloader_num_workers', 8),
            'prefetch_factor': self.config.get('hardware_prefetch_factor', 2),
        }
    
    def print_configuration_summary(self, max_seq_length: int):
        """
        Print optimization configuration summary
        
        Args:
            max_seq_length: Maximum sequence length for the model
        """
        batch_size_train = self.config.get('training_batch_size_train', 4)
        batch_size_eval = self.config.get('training_batch_size_eval', 8)
        gradient_accumulation = self.config.get('training_gradient_accumulation_steps', 4)
        
        print("=" * 60)
        print("OPTIMIZATION CONFIGURATION")
        print("=" * 60)
        print(f"Max sequence length: {max_seq_length:,} tokens")
        print(f"Train batch size: {batch_size_train} (per device)")
        print(f"Eval batch size: {batch_size_eval} (per device)")
        print(f"Gradient accumulation: {gradient_accumulation} steps")
        print(f"Effective batch size: {batch_size_train * gradient_accumulation}")
        print(f"Mixed precision (FP16): {self.config.get('training_fp16', True)}")
        print(f"Gradient checkpointing: {self.config.get('training_gradient_checkpointing', False)}")
        print(f"Dynamic padding: {self.config.get('training_use_dynamic_padding', False)}")
        print(f"Dataloader workers: {self.config.get('data_dataloader_num_workers', 8)}")
        print(f"Prefetch factor: {self.config.get('hardware_prefetch_factor', 2)}")
        
        # Evaluation settings
        eval_strategy = self.config.get('evaluation_eval_strategy', 'steps')
        print(f"Evaluation strategy: {eval_strategy}")
        if eval_strategy == "steps":
            print(f"Evaluation frequency: every {self.config.get('evaluation_eval_steps', 50)} steps")
        else:
            print(f"Evaluation frequency: every epoch")
        
        print(f"Logging frequency: every {self.config.get('logging_logging_steps', 25)} steps")
        
        # Early stopping
        early_stopping = self.config.get('evaluation_early_stopping_patience')
        print(f"Early stopping: {'Enabled' if early_stopping else 'Disabled'}")
        if early_stopping:
            print(f"  Patience: {early_stopping} evaluations")
            print(f"  Threshold: {self.config.get('evaluation_early_stopping_threshold', 0.0001)}")
        
        print(f"Learning rate: {self.config.get('training_learning_rate', 1e-5)}")
        print(f"Max gradient norm: {self.config.get('training_max_grad_norm', 0.5)}")
        print("=" * 60)
    
    def optimize_model_for_inference(self, model):
        """
        Optimize model for inference
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return model
    
    def handle_oom_error(self, batch_size: int, seq_length: int):
        """
        Provide suggestions for handling OOM errors
        
        Args:
            batch_size: Current batch size
            seq_length: Current sequence length
        """
        print("\nCUDA Out of Memory Error!")
        print("Suggestions to resolve:")
        print(f"  Current: batch_size={batch_size}, seq_length={seq_length}")
        print(f"  Try: --batch-size-train {max(1, batch_size//2)} --max-seq-length {max(2048, seq_length//2)}")
        print("  Enable gradient checkpointing: --use-gradient-checkpointing")
        print("  Reduce gradient accumulation steps")
        print("  Use FP16 training: --use-fp16")