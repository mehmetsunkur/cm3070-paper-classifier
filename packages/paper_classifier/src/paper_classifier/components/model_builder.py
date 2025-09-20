"""
ModelBuilderComponent: Handles model initialization and configuration
"""

import os
import torch
from typing import Dict, Optional, Tuple
from ..mamba.model import MambaTextClassification


class ModelBuilderComponent:
    def __init__(self, config: Dict):
        """
        Initialize ModelBuilderComponent with configuration
        
        Args:
            config: Flat configuration dictionary containing model settings
        """
        self.config = config
        self.model = None
        self.device = None
        
    def detect_device(self) -> torch.device:
        """
        Detect and configure the appropriate device (GPU/CPU)
        
        Returns:
            torch.device instance
        """
        if torch.cuda.is_available():
            # Check if we're in a distributed setting
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device(f"cuda:{local_rank}")
            
            print(f"\nDevice Configuration:")
            print(f"  Process rank: {local_rank}")
            print(f"  Using device: {self.device}")
            print(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(local_rank) / 1024**3:.2f} GB")
            print(f"  Memory reserved: {torch.cuda.memory_reserved(local_rank) / 1024**3:.2f} GB")
            
            # Set the current device for this process
            torch.cuda.set_device(local_rank)
        else:
            self.device = torch.device("cpu")
            print("\nUsing CPU device")
        
        return self.device
    
    def build_model(self, tokenizer, num_classes: int = 2) -> MambaTextClassification:
        """
        Build and configure the Mamba model
        
        Args:
            tokenizer: Tokenizer instance for setting special token IDs
            num_classes: Number of classes for classification (default: 2)
            
        Returns:
            Configured MambaTextClassification model
        """
        model_name = self.config.get('model_name', 'state-spaces/mamba-130m')
        print(f"\nLoading model: {model_name}")
        print(f"  Configuring for {num_classes} classes")
        
        # Load the base model first
        from ..config.mamba_config import MambaConfig
        from mamba_ssm.utils.hf import load_config_hf
        
        # Load base config and update num_classes
        config_data = load_config_hf(model_name)
        base_config = MambaConfig(**config_data)
        base_config.num_classes = num_classes
        
        # Load the model with updated config
        self.model = MambaTextClassification.from_pretrained(
            model_name,
            config=base_config,
            num_classes=num_classes
        )
        
        # Move to device
        if self.device:
            self.model.to(self.device)
        
        # Set special token IDs in model config
        self.model.config.eos_token_id = tokenizer.eos_token_id
        self.model.config.bos_token_id = tokenizer.bos_token_id
        self.model.config.pad_token_id = tokenizer.pad_token_id
        
        # Enable gradient checkpointing if requested
        use_gradient_checkpointing = self.config.get('training_gradient_checkpointing', False)
        if use_gradient_checkpointing:
            self.model.config.gradient_checkpointing = True
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            print("  Gradient checkpointing: Enabled")
        else:
            print("  Gradient checkpointing: Disabled")
        
        # Print model statistics
        self._print_model_stats()
        
        return self.model
    
    def _print_model_stats(self):
        """Print model parameter statistics"""
        if not self.model:
            return
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\nModel Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        
        # Memory footprint estimation (assuming float32)
        memory_mb = (trainable_params * 4) / (1024 * 1024)
        print(f"  Estimated memory: {memory_mb:.2f} MB (float32)")
        if self.config.get('training_fp16', False):
            print(f"  With FP16: {memory_mb/2:.2f} MB")
    
    def get_model_info(self) -> Dict:
        """
        Get model information for logging
        
        Returns:
            Dictionary with model information
        """
        if not self.model:
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.config.get('model_name', 'state-spaces/mamba-130m'),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'gradient_checkpointing': self.config.get('training_gradient_checkpointing', False),
            'device': str(self.device) if self.device else 'cpu'
        }
    
    def prepare_for_training(self, model: Optional[MambaTextClassification] = None) -> MambaTextClassification:
        """
        Prepare model for training with optimizations
        
        Args:
            model: Model instance (uses self.model if not provided)
            
        Returns:
            Prepared model
        """
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model available. Call build_model() first.")
        
        # Set model to training mode
        model.train()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return model