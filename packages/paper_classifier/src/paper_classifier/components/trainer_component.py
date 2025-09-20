"""
TrainerComponent: Handles the training logic
"""

import os
import torch
from typing import Dict, Optional, Tuple, Any
from transformers import TrainingArguments, EarlyStoppingCallback
from ..mamba.trainer import MambaTrainer
from ..utils import compute_metrics


class TrainerComponent:
    def __init__(self, config: Dict):
        """
        Initialize TrainerComponent with configuration
        
        Args:
            config: Flat configuration dictionary containing training settings
        """
        self.config = config
        self.trainer = None
        self.training_args = None
        
    def create_training_arguments(self, output_dir: str, run_name: str) -> TrainingArguments:
        """
        Create TrainingArguments from configuration
        
        Args:
            output_dir: Directory for model outputs
            run_name: Name for the training run
            
        Returns:
            TrainingArguments instance
        """
        # Extract configuration values with defaults
        batch_size_train = self.config.get('training_batch_size_train', 4)
        batch_size_eval = self.config.get('training_batch_size_eval', 8)
        gradient_accumulation = self.config.get('training_gradient_accumulation_steps', 4)
        num_epochs = self.config.get('training_num_epochs', 1)
        learning_rate = self.config.get('training_learning_rate', 1e-5)
        max_grad_norm = self.config.get('training_max_grad_norm', 0.5)
        
        # Evaluation settings
        eval_strategy = self.config.get('evaluation_eval_strategy', 'steps')
        eval_steps = self.config.get('evaluation_eval_steps', 50)
        
        # Logging settings
        logging_steps = self.config.get('logging_logging_steps', 25)
        
        # Hardware optimizations
        fp16 = self.config.get('training_fp16', True)
        gradient_checkpointing = self.config.get('training_gradient_checkpointing', False)
        dataloader_workers = self.config.get('data_dataloader_num_workers', 8)
        prefetch_factor = self.config.get('hardware_prefetch_factor', 2)
        
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            
            # Learning rate and schedule
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            
            # Batch sizes and accumulation
            per_device_train_batch_size=batch_size_train,
            per_device_eval_batch_size=batch_size_eval,
            gradient_accumulation_steps=gradient_accumulation,
            
            # Training duration
            num_train_epochs=num_epochs,
            max_steps=-1,
            
            # Memory optimizations
            fp16=fp16,
            fp16_opt_level="O1" if fp16 else None,
            gradient_checkpointing=gradient_checkpointing,
            optim="adamw_torch",
            max_grad_norm=max_grad_norm,
            
            # Data loading optimizations
            dataloader_pin_memory=True,
            dataloader_num_workers=dataloader_workers,
            dataloader_prefetch_factor=prefetch_factor,
            dataloader_persistent_workers=True if dataloader_workers > 0 else False,
            remove_unused_columns=False,
            
            # Evaluation and checkpointing
            eval_strategy=eval_strategy,
            eval_steps=eval_steps if eval_strategy == "steps" else None,
            save_strategy=eval_strategy,
            save_steps=eval_steps if eval_strategy == "steps" else None,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Logging
            logging_strategy="steps",
            logging_steps=logging_steps,
            logging_first_step=True,
            report_to="wandb",
            run_name=run_name,
            
            # Other settings
            push_to_hub=False,
            seed=42,
            data_seed=42,
            
            # Performance settings
            full_determinism=False,
            torchdynamo=None,
            torch_compile=False,
            
            # Distributed training settings
            ddp_find_unused_parameters=False,
            ddp_bucket_cap_mb=25,
            
            # Additional stability settings
            ignore_data_skip=True,
            eval_delay=0,
            logging_nan_inf_filter=True,
        )
        
        return self.training_args
    
    def create_callbacks(self):
        """
        Create training callbacks based on configuration
        
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Early stopping callback
        early_stopping_patience = self.config.get('evaluation_early_stopping_patience')
        if early_stopping_patience:
            early_stopping_threshold = self.config.get('evaluation_early_stopping_threshold', 0.0001)
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold
            ))
            print(f"Early stopping enabled: patience={early_stopping_patience}, threshold={early_stopping_threshold}")
        
        return callbacks if callbacks else None
    
    def initialize_trainer(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        data_collator,
        training_args: Optional[TrainingArguments] = None
    ) -> MambaTrainer:
        """
        Initialize the MambaTrainer
        
        Args:
            model: The model to train
            tokenizer: Tokenizer instance
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator for batching
            training_args: Optional TrainingArguments (uses self.training_args if not provided)
            
        Returns:
            MambaTrainer instance
        """
        if training_args is None:
            training_args = self.training_args
        
        if training_args is None:
            raise ValueError("Training arguments not created. Call create_training_arguments() first.")
        
        # Check if running with FSDP
        is_fsdp = hasattr(training_args, 'fsdp') and training_args.fsdp is not None and training_args.fsdp != ""
        
        # Handle optimizer based on FSDP mode
        if is_fsdp:
            print("FSDP mode detected - using default optimizer creation")
            optimizers = (None, None)
        else:
            print("Standard training mode - using default AdamW optimizer")
            optimizers = (None, None)
        
        # Get callbacks
        callbacks = self.create_callbacks()
        
        # Initialize trainer
        self.trainer = MambaTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            callbacks=callbacks,
        )
        
        return self.trainer
    
    def train(self, trainer: Optional[MambaTrainer] = None) -> Dict:
        """
        Execute training
        
        Args:
            trainer: Optional trainer instance (uses self.trainer if not provided)
            
        Returns:
            Training results dictionary
        """
        if trainer is None:
            trainer = self.trainer
        
        if trainer is None:
            raise ValueError("Trainer not initialized. Call initialize_trainer() first.")
        
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        
        try:
            # Start training
            trainer.train()
            
            print("\nTraining completed successfully!")
            
            # Check if training was stopped early
            if trainer.state.best_metric is not None:
                print(f"\nBest metric: {trainer.state.best_metric:.4f}")
                if hasattr(trainer.state, 'stop_training') and trainer.state.stop_training:
                    print("Training stopped early due to patience exhaustion")
            
            return {
                'success': True,
                'best_metric': trainer.state.best_metric,
                'stopped_early': getattr(trainer.state, 'stop_training', False)
            }
            
        except torch.cuda.OutOfMemoryError as e:
            print("\nCUDA Out of Memory Error!")
            print("Try reducing batch size or sequence length")
            return {
                'success': False,
                'error': 'OOM',
                'message': str(e)
            }
            
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            return {
                'success': False,
                'error': 'GENERAL',
                'message': str(e)
            }
    
    def save_model(self, output_dir: str, trainer: Optional[MambaTrainer] = None):
        """
        Save the trained model
        
        Args:
            output_dir: Directory to save the model
            trainer: Optional trainer instance (uses self.trainer if not provided)
        """
        if trainer is None:
            trainer = self.trainer
        
        if trainer is None:
            raise ValueError("Trainer not initialized")
        
        final_model_path = os.path.join(output_dir, "final_model")
        trainer.save_model(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        return final_model_path
    
    def evaluate(self, test_dataset, trainer: Optional[MambaTrainer] = None) -> Dict:
        """
        Evaluate model on test dataset
        
        Args:
            test_dataset: Test dataset
            trainer: Optional trainer instance (uses self.trainer if not provided)
            
        Returns:
            Evaluation results
        """
        if trainer is None:
            trainer = self.trainer
        
        if trainer is None:
            raise ValueError("Trainer not initialized")
        
        print("\nEvaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        print(f"Test results: {test_results}")
        
        return test_results