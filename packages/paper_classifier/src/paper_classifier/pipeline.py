#!/usr/bin/env python3
"""
Main training pipeline orchestrator using component-based architecture
"""

import sys
import torch
from typing import Dict, Optional
from pathlib import Path

# Import components
from .components import (
    DataLoaderComponent,
    ModelBuilderComponent,
    TrainerComponent,
    OptimizationComponent,
    LoggingComponent
)
from .components.label_mapper import LabelMapper
from .components.validation import LabelValidator


class TrainingPipeline:
    """
    Main orchestrator for the component-based training pipeline
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the training pipeline with configuration
        
        Args:
            config: Flat configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.data_loader = DataLoaderComponent(config)
        self.model_builder = ModelBuilderComponent(config)
        self.trainer_component = TrainerComponent(config)
        self.optimization = OptimizationComponent(config)
        self.logger = LoggingComponent(config)
        
        # Pipeline state
        self.model = None
        self.tokenizer = None
        self.datasets = None
        self.trainer = None
        self.num_classes = None
        self.label_mapper = None
        self.validator = LabelValidator()
        
    def setup_environment(self):
        """Set up the training environment"""
        print("\n" + "=" * 60)
        print("INITIALIZING TRAINING PIPELINE")
        print("=" * 60)
        
        # Set random seeds for reproducibility
        self.optimization.set_seed()
        
        # Detect and configure device
        self.model_builder.detect_device()
        
    def load_data(self, parquet_path: str):
        """
        Load and prepare datasets
        
        Args:
            parquet_path: Path to the parquet file
        """
        print("\n" + "-" * 40)
        print("DATA LOADING")
        print("-" * 40)
        
        # Initialize tokenizer
        self.tokenizer = self.data_loader.initialize_tokenizer()
        
        # Load dataset
        self.data_loader.load_dataset(parquet_path)
        
        # Initialize label mapper
        self.label_mapper = LabelMapper()
        
        # Get label mapping info from dataset
        dataset = self.data_loader.parquet_dataset
        mapping_info = dataset.get_label_mappings()
        
        # Get number of classes from dataset (after potential remapping)
        self.num_classes = self.data_loader.get_num_classes()
        print(f"\nDetected {self.num_classes} classes in dataset")
        
        # Create label mapping if remapping was applied
        if mapping_info['needs_remapping']:
            label_column = self.config.get('subset_generation_label_column', 'label_discipline')
            original_labels = sorted(mapping_info['id2label'].values())
            self.label_mapper.create_mapping(
                label_column=label_column,
                unique_labels=original_labels
            )
            # Store the actual mappings from the dataset
            self.label_mapper.mappings[label_column]['id2label'] = mapping_info['id2label']
            self.label_mapper.mappings[label_column]['label2id'] = mapping_info['label2id']
        
        # Get train, eval, test datasets
        train_dataset, eval_dataset, test_dataset = self.data_loader.get_datasets()
        self.datasets = {
            'train': train_dataset,
            'eval': eval_dataset,
            'test': test_dataset
        }
        
        # Validate labels after remapping
        self.validator.validate_labels_before_training(
            self.data_loader.parquet_dataset,
            self.num_classes
        )
        
        # Validate mapping consistency if applicable
        if self.label_mapper and self.label_mapper.mappings:
            self.validator.validate_mapping_consistency(
                self.label_mapper,
                self.data_loader.parquet_dataset
            )
        
        # Create data collator
        data_collator = self.data_loader.create_data_collator()
        
        return data_collator
    
    def build_model(self):
        """Build and configure the model"""
        print("\n" + "-" * 40)
        print("MODEL INITIALIZATION")
        print("-" * 40)
        
        # Build model with correct number of classes
        if self.num_classes is None:
            raise ValueError(
                "Failed to detect number of classes from the dataset. "
                "Please ensure the label column exists and contains valid class labels."
            )
        
        self.model = self.model_builder.build_model(
            self.tokenizer,
            num_classes=self.num_classes
        )
        
        # Prepare for training
        self.model = self.model_builder.prepare_for_training(self.model)
        
        return self.model
    
    def setup_logging(self, parquet_path: str, model_name: Optional[str] = None):
        """
        Set up logging and WandB
        
        Args:
            parquet_path: Path to the parquet file
            model_name: Optional model name for output directory
        """
        print("\n" + "-" * 40)
        print("LOGGING SETUP")
        print("-" * 40)
        
        # Get dataset info
        dataset_info = self.data_loader.get_dataset_info(parquet_path)
        
        # Create output directory
        output_dir = self.logger.create_output_directory(
            model_name=model_name,
            dataset_slug=dataset_info['dataset_slug']
        )
        
        # Save dataset info
        self.logger.save_dataset_info(dataset_info['dataset_info'].model_dump())
        
        # Save training configuration
        self.logger.save_config(self.config)
        
        # Generate run name
        run_name = self.logger.generate_run_name()
        
        # Create WandB configuration
        dataset_sizes = {
            'train': len(self.datasets['train']),
            'eval': len(self.datasets['eval']),
            'test': len(self.datasets['test'])
        }
        
        model_info = self.model_builder.get_model_info()
        
        wandb_config = self.logger.create_wandb_config(
            parquet_path=parquet_path,
            dataset_sizes=dataset_sizes,
            model_info=model_info,
            max_seq_length=self.data_loader.max_seq_length
        )
        
        # Initialize WandB
        project_name = model_name or dataset_info['dataset_slug']
        self.logger.initialize_wandb(project_name, wandb_config)
        
        return output_dir, run_name
    
    def setup_training(self, output_dir: str, run_name: str, data_collator):
        """
        Set up the trainer
        
        Args:
            output_dir: Directory for model outputs
            run_name: Name for the training run
            data_collator: Data collator for batching
        """
        print("\n" + "-" * 40)
        print("TRAINER SETUP")
        print("-" * 40)
        
        # Print optimization configuration
        self.optimization.print_configuration_summary(self.data_loader.max_seq_length)
        
        # Create training arguments
        training_args = self.trainer_component.create_training_arguments(output_dir, run_name)
        
        # Initialize trainer
        self.trainer = self.trainer_component.initialize_trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.datasets['train'],
            eval_dataset=self.datasets['eval'],
            data_collator=data_collator,
            training_args=training_args
        )
        
        return self.trainer
    
    def train(self):
        """Execute the training"""
        # Print memory stats before training
        self.optimization.print_memory_stats("before training")
        
        # Clear GPU cache
        self.optimization.clear_gpu_cache()
        
        # Train the model
        result = self.trainer_component.train(self.trainer)
        
        return result
    
    def evaluate_and_save(self, output_dir: str):
        """
        Evaluate on test set and save the model
        
        Args:
            output_dir: Directory to save the model
        """
        print("\n" + "-" * 40)
        print("EVALUATION AND SAVING")
        print("-" * 40)
        
        # Save the model
        final_model_path = self.trainer_component.save_model(output_dir, self.trainer)
        
        # Save label mappings if they exist
        if self.label_mapper and self.label_mapper.mappings:
            import os
            mapping_path = os.path.join(output_dir, 'label_mappings.json')
            self.label_mapper.save_mappings(mapping_path)
            self.label_mapper.save_mapping_report(output_dir)
        
        # Evaluate on test set
        test_results = self.trainer_component.evaluate(self.datasets['test'], self.trainer)
        
        # Log test results to WandB
        self.logger.log_metrics(test_results)
        
        return final_model_path, test_results
    
    def cleanup(self):
        """Clean up resources"""
        print("\n" + "-" * 40)
        print("CLEANUP")
        print("-" * 40)
        
        # Print final memory stats
        self.optimization.print_memory_stats("after training")
        
        # Clear GPU cache
        self.optimization.clear_gpu_cache()
        
        # Finish WandB run
        self.logger.finish()
        
        print("\nPipeline completed successfully!")
    
    def run(self, parquet_path: str, model_name: Optional[str] = None):
        """
        Run the complete training pipeline
        
        Args:
            parquet_path: Path to the parquet file
            model_name: Optional model name for output directory
            
        Returns:
            Dictionary with training results
        """
        try:
            # Setup environment
            self.setup_environment()
            
            # Load data
            data_collator = self.load_data(parquet_path)
            
            # Build model
            self.build_model()
            
            # Setup logging
            output_dir, run_name = self.setup_logging(parquet_path, model_name)
            
            # Setup training
            self.setup_training(output_dir, run_name, data_collator)
            
            # Train
            train_result = self.train()
            
            if train_result['success']:
                # Evaluate and save
                final_model_path, test_results = self.evaluate_and_save(output_dir)
                
                # Cleanup
                self.cleanup()
                
                return {
                    'success': True,
                    'model_path': final_model_path,
                    'test_results': test_results,
                    'train_result': train_result
                }
            else:
                # Handle training failure
                if train_result['error'] == 'OOM':
                    self.optimization.handle_oom_error(
                        batch_size=self.config.get('training_batch_size_train', 4),
                        seq_length=self.data_loader.max_seq_length
                    )
                
                self.cleanup()
                return train_result
                
        except Exception as e:
            print(f"\nPipeline failed with error: {e}")
            self.cleanup()
            raise e


def run_pipeline(config: Dict, parquet_path: str, model_name: Optional[str] = None):
    """
    Convenience function to run the training pipeline
    
    Args:
        config: Configuration dictionary
        parquet_path: Path to the parquet file
        model_name: Optional model name for output directory
        
    Returns:
        Training results
    """
    pipeline = TrainingPipeline(config)
    return pipeline.run(parquet_path, model_name)


if __name__ == "__main__":
    # This script should be called through trainer_config.py
    # which provides the configuration
    print("This script should be called through trainer_config.py")
    print("Use: python trainer_config.py --config <config_file>")
    sys.exit(1)