import numpy as np
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.hf import load_config_hf,load_state_dict_hf
from collections import namedtuple
import torch.nn as nn
import torch

from ..config.mamba_config import MambaConfig
from .head import MambaClassificationHead

class MambaTextClassification(MambaLMHeadModel):
    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg = None,
        device = None,
        dtype = None,
    ) -> None:
        super().__init__(config, initializer_cfg, device, dtype)
        
        # Create a classification head using MambaClassificationHead with input size of d_model and configurable number of classes.
        self.classification_head = MambaClassificationHead(d_model=config.d_model, num_classes=config.num_classes)
        
        del self.lm_head
        
        # Store config for gradient checkpointing
        self.config = config
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs = None):
        """Enable gradient checkpointing for memory efficiency."""
        print(gradient_checkpointing_kwargs)
        self.config.gradient_checkpointing = True
        print("Gradient checkpointing enabled in model")
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.config.gradient_checkpointing = False
        print("Gradient checkpointing disabled in model")
    
    def forward(self, input_ids, attention_mask = None, labels = None):
        # Pass input_ids through the backbone model to receive hidden_states.
        # Check if gradient checkpointing is enabled
        if getattr(self.config, 'gradient_checkpointing', False) and self.training:
            # Use gradient checkpointing for memory efficiency
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.backbone),
                input_ids,
                use_reentrant=False
            )
        else:
            hidden_states = self.backbone(input_ids)
        
        # Take the mean of hidden_states along the second dimension to create a representative [CLS] feature.
        mean_hidden_states = hidden_states.mean(dim = 1)
        
        # Pass mean_hidden_states through the classification head to get logits.
        logits = self.classification_head(mean_hidden_states)
        
        if labels is None:
            ClassificationOutput = namedtuple("ClassificationOutput", ["logits"])
            return ClassificationOutput(logits = logits)
        else:
            ClassificationOutput = namedtuple("ClassificationOutput", ["loss", "logits"])
            
            # Use CrossEntropyLoss loss function to compute the loss.
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
            return ClassificationOutput(loss = loss, logits = logits)
    def predict(self, text, tokenizer, id2label = None):
        input_ids = torch.tensor(tokenizer(text)['input_ids'], device = "cuda")[None]
        with torch.no_grad():
            logits = self.forward(input_ids).logits[0]
            label = np.argmax(logits.cpu().numpy())
            
        if id2label is not None:
            return id2label[label]
        else:
            return label
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name, device = None, dtype = None, config = None, num_classes = None, **kwargs):
        # Load the configuration from the pre-trained model.
        if config is None:
            config_data = load_config_hf(pretrained_model_name)
            config = MambaConfig(**config_data)
        
        # Override num_classes if provided
        if num_classes is not None:
            config.num_classes = num_classes
            print(f"Setting model to {num_classes} classes")
        
        # Initialize the model from the configuration and move it to the desired device and data type.
        model = cls(config, device = device, dtype = dtype, **kwargs)
        
        # Load the state of the pre-trained model.
        model_state_dict = load_state_dict_hf(pretrained_model_name, device = device, dtype = dtype)
        model.load_state_dict(model_state_dict , strict=False)
        
        # Print the newly initialized embedding parameters.
        print (" Newly initialized embedding :", 
              set(model.state_dict().keys()) - set(model_state_dict.keys())
        )

        return model.to(device)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device = None, dtype = None):
        """Load a model from a local checkpoint directory."""
        import json
        import torch
        
        # Load config from checkpoint
        config_path = f"{checkpoint_path}/config.json"
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = MambaConfig(**config_data)
        
        # Initialize model with config (including num_classes)
        model = cls(config, device = device, dtype = dtype)
        
        # Load state dict from checkpoint
        state_dict_path = f"{checkpoint_path}/pytorch_model.bin"
        state_dict = torch.load(state_dict_path, map_location=device if device else 'cpu')
        model.load_state_dict(state_dict)
        
        print(f"Loaded model from checkpoint: {checkpoint_path}")
        print(f"Model configured for {config.num_classes} classes")
        
        return model.to(device) if device else model