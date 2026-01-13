import torch.nn as nn
import torch.nn.functional as F
import torch

from transformers import AutoModel

import sys
import os
sys.path.append(os.environ.get("TAP_EVAL_PATH"))

from config.ModelConfig import MPNET_PATH


class SentenceEncoderWithProjection(nn.Module):
    """
    An encoder that wraps a pre-trained Transformer model with custom pooling and projection head.
    
    This model converts input text sentences into fixed-dimensional embeddings (e.g., 256-dimensional)
    suitable for specific tasks (e.g., harmfulness assessment).
    """
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', projection_dim=256):
        """
        Initialize the encoder.
        
        Args:
        - model_name (str): Pre-trained model name from Hugging Face Hub.
        - projection_dim (int): Dimension of the final output harmfulness representation space.
        """
        super(SentenceEncoderWithProjection, self).__init__()
        
        # 1. Load pre-trained base model (without tokenizer)
        model_path = MPNET_PATH
        # model_path = os.environ.get("MPNET_MODEL_PATH", "sentence-transformers/all-mpnet-base-v2")
        self.base_model = AutoModel.from_pretrained(model_path)
        
        # Note: tokenizer is moved to external, not initialized inside the model

        # 2. Dynamically get base model output dimension (usually 768)
        self.input_embedding_dim = self.base_model.config.hidden_size
        
        # 3. Define a trainable projection head
        #    This head maps 768-dimensional general semantic vectors to 256-dimensional harmfulness semantic space
        #    We add a non-linear activation function (ReLU) to enhance its ability to learn complex mappings
        self.projection_head = nn.Sequential(
            nn.Linear(self.input_embedding_dim, self.input_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.input_embedding_dim, projection_dim)
        )

    def mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging"""
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model - modified to receive tokenized tensor
        
        Args:
        - input_ids: [batch_size, seq_len] tensor 
        - attention_mask: [batch_size, seq_len] tensor
        
        Returns:
        - torch.Tensor: Final embedding vectors of shape (batch_size, projection_dim).
        """
        # Use tokenized tensor input directly
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        # Perform mean pooling to get sentence-level vectors (768-dimensional)
        pooled_embedding = self.mean_pooling(outputs, attention_mask)
        
        # Pass pooled vectors through projection head to get final representation (256-dimensional)
        final_embedding = self.projection_head(pooled_embedding)

        # Layer normalization
        final_embedding = F.normalize(final_embedding, p=2, dim=1)

        return final_embedding
    
