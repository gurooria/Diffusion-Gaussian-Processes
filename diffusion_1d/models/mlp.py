import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreModel(nn.Module):
    def __init__(self, hidden_dims=[16, 32, 64], time_embed_dim=32):
        """
        MLP model for predicting the conditional score and noise variance in diffusion models.
        
        Args:
            hidden_dims (list): List of hidden dimensions for the MLP layers
            time_embed_dim (int): Dimension of time embedding
        """
        super().__init__()
        
        # Time embedding
        self.time_embed_dim = time_embed_dim
        
        # Input dimension is 2 + time_embed_dim (x, y_t, time embedding)
        input_dim = 2 + time_embed_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim), 
                nn.SiLU(),  # SiLU activation
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final layer to output score (noise)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def get_timestep_embedding(self, t):
        """
        Create sinusoidal time embedding.
        
        Args:
            t (torch.Tensor): Time steps tensor of shape (batch_size,)
            
        Returns:
            torch.Tensor: Time embedding of shape (batch_size, time_embed_dim)
        """
        half_dim = self.time_embed_dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :].to(t.device)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
    def forward(self, x, y_t, t):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input locations, shape (batch_size,)
            y_t (torch.Tensor): Noised samples at time step t, shape (batch_size,)
            t (torch.Tensor): Time steps, shape (batch_size,)
            
        Returns:
            torch.Tensor: Predicted score (noise), shape (batch_size,)
        """
        # Get time embeddings
        t_embed = self.get_timestep_embedding(t)
        
        # Concatenate inputs along feature dimension
        inputs = torch.cat([x.unsqueeze(-1), y_t.unsqueeze(-1), t_embed], dim=1)
        
        # Pass through MLP
        score = self.mlp(inputs)
        
        # Remove last dimension to get shape (batch_size,)
        return score.squeeze(-1)