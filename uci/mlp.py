import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreModel(nn.Module):
    def __init__(self, hidden_dims=[32, 64, 64], time_embed_dim=64):
        """
        Model for predicting conditional score in diffusion models.
        With improvements for better variance modeling:
        - Increased network capacity (wider and deeper)
        - Added residual connections
        - Group normalization instead of batch norm for better stability
        - Increased time embedding dimension
        
        Args:
            hidden_dims (list): List of hidden dimensions for the UNet layers
            time_embed_dim (int): Dimension of time embedding
        """
        super().__init__()
        
        self.time_embed_dim = time_embed_dim

        # Time embedding layers with increased capacity
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )

        # Input projection with wider initial representation
        self.input_proj = nn.Linear(18 + 1 + time_embed_dim, hidden_dims[0])  # 18 features + 1 noisy sample + time embedding
        
        # Encoder blocks with residual connections and group norm
        self.down_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.down_blocks.append(nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.GroupNorm(8, hidden_dims[i+1]),
                    nn.SiLU(),
                    nn.Linear(hidden_dims[i+1], hidden_dims[i+1]),
                    nn.GroupNorm(8, hidden_dims[i+1]),
                    nn.SiLU()
                ),
                nn.Linear(hidden_dims[i], hidden_dims[i+1])  # Skip connection
            ]))
            
        # Middle block with self-attention
        mid_dim = hidden_dims[-1]
        self.mid_block = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.GroupNorm(8, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.GroupNorm(8, mid_dim),
            nn.SiLU()
        )
        
        # Decoder blocks with enhanced skip connections
        self.up_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)-1, 0, -1):
            self.up_blocks.append(nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dims[i] + hidden_dims[i-1], hidden_dims[i-1]),
                    nn.GroupNorm(8, hidden_dims[i-1]),
                    nn.SiLU(),
                    nn.Linear(hidden_dims[i-1], hidden_dims[i-1]),
                    nn.GroupNorm(8, hidden_dims[i-1]),
                    nn.SiLU()
                ),
                nn.Linear(hidden_dims[i] + hidden_dims[i-1], hidden_dims[i-1])  # Skip connection
            ]))
            
        # Output projection with two heads for better variance modeling
        self.output_mean = nn.Linear(hidden_dims[0], 1)
        self.output_scale = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
            nn.SiLU(),
            nn.Linear(hidden_dims[0] // 2, 1),
            nn.Softplus()
        )

    def get_timestep_embedding(self, t):
        """
        Create sinusoidal time embedding with higher dimensionality.
        
        Args:
            t (torch.Tensor): Time steps tensor of shape (batch_size,)
            
        Returns:
            torch.Tensor: Time embedding of shape (batch_size, time_embed_dim)
        """
        half_dim = self.time_embed_dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :].to(t.device)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=1)
        return self.time_mlp(embeddings)

    def forward(self, x, y_t, t):
        """
        Forward pass of the UNet model with improved variance handling.
        
        Args:
            x (torch.Tensor): Input features of shape (batch_size, dim_x)
            y_t (torch.Tensor): Noisy samples at time t of shape (batch_size,)
            t (torch.Tensor): Time steps of shape (batch_size,)
            
        Returns:
            torch.Tensor: Predicted score of shape (batch_size,)
        """
        t_emb = self.get_timestep_embedding(t)
        h = torch.cat([x, y_t.unsqueeze(1), t_emb], dim=1)
        h = self.input_proj(h)
        
        # Store skip connections
        skips = []
        
        # Encoder with residual connections
        for down, down_skip in self.down_blocks:
            skips.append(h)
            h = down(h) + down_skip(h)
            
        # Middle
        h = self.mid_block(h)
        
        # Decoder with enhanced skip connections
        for up, up_skip in self.up_blocks:
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = up(h) + up_skip(h)
            
        # Output mean and scale components
        mean = self.output_mean(h).squeeze(1)
        scale = self.output_scale(h).squeeze(1)
        
        # Combine for final score prediction
        return mean * scale