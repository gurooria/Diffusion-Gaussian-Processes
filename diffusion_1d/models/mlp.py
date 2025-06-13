# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ScoreModel(nn.Module):
#     def __init__(self, hidden_dims=[32, 64, 64], time_embed_dim=64):
#         """
#         Model for predicting conditional score in diffusion models.
#         With improvements for better variance modeling:
#         - Increased network capacity (wider and deeper)
#         - Added residual connections
#         - Group normalization instead of batch norm for better stability
#         - Increased time embedding dimension
        
#         Args:
#             hidden_dims (list): List of hidden dimensions for the UNet layers
#             time_embed_dim (int): Dimension of time embedding
#         """
#         super().__init__()
        
#         self.time_embed_dim = time_embed_dim

#         # Time embedding layers with increased capacity
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_embed_dim, time_embed_dim * 2),
#             nn.SiLU(),
#             nn.Linear(time_embed_dim * 2, time_embed_dim * 2),
#             nn.SiLU(),
#             nn.Linear(time_embed_dim * 2, time_embed_dim)
#         )

#         # Input projection with wider initial representation
#         self.input_proj = nn.Linear(2 + time_embed_dim, hidden_dims[0])
        
#         # Encoder blocks with residual connections and group norm
#         self.down_blocks = nn.ModuleList()
#         for i in range(len(hidden_dims)-1):
#             self.down_blocks.append(nn.ModuleList([
#                 nn.Sequential(
#                     nn.Linear(hidden_dims[i], hidden_dims[i+1]),
#                     nn.GroupNorm(8, hidden_dims[i+1]),
#                     nn.SiLU(),
#                     nn.Linear(hidden_dims[i+1], hidden_dims[i+1]),
#                     nn.GroupNorm(8, hidden_dims[i+1]),
#                     nn.SiLU()
#                 ),
#                 nn.Linear(hidden_dims[i], hidden_dims[i+1])  # Skip connection
#             ]))
            
#         # Middle block with self-attention
#         mid_dim = hidden_dims[-1]
#         self.mid_block = nn.Sequential(
#             nn.Linear(mid_dim, mid_dim),
#             nn.GroupNorm(8, mid_dim),
#             nn.SiLU(),
#             nn.Linear(mid_dim, mid_dim),
#             nn.GroupNorm(8, mid_dim),
#             nn.SiLU()
#         )
        
#         # Decoder blocks with enhanced skip connections
#         self.up_blocks = nn.ModuleList()
#         for i in range(len(hidden_dims)-1, 0, -1):
#             self.up_blocks.append(nn.ModuleList([
#                 nn.Sequential(
#                     nn.Linear(hidden_dims[i] + hidden_dims[i-1], hidden_dims[i-1]),
#                     nn.GroupNorm(8, hidden_dims[i-1]),
#                     nn.SiLU(),
#                     nn.Linear(hidden_dims[i-1], hidden_dims[i-1]),
#                     nn.GroupNorm(8, hidden_dims[i-1]),
#                     nn.SiLU()
#                 ),
#                 nn.Linear(hidden_dims[i] + hidden_dims[i-1], hidden_dims[i-1])  # Skip connection
#             ]))
            
#         # Output projection with two heads for better variance modeling
#         self.output_mean = nn.Linear(hidden_dims[0], 1)
#         self.output_scale = nn.Sequential(
#             nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
#             nn.SiLU(),
#             nn.Linear(hidden_dims[0] // 2, 1),
#             nn.Softplus()
#         )

#     def get_timestep_embedding(self, t):
#         """
#         Create sinusoidal time embedding with higher dimensionality.
        
#         Args:
#             t (torch.Tensor): Time steps tensor of shape (batch_size,)
            
#         Returns:
#             torch.Tensor: Time embedding of shape (batch_size, time_embed_dim)
#         """
#         half_dim = self.time_embed_dim // 2
#         embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
#         embeddings = t[:, None] * embeddings[None, :].to(t.device)
#         embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=1)
#         return self.time_mlp(embeddings)

#     def forward(self, x, y_t, t):
#         """
#         Forward pass of the UNet model with improved variance handling.
        
#         Args:
#             x (torch.Tensor): Input locations of shape (batch_size,)
#             y_t (torch.Tensor): Noisy samples at time t of shape (batch_size,)
#             t (torch.Tensor): Time steps of shape (batch_size,)
            
#         Returns:
#             torch.Tensor: Predicted score of shape (batch_size,)
#         """
#         # Get time embedding
#         t_emb = self.get_timestep_embedding(t)
        
#         # Combine inputs
#         h = torch.cat([x.unsqueeze(1), y_t.unsqueeze(1), t_emb], dim=1)
#         h = self.input_proj(h)
        
#         # Store skip connections
#         skips = []
        
#         # Encoder with residual connections
#         for down, down_skip in self.down_blocks:
#             skips.append(h)
#             h = down(h) + down_skip(h)
            
#         # Middle
#         h = self.mid_block(h)
        
#         # Decoder with enhanced skip connections
#         for up, up_skip in self.up_blocks:
#             skip = skips.pop()
#             h = torch.cat([h, skip], dim=1)
#             h = up(h) + up_skip(h)
            
#         # Output mean and scale components
#         mean = self.output_mean(h).squeeze(1)
#         scale = self.output_scale(h).squeeze(1)
        
#         # Combine for final score prediction
#         return mean * scale

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMLPBlock(nn.Module):
    """
    A single residual block:
        input ─► Linear(d, d) ─► GroupNorm ─► SiLU ─►
                  Linear(d, d) ─► GroupNorm ─► SiLU
           │                                      ▲
           └────────────── skip-Linear(d, d) ────┘
    output = main_path(input) + skip(input)
    """
    def __init__(self, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
        )
        self.skip = nn.Linear(dim, dim)

    def forward(self, x):
        return self.main(x) + self.skip(x)


class ScoreModel(nn.Module):
    """
    Pure-MLP “equivalent” of the original ScoreModel.  It uses:
      - The same __init__(hidden_dims=[32,64,64], time_embed_dim=64) signature
      - The same sinusoidal + MLP time embedding
      - An input projection from (x, y_t, t_emb) → hidden_dims[0]
      - A straight stack of ResidualMLPBlock(hidden_dims[0]) for len(hidden_dims) layers
      - Two output heads (mean, scale) exactly as before
      - forward(x, y_t, t) returns mean * scale
    """
    def __init__(self, hidden_dims=[32, 64, 64], time_embed_dim=64):
        """
        Args:
          hidden_dims     : exactly the same list you passed before (e.g. [32,64,64])
                            We only end up using hidden_dims[0] as the MLP’s width.
                            (You can still change hidden_dims[0] to widen or narrow.)
          time_embed_dim  : same as before (default = 64)
        """
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.hidden_dim = hidden_dims[0]
        self.num_blocks = len(hidden_dims)   # we will stack this many residual blocks

        # 1) Time embedding MLP (identical to your original ScoreModel)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        # 2) Input projection: [ x, y_t, time_emb ] → hidden_dims[0]
        self.input_proj = nn.Linear(2 + time_embed_dim, self.hidden_dim)

        # 3) A stack of `num_blocks` ResidualMLPBlock’s, each of dimension = hidden_dim
        self.res_blocks = nn.ModuleList([
            ResidualMLPBlock(self.hidden_dim) for _ in range(self.num_blocks)
        ])

        # 4) Two output heads: mean & scale (exactly as in your UNet version)
        self.output_mean = nn.Linear(self.hidden_dim, 1)
        self.output_scale = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Softplus()
        )

    def get_timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal + small MLP for time embedding (exactly as before).
        Input:
          t   : (batch_size,) of scalars
        Output:
          (batch_size, time_embed_dim)
        """
        half_dim = self.time_embed_dim // 2
        # Build the frequency schedule
        freq = torch.exp(
            -torch.log(torch.tensor(10000.0)) 
            * torch.arange(half_dim, device=t.device) 
            / (half_dim - 1)
        )  # shape = (half_dim,)
        # Multiply: (batch_size, 1) * (1, half_dim) → (batch_size, half_dim)
        angles = t.unsqueeze(1) * freq.unsqueeze(0)
        sin_cos = torch.cat([angles.sin(), angles.cos()], dim=1)  # (batch_size, time_embed_dim)
        return self.time_mlp(sin_cos)  # → (batch_size, time_embed_dim)

    def forward(self, x: torch.Tensor, y_t: torch.Tensor, t: torch.Tensor):
        """
        Forward pass:
          x   : (batch_size,)         # same as before
          y_t : (batch_size,)         # same as before
          t   : (batch_size,)         # same as before
        Returns:
          score: (batch_size,) = mean(x,y_t,t) * scale(x,y_t,t)
        """
        # 1) Build time embedding
        t_emb = self.get_timestep_embedding(t)  # (batch, time_embed_dim)

        # 2) Concatenate [ x, y_t, t_emb ] → (batch, 2 + time_embed_dim)
        h = torch.cat([x.unsqueeze(1), y_t.unsqueeze(1), t_emb], dim=1)

        # 3) Project to hidden_dim
        h = self.input_proj(h)  # (batch, hidden_dim)

        # 4) Pass through a straight stack of residual blocks
        for block in self.res_blocks:
            h = block(h)  # (batch, hidden_dim)

        # 5) Compute “mean” and “scale” heads, then multiply
        mean = self.output_mean(h).squeeze(1)   # (batch,)
        scale = self.output_scale(h).squeeze(1) # (batch,)
        return mean * scale