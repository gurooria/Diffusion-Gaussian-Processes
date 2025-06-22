import torch

def cosine_schedule(T: int, s: float = 0.008, max_beta: float = 0.999, min_alpha: float = 0.001) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Initialise for time steps 0 to T
    t = torch.linspace(0, T, T + 1) # shape: (T+1,)
    betas = torch.zeros(T + 1) # shape: (T+1,)
    alphas = torch.ones(T + 1) # shape: (T+1,)
    alphas_cumprod = torch.ones(T + 1) # shape: (T+1,)

    # Compute schedule values
    alphas_cumprod[:] = torch.cos((t / T + s) / (1 + s) * torch.pi / 2) ** 2  # length T+1
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalise by first element
    betas[1:] = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]  # length T+1, first element is 0
    alphas[1:] = 1 - betas[1:]  # length T+1, first element is 1

    return torch.clamp(betas, 0, max_beta), torch.clamp(alphas, min_alpha, 1.0), alphas_cumprod

class Diffusion:
    def __init__(self, T: int, device: torch.device = None):
        # Initialisations
        self.T = T
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Get schedule parameters & coefficients for the posterior q distribution q(y_t | y_0)
        self.betas, self.alphas, self.alphas_cumprod = cosine_schedule(self.T)
        self.posterior_variance = self.betas[1:] * (1 - self.alphas_cumprod[:-1]) / (1 - self.alphas_cumprod[1:]) # shape: (T,)
        self.posterior_mean_coef1 = self.betas[1:] * torch.sqrt(self.alphas_cumprod[:-1]) / (1 - self.alphas_cumprod[1:]) # shape: (T,)
        self.posterior_mean_coef2 = torch.sqrt(self.alphas[1:]) * (1 - self.alphas_cumprod[:-1]) / (1 - self.alphas_cumprod[1:]) # shape: (T,)

    def forward_q(self, y_0: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the mean and variance of the forward distribution q(y_t | y_0).

        Args:
            y_0 (torch.Tensor): The clean samples across the batch (shape: (batch_size, 1)).
            t (int): The time step  to noise the clean sample to across the batch.

        Returns:
            mean (torch.Tensor): The means of the forward distribution across the batch (shape: (batch_size, 1)).
            var (torch.Tensor): The variances of the forward distribution across the batch (shape: (batch_size, 1)).
        """
        # Initialise
        mean = torch.zeros_like(y_0) # shape: (batch_size, 1)
        var = torch.zeros_like(y_0) # shape: (batch_size, 1)
        
        # Compute mean and variance
        mean[:] = torch.sqrt(self.alphas_cumprod[t]) * y_0
        var[:] = (1 - self.alphas_cumprod[t]) * torch.ones_like(y_0)

        return mean, var
    
    def q_sample(self, y_0: torch.Tensor, t: int, noise: torch.Tensor) -> torch.Tensor:
        """
        Sample from the forward distribution q(y_t | y_0).

        Args:
            y_0 (torch.Tensor): The clean samples across the batch (shape: (batch_size, 1)).
            t (int): The time step to noise the clean sample to across the batch.
            noise (torch.Tensor): The noise to add to the clean sample across the batch (shape: (batch_size, 1)).

        Returns:
            y_t (torch.Tensor): The noisy samples across the batch (shape: (batch_size, 1)).
        """
        mean, var = self.forward_q(y_0, t)
        return mean + noise * torch.sqrt(var)