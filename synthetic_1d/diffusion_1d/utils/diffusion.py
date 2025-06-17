import torch

def cosine_schedule(T: int, s: float = 0.008, max_beta: float = 0.999, min_alpha: float = 0.001) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cosine schedule for the diffusion process.

    Args:
        n_steps (int): Number of steps in the diffusion process
        s (float): Small constant to avoid division by zero
    Returns:
        betas (torch.Tensor): Beta schedule for the diffusion process
        alphas (torch.Tensor): Alpha schedule for the diffusion process
        alphas_cumprod (torch.Tensor): Cumulative product of alphas
    """

    # Initialise for time steps 0 to T
    t = torch.linspace(0, T, T + 1)
    betas = torch.zeros(T + 1)
    alphas = torch.ones(T + 1)
    alphas_cumprod = torch.ones(T + 1)

    # Compute the values
    alphas_cumprod[:] = torch.cos((t / T + s) / (1 + s) * torch.pi / 2) ** 2  # length T+1
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalise by first element
    betas[1:] = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]  # length T+1, first element is 0
    alphas[1:] = 1 - betas[1:]  # length T+1, first element is 1

    return torch.clamp(betas, 0, max_beta), torch.clamp(alphas, min_alpha, 1.0), alphas_cumprod

class Diffusion:
    """
    Diffusion class for the conditional diffusion process.
    """
    def __init__(self, T: int, device: torch.device = None):
        """
        Args:
            T (int): Number of steps in the diffusion process
            device (torch.device): Device to run the diffusion process on
        """
        # Initialise basic inputs
        self.T = T
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

        # Get schedule parameters
        self.betas, self.alphas, self.alphas_cumprod = cosine_schedule(self.T)

        # Get computations for the posterior q
        self.posterior_variance = self.betas[1:] * (1 - self.alphas_cumprod[:-1]) / (1 - self.alphas_cumprod[1:])
        self.posterior_mean_coef1 = self.betas[1:] * torch.sqrt(self.alphas_cumprod[:-1]) / (1 - self.alphas_cumprod[1:])
        self.posterior_mean_coef2 = torch.sqrt(self.alphas[1:]) * (1 - self.alphas_cumprod[:-1]) / (1 - self.alphas_cumprod[1:])

    def forward_q(self, y_0: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the forward distribution q(y_t | y_0).

        Args:
            y_0 (torch.Tensor): The clean samples across the batch (shape: (batch_size,)).
            t (int): The time step to noise the clean sample to across the batch.

        Returns:
            mean (torch.Tensor): The means of the forward distribution across the batch (shape: (batch_size,)).
            var (torch.Tensor): The variances of the forward distribution across the batch (shape: (batch_size,)).
        """
        mean = torch.zeros_like(y_0)
        var = torch.zeros_like(y_0)
        mean[:] = torch.sqrt(self.alphas_cumprod[t]) * y_0
        var[:] = (1 - self.alphas_cumprod[t]) * torch.ones_like(y_0)

        return mean, var
    
    def q_sample(self, y_0: torch.Tensor, t: int, noise: torch.Tensor) -> torch.Tensor:
        """
        Sample from the forward distribution q(y_t | y_0).

        Args:
            y_0 (torch.Tensor): The clean samples across the batch (shape: (batch_size,)).
            t (int): The time step to noise the clean sample to across the batch.
            noise (torch.Tensor): The noise to add to the clean sample across the batch (shape: (batch_size,)).

        Returns:
            y_t (torch.Tensor): The noisy samples across the batch (shape: (batch_size,)).
        """
        mean, var = self.forward_q(y_0, t)
        return mean + noise * torch.sqrt(var)
    
    def reverse_q(self, y_0: torch.Tensor, y_t1: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the posterior distribution q(y_t | y_{t+1}, t).

        Args:
            y_0 (torch.Tensor): The clean samples across the batch (shape: (batch_size,)).
            y_t1 (torch.Tensor): The noisy samples across the batch (shape: (batch_size,)).
            t (int): The time step to move to for y_t.

        Returns:
            mean (torch.Tensor): The means of the posterior distribution across the batch (shape: (batch_size,)).
            var (torch.Tensor): The variances of the posterior distribution across the batch (shape: (batch_size,)).
        """
        mean = torch.zeros_like(y_0)
        var = torch.zeros_like(y_0)
        mean[:] = self.posterior_mean_coef1[t] * y_0 + self.posterior_mean_coef2[t] * y_t1
        var[:] = self.posterior_variance[t] * torch.ones_like(y_0)
        return mean, var