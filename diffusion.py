import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

class Diffusion:
    def __init__(self, T:int):
        # Initialisations
        self.T = T
        self.betas, self.alphas, self.alphas_cumprod = self.cosine_schedule(self.T)

    def forward_q(self, y_0:torch.Tensor, t:int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the mean and variance of the forward distribution q(y_t | y_0).

        Args:
            y_0 (torch.Tensor): The clean samples across the batch.
            t (int): The time step  to noise the clean sample to across the batch.

        Returns:
            mean (torch.Tensor): The means of the forward distribution across the batch.
            var (torch.Tensor): The variances of the forward distribution across the batch.
        """
        # Initialisations
        mean = torch.zeros_like(y_0)
        var = torch.zeros_like(y_0)

        # Compute mean and variance at the given time step t
        mean[:] = torch.sqrt(self.alphas_cumprod[t]) * y_0
        var[:] = (1 - self.alphas_cumprod[t]) * torch.ones_like(y_0)
        assert mean.shape == var.shape == y_0.shape, f'mean and var have different shapes: {mean.shape} != {var.shape} != {y_0.shape}'

        return mean, var
    
    def sample_q(self, y_0:torch.Tensor, t:torch.Tensor, noise:torch.Tensor) -> torch.Tensor:
        """
        Sample from the forward distribution q(y_t | y_0).

        Args:
            y_0 (torch.Tensor): The clean samples across the batch.
            t (torch.Tensor): The time steps to noise the clean samples to across the batch.
            noise (torch.Tensor): The noise to add to the clean sample across the batch.

        Returns:
            y_t (torch.Tensor): The noisy samples across the batch.
        """
        # Intialisations
        y_t = torch.zeros_like(y_0)
        assert y_0.shape == noise.shape, f'y_0 and noise have different shapes: {y_0.shape} != {noise.shape}'

        # Compute y_t
        mean, var = self.forward_q(y_0, t)
        y_t[:] = mean + torch.sqrt(var) * noise
        assert y_t.shape == y_0.shape, f'y_t and y_0 have different shapes: {y_t.shape} != {y_0.shape}'
        return y_t
    
    def plot_q_samples(self, x:torch.Tensor, y_0:torch.Tensor, t:int) -> None:
        """
        Plot the samples with the distribution from the forward distribution q(y_t | y_0).

        Args:
            x (torch.Tensor): The input features.
            y (torch.Tensor): The target values.
            t (int): The time step to plot the samples from.
        """
        # Initialisations
        mean, var = self.forward_q(y_0, t)
        y_t = self.sample_q(y_0, t, torch.randn_like(y_0))

        # Plot
        plt.figure(figsize=(5, 3))
        plt.title(f'q(y_t | y_0) with samples, t={t}')
        plt.xlabel('sample index', fontsize=12)
        plt.ylabel('y_t', fontsize=12)
        plt.plot(range(len(mean)), mean[:,0], label='q_mean', c='b')
        plt.fill_between(range(len(mean)), mean[:,0] - var[:,0]**0.5, mean[:,0] + var[:,0]**0.5, alpha=0.2, color='b')
        plt.scatter(range(len(y_t)), y_t[:,0], s=3, c='r', label='y_t')
        plt.legend(fontsize=11)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.grid(alpha=0.3)
        plt.show()
    
    @staticmethod
    def cosine_schedule(T:int, s:float=0.008, max_beta:float=0.999, min_alpha:float=0.001, plot:bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the cosine noising schedule for the diffusion process.

        Args:
            T (int): Number of diffusion time steps.
            s (float): Small offset for the cosine schedule to avoid division by zero.
            max_beta (float): Maximum beta value to avoid numerical instability.
            min_alpha (float): Minimum alpha value to avoid numerical instability.

        Returns:
            betas (torch.Tensor): Beta values for the diffusion process (shape: (T+1,)).
            alphas (torch.Tensor): Alpha values for the diffusion process (shape: (T+1,)).
            alphas_cumprod (torch.Tensor): Cumulative product of alpha values for the diffusion process (shape: (T+1,)).
        """
        # Initialise for time steps 0 to T, shape (T+1,) for all tensors
        t = torch.linspace(0, T, T + 1)
        betas = torch.zeros(T + 1)
        alphas = torch.ones(T + 1)
        alphas_cumprod = torch.ones(T + 1)
        
        # Compute schedule values
        alphas_cumprod[:] = torch.cos((t / T + s) / (1 + s) * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalise by first element
        betas[1:] = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]  # first element is 0
        alphas[1:] = 1 - betas[1:]  # first element is 1

        # Visualise schedule
        if plot:
            plt.figure(figsize=(5,3))
            plt.grid(alpha=0.3)
            plt.plot(betas, label='beta', c='b')
            plt.plot(alphas, label='alpha', c='r')
            plt.plot(alphas_cumprod, label='alphas_cumprod', c='g')
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            plt.legend(fontsize=11)
            plt.xlabel('time step', fontsize=12)
            plt.ylabel('value', fontsize=12)
            plt.title(f'Cosine Noising Schedule, T={T}', fontsize=14)
            plt.show()

        return torch.clamp(betas, 0, max_beta), torch.clamp(alphas, min_alpha, 1.0), alphas_cumprod