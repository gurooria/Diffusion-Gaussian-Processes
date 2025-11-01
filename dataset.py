from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

## Generate synthetic datasets
def synthetic(x, name):
    """
    Generate synthetic 1D toy functions.

    Args:
        x (torch.Tensor): Input features of shape (n_samples, 1).
        name (str): Name of the synthetic function.

    Returns:
        mean (torch.Tensor): Mean of the synthetic dataset of shape (n_samples, 1).
        sigma (torch.Tensor): Standard deviation of the synthetic dataset of shape (n_samples, 1).
    """
    # Sinusoidal function with heteroscedastic Gaussian noise
    if name == 'sinusoidal':
        def f(x): # mean
            return 1 + torch.sin(2 * torch.pi * x)
        def sigma(x): # std
            return 0.5 + 0.4 * torch.cos(6 * torch.pi * x)
        return f(x), sigma(x)

    # Piecewise linear function with log-normal noise
    elif name == 'piecewise':
        # Define breakpoints and slopes of piecewise linear function
        breakpoints = torch.tensor([0.0, 0.3, 0.6, 0.8, 1.0])
        slopes = torch.tensor([6.0, -12.0, 15.0, 0])  # num_pieces = 3
        intercepts = torch.zeros_like(slopes)
        for i in range(1, len(slopes)):
            intercepts[i] = (slopes[i - 1] * breakpoints[i] + intercepts[i - 1]) - slopes[i] * breakpoints[i]
        def f(x):
            idx = torch.bucketize(x.squeeze(-1), breakpoints[1:], right=False)
            return (slopes[idx] * x.squeeze(-1) + intercepts[idx]).reshape(x.shape)
        def sigma(x):
            return torch.full_like(x, 0.5)  # sigma = 0.5
        return f(x), sigma(x)

    ## Invalid function name
    else:
        raise ValueError(f"Invalid synthetic function name: {name}, choose from 'sinusoidal' or 'piecewise'")

## Dataset class for diffusion models
class DiffusionDataset(Dataset):
    """
    Raw dataset class for diffusion Gaussian processes.

    Args:
        dataset_name (str): Name of the dataset to load.
        x_normalise (bool): Whether to normalise x.
        y_normalise (bool): Whether to normalise y.

    Returns:
        x (torch.Tensor): Input features of shape (n_samples, n_features).
        y (torch.Tensor): Target values of shape (n_samples, 1).
    """
    def __init__(self, dataset_name, x_normalise=False, y_normalise=False):
        # Initialisations
        self.dataset_name = dataset_name
        self.x_scaler = None
        self.y_scaler = None

        # Load raw datasets -> x, y = tensors
        if self.dataset_name == 'sinusoidal':
            torch.manual_seed(42) # fixed seed for reproducibility
            self.x = torch.rand(5000, dtype=torch.float32)
            mean, sigma = synthetic(x=self.x, name='sinusoidal')
            mean = torch.tensor(mean, dtype=torch.float32)
            sigma = torch.tensor(sigma, dtype=torch.float32)
            self.y = mean + sigma * torch.randn(mean.shape[0], dtype=torch.float32)
        elif self.dataset_name == 'piecewise':
            torch.manual_seed(42) # fixed seed for reproducibility
            self.x = torch.rand(5000, dtype=torch.float32)
            mean, sigma = synthetic(x=self.x, name='piecewise')
            mean = torch.tensor(mean, dtype=torch.float32)
            sigma = torch.tensor(sigma, dtype=torch.float32)
            self.y = mean * torch.exp(sigma * torch.randn(mean.shape[0], dtype=torch.float32))
        else: # UCI datasets
            self.x = uci_Dataset(self.dataset_name).x
            self.y = uci_Dataset(self.dataset_name).y

        # Reshape x and y to 2D if they are 1D
        if self.x.ndim == 1:
            print(f"Reshaping x from 1D to 2D, shape: {self.x.shape} -> {self.x.shape[0],1}")
            self.x = self.x.reshape(-1,1)
        if self.y.ndim == 1:
            print(f"Reshaping y from 1D to 2D, shape: {self.y.shape} -> {self.y.shape[0],1}")
            self.y = self.y.reshape(-1,1)
        
        # Normalisation to mean 0 and std 1
        if x_normalise:
            self.x_scaler = StandardScaler()
            self.x = self.x_scaler.fit_transform(self.x)
        if y_normalise:
            self.y_scaler = StandardScaler()
            self.y = self.y_scaler.fit_transform(self.y)

        # Convert to torch tensors of desired format
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

        # Check data formats before returning
        assert self.x.ndim == 2 and self.y.ndim == 2, f"x and y must be 2D tensors, got {self.x.ndim} and {self.y.ndim}"
        assert type(self.x) == torch.Tensor and type(self.y) == torch.Tensor, f"x and y must be tensors, got {type(self.x)} and {type(self.y)}"
        assert self.x.dtype == torch.float32 and self.y.dtype == torch.float32, f"x and y must be float32 tensors, got {self.x.dtype} and {self.y.dtype}"

    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx:int) -> dict:
        return {'x': self.x[idx], 'y': self.y[idx]}

    def get_scalers(self) -> tuple[StandardScaler, StandardScaler]:
        return self.x_scaler, self.y_scaler


## Dataloader function for diffusion models, batches data for training, validation and test sets
def get_dataloader(dataset:DiffusionDataset, batch_size:int, train_ratio:float=0.8, test_ratio:float=0.1, shuffle:bool=True, seed:int=None) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get dataloaders for training, validation and test sets.

    Args:
        dataset (DiffusionDataset): Dataset to split.
        batch_size (int): Batch size.
        train_ratio (float): Ratio of training set.
        test_ratio (float): Ratio of test set.
        shuffle (bool): Whether to shuffle the training dataset.
        seed (int): Random seed for splitting the dataset.

    Returns:
        train_loader (DataLoader): Dataloader for training set.
        val_loader (DataLoader): Dataloader for validation set.
        test_loader (DataLoader): Dataloader for test set.
    """
    # Initialisations
    n_samples = len(dataset)
    n_train = int(train_ratio * n_samples)
    n_test = int(test_ratio * n_samples)
    n_val = n_samples - n_train - n_test

    # Split dataset
    if seed is not None: # seed for splitting the dataset
        torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,[n_train, n_val, n_test]) # splits into training and validation datasets of desired ratio

    # Create dataloaders: divide each dataset into batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader