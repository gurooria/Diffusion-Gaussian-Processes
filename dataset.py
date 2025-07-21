from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

## True underlying distribution of synthetic 1D datasets
def sin_1d(x) -> tuple[np.ndarray, np.ndarray]: 
    # True underlying distribution
    def f(x): # mean
        return 1 + np.sin(2 * np.pi * x)
    def sigma(x): # std
        return 0.5 + 0.4 * np.cos(6 * np.pi * x)
    return f(x), sigma(x)

## Dataset class for diffusion models
class DiffusionDataset(Dataset):
    """
    Raw dataset class for diffusion Gaussian processes.

    Args:
        dataset_name (str): Name of the dataset to load.
        x_normalise (bool): Whether to normalise x.
        y_normalise (bool): Whether to normalise y.
        seed (int): Random seed.
        device (str): Device to use for the dataset.

    Returns:
        x (torch.Tensor): Input features of shape (n_samples, n_features).
        y (torch.Tensor): Target values of shape (n_samples, 1).
    """
    def __init__(self, dataset_name:str, x_normalise:bool=False, y_normalise:bool=False, device:str=None):
        # Initialisations
        self.dataset_name = dataset_name
        self.x_scaler = None
        self.y_scaler = None
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load raw datasets
        if self.dataset_name == 'sin_1d':
            np.random.seed(42) # fixed seed for reproducibility
            self.x = np.random.rand(4000)
            mean, sigma = sin_1d(self.x)
            self.y = mean + sigma * np.random.randn(mean.shape[0])
        else: # UCI datasets
            self.x, self.y = fetch_openml(dataset_name, version=1, return_X_y=True, as_frame=False)

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

        # Convert to torch tensors with float32 dtype
        self.x = torch.tensor(self.x, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(self.y, dtype=torch.float32, device=self.device)

        # Output checks for x and y
        assert self.x.ndim == 2 and self.y.ndim == 2, f"x and y must be 2D arrays, but got {self.x.ndim} and {self.y.ndim}"
        assert type(self.x) == torch.Tensor and type(self.y) == torch.Tensor, f"x and y must be torch.Tensor, but got {type(self.x)} and {type(self.y)}"

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