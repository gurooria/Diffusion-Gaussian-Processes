import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

## Function to generate data from the desired distribution
def sin_1d(n_samples:int, seed:int=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate data from the true function with heteroscedastic noise.
    
    Args:
        n_samples (int): Number of data points to generate
        seed (int): Random seed for reproducibility
    Returns:
        x (np.ndarray): Input locations (n_samples)
        y (np.ndarray): Function values with noise (n_samples)
    """
    # Initialisation
    if seed is not None:
        np.random.seed(seed)
    x = np.zeros(n_samples, dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    # True underlying distribution
    def f(x): # True function
        return 1 + np.sin(2 * np.pi * x)
    def noise_std(x): # Heteroscedastic noise
        return 0.1 + 0.4 * np.cos(3 * np.pi * x)**2
    
    # Generate random x values
    x[:] = np.random.rand(n_samples) # sample from standard uniform distribution
    y[:] = f(x) + np.random.randn(n_samples) * noise_std(x) # add heteroscedastic Gaussian noise
    
    return x, y

## Create dataset class & dataloader
class DiffusionDataset(Dataset):
    def __init__(self, dataset_name:str='sin_1d', n_samples:int=1000, seed:int=None):
        """
        Args:
            dataset_name (str): Name of the dataset
            n_samples (int): Number of data points to generate
            seed (int): Random seed for reproducibility
        """
        if dataset_name == 'sin_1d':
            self.x, self.y = sin_1d(n_samples, seed)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
    
    def __len__(self) -> int: # called by len(dataset)
        # Number of samples
        return len(self.x)
    
    def __getitem__(self, idx:int) -> dict: # called by dataset[idx]
        # Fetch sample at given index
        return {'x': self.x[idx], 'y': self.y[idx]}
    
## Create dataloader
def get_dataloader(dataset:Dataset, train_ratio:float=0.8, val_ratio:float=0.1, batch_size:int=32, shuffle:bool=True, seed:int=None) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, test, and validation dataloaders from dataset.
    
    Args:
        dataset (Dataset): Dataset object
        train_ratio (float): Ratio of training data to validation data
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the training data
        seed (int): Random seed for reproducibility
    Returns:
        train_loader (DataLoader): Training dataloader
        test_loader (DataLoader): Test dataloader
        val_loader (DataLoader): Validation dataloader
    """
    # Initialise
    if seed is not None:
        np.random.seed(seed)

    # Split dataset into training and validation sets
    n_samples = len(dataset)
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    n_test = n_samples - n_train - n_val
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,[n_train, n_val, n_test]) # splits into training and validation datasets of desired ratio

    # Create dataloaders: divide each dataset into batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader, val_loader