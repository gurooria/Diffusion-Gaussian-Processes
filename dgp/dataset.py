import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
# from diffusion import Diffusion

class DiffusionDataset(Dataset):
    def __init__(self, dataset_name:str, x_normalise:bool=True, y_normalise:bool=True, seed:int=None):
        """
        Dataset class for diffusion Gaussian processes.

        Args:
            dataset_name (str): Name of the dataset to load.
            x_normalise (bool): Whether to normalise x.
            y_normalise (bool): Whether to normalise y.
            seed (int): Random seed.
            augment (bool): Whether to augment the dataset.
            T (int): Number of time steps for augmentation.

        Returns:
            x (torch.Tensor): Input features of shape (n_samples, n_features).
            y (torch.Tensor): Target values of shape (n_samples, 1).
        """
        # Initialisations
        self.dataset_name = dataset_name
        self.x_normalise = x_normalise
        self.y_normalise = y_normalise
        self.x_scaler = None
        self.y_scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if seed is not None: # seed for generating sin_1d dataset
            np.random.seed(seed)

        # Load raw dataset as numpy arrays of 2D shape
        if dataset_name == 'sin_1d':
            x, mean, std = self.sin_1d()
            self.x = x
            self.y = (mean + std * np.random.randn(x.shape[0]))
        else: # UCI datasets
            self.x, self.y = fetch_openml(dataset_name, version=1, return_X_y=True, as_frame=False)
        
        # Reshape x and y to 2D if they are 1D
        if self.x.ndim == 1:
            print(f"Reshaping x from 1D to 2D, shape: {self.x.shape} -> {self.x.shape[0],1}")
            self.x = self.x.reshape(-1,1)
        if self.y.ndim == 1:
            print(f"Reshaping y from 1D to 2D, shape: {self.y.shape} -> {self.y.shape[0],1}")
            self.y = self.y.reshape(-1,1)
        assert self.x.ndim == 2 and self.y.ndim == 2, f"x and y must be 2D arrays, but got {self.x.ndim} and {self.y.ndim}"

        # Normalisation to mean 0 and std 1
        if self.x_normalise:
            self.x_scaler = StandardScaler()
            self.x = self.x_scaler.fit_transform(self.x)
        if self.y_normalise:
            self.y_scaler = StandardScaler()
            self.y = self.y_scaler.fit_transform(self.y)

        # Convert to torch tensors with float32 dtype
        self.x = torch.tensor(self.x, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(self.y, dtype=torch.float32, device=self.device)

    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx:int) -> dict:
        return {'x': self.x[idx], 'y': self.y[idx]}

    @staticmethod
    def sin_1d(n_samples:int=5000) -> tuple[np.ndarray, np.ndarray]: 
        # True underlying distribution
        def f(x):
            return 1 + np.sin(2 * np.pi * x)
        def std(x):
            return 0.5 + 0.4 * np.cos(6 * np.pi * x)
        
        # Generate random x values
        x = np.random.rand(n_samples)
        return x, f(x), std(x) # shape (n_samples,)


def get_dataloader(dataset:DiffusionDataset, batch_size:int, train_ratio:float=0.8, val_ratio:float=0.1, shuffle:bool=True, seed:int=None) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get dataloaders for training, validation and test sets.

    Args:
        dataset (DiffusionDataset): Dataset to split.
        batch_size (int): Batch size.
        train_ratio (float): Ratio of training set.
        val_ratio (float): Ratio of validation set.
        shuffle (bool): Whether to shuffle the training dataset.
        seed (int): Random seed for splitting the dataset.

    Returns:
        train_loader (DataLoader): Dataloader for training set.
        val_loader (DataLoader): Dataloader for validation set.
        test_loader (DataLoader): Dataloader for test set.
    """
    # Initialisations
    if seed is not None: # seed for splitting the dataset
        torch.manual_seed(seed)
    
    # Split dataset
    n_samples = len(dataset)
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    n_test = n_samples - n_train - n_val
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,[n_train, n_val, n_test]) # splits into training and validation datasets of desired ratio
    
    # Create dataloaders: divide each dataset into batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader