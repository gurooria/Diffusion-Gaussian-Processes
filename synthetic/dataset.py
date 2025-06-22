import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
import torch

## Generates 1D synthetic data with heteroscedastic noise
def sin_1d(n_samples:int, seed:int=None) -> tuple[np.ndarray, np.ndarray]:
    # Initialisations
    x = np.zeros(n_samples, dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    if seed is not None:
        np.random.seed(seed)

    # True underlying distribution
    def f(x): # True function
        return 1 + np.sin(2 * np.pi * x)
    def noise_std(x): # Heteroscedastic noise
        return 0.5 + 0.4 * np.cos(6 * np.pi * x)
    
    # Generate random x values
    x[:] = np.random.rand(n_samples) # sample locations from standard uniform distribution
    y[:] = f(x) + np.random.randn(n_samples) * noise_std(x) # add heteroscedastic Gaussian noise
    
    return x, y


## Diffusion dataset class
class DiffusionDataset(Dataset):
    def __init__(self, dataset_name:str, n_samples:int, x_normalise:bool=True, y_normalise:bool=True, seed:int=None):
        # Initialisations
        self.dataset_name = dataset_name
        self.x_normalise = x_normalise
        self.y_normalise = y_normalise
        self.x_scaler = None
        self.y_scaler = None

        # Load raw dataset as numpy arrays
        if dataset_name == 'sin_1d':
            self.X, self.y = sin_1d(n_samples, seed)
        else: # UCI datasets
            self.X, self.y = fetch_openml(dataset_name, version=1, return_X_y=True, as_frame=False)

        # Reshape X & y to 2D if they are 1D
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)
        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)
        
        # Normalise
        if x_normalise:
            self.x_scaler = StandardScaler()
            self.X = self.x_scaler.fit_transform(self.X)
        if y_normalise:
            self.y_scaler = StandardScaler()
            self.y = self.y_scaler.fit_transform(self.y)
        
        # Convert to torch tensors with float32 dtype
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx:int) -> dict:
        return {'X': self.X[idx], 'y': self.y[idx]}
    
    def get_scalers(self) -> tuple[StandardScaler, StandardScaler]:
        return self.x_scaler, self.y_scaler



## Create dataloader
def get_dataloader(dataset, batch_size:int=32, train_ratio:float=0.8, val_ratio:float=0.1, shuffle:bool=True, seed:int=None) -> tuple[DataLoader, DataLoader, DataLoader]:
    # Initialise
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

    return train_loader, test_loader, val_loader