from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

class uciDataset(Dataset):
    def __init__(self, dataset_name:str, x_normalise:bool=True, y_normalise:bool=False):
        self.dataset_name = dataset_name
        self.x_normalise = x_normalise
        self.y_normalise = y_normalise
        self.x_scaler = None
        self.y_scaler = None

        # Fetch data
        self.X, self.y = fetch_openml(dataset_name, version=1, return_X_y=True, as_frame=False)

        # Normalise
        if self.x_normalise:
            self.x_scaler = StandardScaler()
            self.X = self.x_scaler.fit_transform(self.X)
        if self.y_normalise:
            self.y_scaler = StandardScaler()
            self.y = self.y_scaler.fit_transform(self.y.reshape(-1, 1)).flatten()

        # Convert to torch tensors with float64 dtype
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {'X': self.X[idx], 'y': self.y[idx]}
    
    def get_scalers(self):
        return self.x_scaler, self.y_scaler


def get_dataloader(dataset:uciDataset, batch_size:int=32, train_ratio:float=0.8, val_ratio:float=0.1, shuffle:bool=True, seed:int=None):
    # Initialise
    if seed is not None:
        np.random.seed(seed) # Controls randomness of splitting dataset

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