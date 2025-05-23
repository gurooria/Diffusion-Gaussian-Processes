# train model
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from models.mlp import ScoreModel
from utils.diffusion import Diffusion

def train_score_mlp(
    train_loader,
    val_loader,
    hidden_dims: list[int] = [8, 16, 32],
    epoch_update: int = 10,
    num_epochs: int = 100,
    time_steps: int = 1000,
    learning_rate: float = 1e-4,
    device_type = 'cpu',
    save_path: str = None
):
    """
    Training function for the ScoreMLP model.
    
    Args:
        train_loader: DataLoader containing training data
        val_loader: DataLoader containing validation data
        num_epochs (int): Number of epochs to train for
        learning_rate (float): Learning rate for the optimizer
        device (str): Device to train on ('cuda' or 'cpu')
        save_path (str, optional): Path to save the best model
    """
    # Initialisations
    device = torch.device(device_type)
    diffusion = Diffusion(T=time_steps, device=device)
    model = ScoreModel(hidden_dims=hidden_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Initialise losses
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_epoch_loss = 0
        train_num_batches = 0
        for batch in train_loader:
            x = batch['x'].to(device) # (batch_size,)
            y0 = batch['y'].to(device) # (batch_size,)

            # Sample a random time step between 1 to T for each sample in the batch
            t = torch.randint(1, time_steps, (x.shape[0],), device=device)
            noise = torch.randn_like(y0, device=device)
            y_t = diffusion.q_sample(y0, t, noise)

            # Predict the score of the noisy sample
            score = model(x, y_t, t) # t is now the time step of the noisy sample
            loss = loss_fn(score, noise)
            
            # Backpropagate the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update the running loss
            train_epoch_loss += loss.item()
            train_num_batches += 1
        train_losses.append(train_epoch_loss / train_num_batches) # Compute the average loss for the epoch

        # Validation phase
        model.eval()
        val_epoch_loss = 0
        val_num_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y0 = batch['y'].to(device)
                t = torch.randint(1, time_steps, (x.shape[0],), device=device)
                noise = torch.randn_like(y0, device=device)
                y_t = diffusion.q_sample(y0, t, noise)

                # Compute the score
                score = model(x, y_t, t)
                loss = loss_fn(score, noise)

                # Update the running loss
                val_epoch_loss += loss.item()
                val_num_batches += 1
        val_losses.append(val_epoch_loss / val_num_batches) # Compute the average loss for the epoch

        # Save the best model
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            if save_path is not None:
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, save_path)
        
        # Print the loss for the epoch
        if epoch % epoch_update == 0:
            print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    
    return model, train_losses, val_losses