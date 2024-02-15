import numpy as np
from sklearn.linear_model import Ridge
import argparse
import torch
import wandb
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torchsummary import summary
from torchviz import make_dot

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

sys.path.append("C:/Users/sachi/SensoriumDecoding/src/utils")

from data import load_args, get_training_ds
from model import get_model

# Change working directory to src
os.chdir("C:/Users/sachi/SensoriumDecoding")

class Args:
    def __init__(self):
        self.output_dir = "runs/DNN"
args = Args()
load_args(args)


@torch.no_grad()
def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Metrics to compute as part of training and validation step"""
    # Pixel wise correlation
    # RMSE
    # SSIM

def train_step(
    mouse_id: str,
    batch: t.Dict[str, torch.Tensor],
    model: Model,
    optimizer: torch.optim,
    criterion: losses.Loss,
):
    # Define single train step with microbatching
    pass


def train_model(model, train_loaders, val_loaders, optimizer, criterion, device, num_epochs=10):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        model.train()
        train_loss = 0.0
        for mouse_id, train_loader in train_loaders.items():
            for batch_idx, data in enumerate(train_loader):
                print(f"Mouse {mouse_id}, Batch {batch_idx}")
                # Move data to device
                x, behavior, image = data['response'], data['behavior'], data['image']

                batch_size = x.size(0)
                image = image.view(batch_size, -1)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                output = model(x, behavior, mouse_id)
                loss = criterion(output, image)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

        # Compute average training loss
        train_loss /= len(train_loaders)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mouse_id, val_loader in val_loaders.items():
                for batch_idx, data in enumerate(val_loader):
                    print(f"Eval Mouse {mouse_id}, Batch {batch_idx}")
                    x, behavior, image = data['response'].to(device), data['behavior'].to(device), data['image'].to(device)
                    batch_size = x.size(0)
                    image = image.view(batch_size, -1)
                    output = model(x, behavior, mouse_id)
                    loss = criterion(output, image)
                    val_loss += loss.item()

        # Compute average validation loss
        val_loss /= len(val_loaders)
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses


def main():
    # Get dataloaders
    train_ds, val_ds, test_ds = get_training_ds(
        args,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        data_dir=args.dataset,
        device=args.device,
    )

    # Get model architecture
    model = get_model(args).to(args.device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()

    # Train model
    train_losses, val_losses = train_model(
        model,
        train_ds,
        val_ds,
        optimizer,
        criterion,
        args.device,
        num_epochs=args.num_epochs,
    )
    # Plot training and validation losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    

    

if __name__ == "__main__":
    main()