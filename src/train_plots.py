import numpy as np
from sklearn.linear_model import Ridge
import argparse
import torch
import wandb
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.nn.functional import mse_loss
from skimage.metrics import structural_similarity as ssim
import typing as t
from tqdm import tqdm
from time import time


from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

sys.path.append("C:/Users/sachi/SensoriumDecoding/src/utils")

import data
from data import load_args, get_training_ds, DataLoader
from losses import get_criterion
import utils
from models.model import get_model, FullModel

# Change working directory to src
os.chdir("C:/Users/sachi/SensoriumDecoding")

class Args:
    def __init__(self):
        self.output_dir = "runs/DNN/mouseMLP_1000_relu_coreMLP_1000"
args = Args()
load_args(args)

def load_checkpoint(args, model, optimizer, scheduler):
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    
    # Get list of checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir)]
    
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found.")
    
    # Get the latest epoch folder
    latest_epoch_file = max(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    checkpoint_file = os.path.join(checkpoint_dir, latest_epoch_file)
    checkpoint = torch.load(checkpoint_file)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    history = checkpoint['history']
    
    # Load parameters of each mouse-specific MLP separately
    for mouse_id, mouse_mlp in model.mouse_mlp_dict.items():
        mouse_mlp.load_state_dict(checkpoint[f'mouse_mlp_{mouse_id}_state_dict'])

    return model, optimizer, scheduler, epoch, history

def main():
    # Load checkpoint
    model = get_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    model, optimizer, scheduler, epoch, history = load_checkpoint(args, model, optimizer, scheduler)

    # Plot history
    plt.plot(history['train_loss'], label='Train loss')
    plt.plot(history['val_loss'], label='Val loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "history.png"))


if __name__ == "__main__":
    main()