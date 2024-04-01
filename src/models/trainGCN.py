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
from torch_geometric.data import Data


from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

sys.path.append("C:/Users/sachi/SensoriumDecoding/src/utils")

import data
from data import load_args, get_training_ds, DataLoader
from losses import get_criterion
import utils
from utils import Logger, Scheduler
from model import get_model, FullModel

# Change working directory to src
os.chdir("C:/Users/sachi/SensoriumDecoding")

class Args:
    def __init__(self):
        self.output_dir = "runs\DNN\mouseGraphConv1_128_nn30_tanh_coreMLP"
args = Args()
load_args(args)

def save_checkpoint(model, optimizer, scheduler, args, epoch, history):
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history
    }
    
    # Save parameters of each mouse-specific MLP separately
    for mouse_id, mouse_mlp in model.mouse_mlp_dict.items():
        checkpoint[f'mouse_mlp_{mouse_id}_state_dict'] = mouse_mlp.state_dict()

    torch.save(checkpoint, checkpoint_path)

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

def gather(result: t.Dict[str, t.List[torch.Tensor]]):
    return {k: torch.sum(torch.stack(v)).cpu() for k, v in result.items()}

def vstack(tensors: t.List[torch.Tensor]):
    return torch.vstack(tensors).cpu()

@torch.no_grad()
def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Metrics to compute as part of training and validation step"""
    batch_size, h, w = y_true.size()
    # Reshape tensors to match the required format for calculations
    y_true = y_true.view(batch_size, -1)
    y_pred = y_pred.view(batch_size, -1)
    # Pixel wise correlation
    correlations = []
    for i in range(y_true.shape[1]):
        correlation, _ = pearsonr(y_true[:, i].cpu(), y_pred[:, i].cpu())
        correlations.append(correlation)
    correlation = torch.tensor(correlations).mean().item()
    # RMSE
    rmse = torch.sqrt(mse_loss(y_true, y_pred))
    # SSIM
    ssim_score = torch.tensor([ssim(y_true[i].cpu().numpy(), y_pred[i].cpu().numpy(), data_range=4) for i in range(batch_size)]).mean()
    return {
        "correlation": correlation,
        "rmse": rmse.item(),
        "ssim": ssim_score.item(),
    }


def train_step(
    mouse_id: str,
    batch: Data,
    model: FullModel,
    optimizer: torch.optim,
    criterion: torch.nn.Module,
    update: bool,
    micro_batch_size: int,
    device: torch.device = "cpu"
):
    model.to(device)
    batch_size = len(batch.image)
    result = {"loss/loss": []}

    y_true = torch.stack([torch.from_numpy(img) for img in batch.image]).to(device)
    y_pred = model(
        x=batch.x.unsqueeze(1).to(device),
        edge_index=batch.edge_index.to(device),
        batch=batch.batch.to(device),
        mouse_id=mouse_id,
        behaviours=torch.stack([torch.from_numpy(behavior) for behavior in batch.behavior]).to(device),
        pupil_centers=torch.stack([torch.from_numpy(pupil_center) for pupil_center in batch.pupil_center]).to(device)
    )
    y_pred = y_pred.view(y_true.size(0), 36, 64)
    loss = criterion(
        y_true=y_true,
        y_pred=y_pred,
        mouse_id=mouse_id,
        batch_size=batch_size,
        reduction="sum"
    )
    loss.backward()
    result["loss/loss"].append(loss.detach())

    if update:
        optimizer.step()
        optimizer.zero_grad()
    return gather(result)

def train(
    args,
    ds: t.Dict[str, DataLoader],
    model: FullModel,
    optimizer: torch.optim,
    criterion: torch.nn.Module,
    epoch: int
):
    mouse_ids = list(ds.keys())
    results = {mouse_id: {} for mouse_id in mouse_ids}
    ds = data.CycleDataloaders(ds)
    update_frequency = len(mouse_ids)
    model.train(True)
    optimizer.zero_grad()
    for i, (mouse_id, mouse_batch) in tqdm(
        enumerate(ds), desc="Train", total=len(ds), disable=args.verbose < 2
    ):
        result = train_step(
            mouse_id=mouse_id,
            batch=mouse_batch,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            update=(i + 1) % update_frequency == 0,
            micro_batch_size=args.micro_batch_size,
            device=args.device
        )
        utils.update_dict(results[mouse_id], result)
    return utils.log_metrics(results, epoch=epoch, mode=0)

@torch.no_grad()
def validation_step(
    mouse_id: str,
    batch: t.Dict[str, torch.Tensor],
    model: FullModel,
    criterion: torch.nn.Module,
    micro_batch_size: int,
    device: torch.device = "cpu"
):
    model.to(device)
    batch_size = len(batch.image)
    result = {"loss/loss": []}
    targets, predictions = [], []
    y_true = torch.stack([torch.from_numpy(img) for img in batch.image]).to(device)
    y_pred = model(
        x=batch.x.unsqueeze(1).to(device),
        edge_index=batch.edge_index.to(device),
        batch=batch.batch.to(device),
        mouse_id=mouse_id,
        behaviours=torch.stack([torch.from_numpy(behavior) for behavior in batch.behavior]).to(device),
        pupil_centers=torch.stack([torch.from_numpy(pupil_center) for pupil_center in batch.pupil_center]).to(device)
    )
    y_pred = y_pred.view(y_true.size(0), 36, 64)
    loss = criterion(
        y_true=y_true,
        y_pred=y_pred,
        mouse_id=mouse_id,
        batch_size=batch_size,
        reduction="sum"
    )
    # loss /= micro_batch_size
    result["loss/loss"].append(loss)
    targets.append(y_true)
    predictions.append(y_pred)
    return gather(result), vstack(targets), vstack(predictions)


def validate(
    args,
    ds: t.Dict[str, DataLoader],
    model: FullModel,
    criterion: torch.nn.Module,
    epoch: int
):
    model.train(False)
    results = {}
    with tqdm(desc="Val", total=len(ds), disable=args.verbose < 2) as pbar:
        for mouse_id, mouse_ds in ds.items():
            mouse_result, y_true, y_pred = {}, [], []
            for batch in mouse_ds:
                result, targets, predictions = validation_step(
                    mouse_id=mouse_id,
                    batch=batch,
                    model=model,
                    criterion=criterion,
                    micro_batch_size=args.micro_batch_size,
                    device=args.device
                )
                utils.update_dict(mouse_result, result)
                y_true.append(targets)
                y_pred.append(predictions)
                pbar.update(1)
            y_true, y_pred = vstack(y_true), vstack(y_pred)
            mouse_result.update(compute_metrics(y_true=y_true, y_pred=y_pred))
            results[mouse_id] = mouse_result
            del y_true, y_pred
    return utils.log_metrics(results, epoch=epoch, mode=1)

def main(args):
    Logger(args)

    train_ds, val_ds, test_ds = data.get_training_ds(
        args,
        data_dir=args.dataset,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )

    model = get_model(args).to(args.device)  # Define the model architecture
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # Resume training
    if args.resume:
        model, optimizer, scheduler, start_epoch, history = load_checkpoint(args, model, optimizer, scheduler)
        start_epoch += 1
    else:
        start_epoch = 1
        history = {"train_loss": [], "val_loss": [], "val_correlation": []}

    criterion = get_criterion(args, ds=train_ds)
    
 
    for epoch in range(start_epoch, args.epochs + 1):
        if args.verbose:
            print(f"\nEpoch {epoch:03d}/{args.epochs:03d}")

        start = time()
        train_result = train(
            args,
            ds=train_ds,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
        )

        val_result = validate(
            args,
            ds=val_ds,
            model=model,
            criterion=criterion,
            epoch=epoch
        )

        history["train_loss"].append(train_result["loss"])
        history["val_loss"].append(val_result["loss"])
        history["val_correlation"].append(val_result["correlation"])

        elapse = time() - start
        if args.verbose:
            print(
                f'Train\t\tloss: {train_result["loss"]:.04f}\n',
                f'Val\t\tloss: {val_result["loss"]:.04f}\n',
                f'Correlation: {val_result["correlation"]:.04f}\n',
            )

        optimizer.step()
        scheduler.step()
        save_checkpoint(model, optimizer, scheduler, args, epoch, history)

    return model

if __name__ == "__main__":
    model = main(args)