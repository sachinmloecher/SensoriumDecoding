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
from utils import Logger, Scheduler
from model import get_model, FullModel

# Change working directory to src
os.chdir("C:/Users/sachi/SensoriumDecoding")

class Args:
    def __init__(self):
        self.output_dir = "runs/DNN/mouseMLP_4500_3000_tanh_02_Decoder_512_tanh_Behaviour"
args = Args()
load_args(args)

def vstack(tensors: t.List[torch.Tensor]):
    return torch.vstack(tensors).cpu()

def gather(result: t.Dict[str, t.List[torch.Tensor]]):
    return {k: torch.sum(torch.stack(v)).cpu() for k, v in result.items()}

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

def load_checkpoint(args, checkpoint_file, model, optimizer, scheduler):
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    checkpoint_file = os.path.join(checkpoint_dir, checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    history = checkpoint['history']
    
    # Load parameters of each mouse-specific MLP separately
    for mouse_id, mouse_mlp in model.mouse_mlp_dict.items():
        mouse_mlp.load_state_dict(checkpoint[f'mouse_mlp_{mouse_id}_state_dict'])

    return model, optimizer, scheduler, epoch, history#


@torch.no_grad()
def test_step(
    mouse_id: str,
    batch: t.Dict[str, torch.Tensor],
    model: FullModel,
    criterion: torch.nn.Module,
    micro_batch_size: int,
    device: torch.device = "cpu"
):
    model.to(device)
    batch_size = batch["image"].size(0)
    result = {"loss/loss": []}
    targets, predictions = [], []
    for micro_batch in data.micro_batching(batch, micro_batch_size):
        y_true = micro_batch["image"].to(device)
        y_pred = model(
            x=micro_batch["response"].to(device),
            edge_index=None,
            batch=None,
            mouse_id=mouse_id,
            behaviours=micro_batch["behavior"].to(device),
            pupil_centers=micro_batch["pupil_center"].to(device),
        )
        y_pred = y_pred.view(y_true.size(0), 36, 64)
        loss = criterion(
            y_true=y_true,
            y_pred=y_pred,
            mouse_id=mouse_id,
            batch_size=batch_size,
            reduction="sum"
        )
        result["loss/loss"].append(loss)
        targets.append(y_true)
        predictions.append(y_pred)
    return gather(result), vstack(targets), vstack(predictions)

def test(
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
                result, targets, predictions = test_step(
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

    model = get_model(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    criterion = get_criterion(args, ds=test_ds)

    model, optimizer, scheduler, epoch, history = load_checkpoint(args, "checkpoint_epoch_54.pt", model, optimizer, scheduler)
    test_results = test(args, test_ds, model, criterion, epoch)

    print(
        f'Test\t\tloss: {test_results["loss"]:.04f}\n',
        f'Correlation: {test_results["correlation"]:.04f}\n',
    )

    # Plot history
    plt.plot(history["train_loss"])
    plt.plot(history["val_loss"])
    plt.title('Model loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    main(args)