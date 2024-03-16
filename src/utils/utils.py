import torch
import numpy as np
import typing as t
import os
import sys
import torch.nn as nn
from collections import OrderedDict
from torch.optim import Optimizer



def update_dict(target: dict, source: dict, replace: bool = False):
    """Update target dictionary with values from source dictionary"""
    for k, v in source.items():
        if replace:
            target[k] = v
        else:
            if k not in target:
                target[k] = []
            target[k].append(v)


def log_metrics(
    results: t.Dict[str, t.Dict[str, t.Any]],
    epoch: int,
    mode: int = 0,
):
    """Compute the mean of the metrics in results and log to Summary

    Args:
        results: t.Dict[str, t.Dict[str, t.List[float]]],
            a dictionary of tensors where keys are the name of the metrics
            that represent results from of a mouse.
        epoch: int, the current epoch number.
        mode: int, Summary logging mode.
    """
    mouse_ids = list(results.keys())
    metrics = list(results[mouse_ids[0]].keys())
    for mouse_id in mouse_ids:
        for metric in metrics:
            value = results[mouse_id][metric]
            if isinstance(value, list):
                if torch.is_tensor(value[0]):
                    results[mouse_id][metric] = torch.mean(torch.stack(value)).item()
                else:
                    results[mouse_id][metric] = np.mean(value)
    overall_result = {}
    for metric in metrics:
        value = np.mean([results[mouse_id][metric] for mouse_id in mouse_ids])
        overall_result[metric[metric.find("/") + 1 :]] = value
    return overall_result


class RedirectOutput:
    """Re-direct stdout or stderr to log file"""

    def __init__(self, file: t.TextIO, stream: t.Literal["stdout", "stderr"]):
        assert stream in ("stdout", "stderr")
        self.file = file
        self.console = sys.stdout if stream == "stdout" else sys.stderr
        self._tqdm_newline = False

    def write(self, message: str):
        self.console.write(message)
        self.console.flush()
        if message:
            if self._tqdm_newline and message == "\n":
                # skip the newline message by tqdm when a loop has ended
                self._tqdm_newline = False
            elif message.startswith("\r"):
                # skip carriage return from tqdm
                self._tqdm_newline = True
            else:
                self.file.write(message)
                self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()


class Logger:
    """Write both stdout and stderr to file"""

    def __init__(self, args):
        filename = os.path.join(args.output_dir, "output.log")
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.file = open(filename, mode="a")
        sys.stdout = self.stdout = RedirectOutput(self.file, stream="stdout")
        sys.stderr = self.stderr = RedirectOutput(self.file, stream="stderr")

    def close(self):
        self.file.close()

class Scheduler:
    def __init__(
        self,
        args,
        model: nn.Module,
        optimizer: Optimizer = None,
        scaler = None,
        mode: t.Literal["min", "max"] = "max",
        max_reduce: int = 2,
        lr_patience: int = 10,
        factor: float = 0.3,
        min_epochs: int = 0,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        module_names: t.List[str] = None,
    ):
        """
        Args:
            args: argparse parameters.
            model: Model, model.
            optimizer: (optional) torch.optim, optimizer.
            scaler: (optional), GradScaler.
            mode: 'min' or 'max', compare objective by minimum or maximum
            max_reduce: int, maximum number of learning rate reductions before
                terminating early stopping.
            lr_patience: int, the number of epochs to wait before reducing the
                learning rate.
            factor: float, learning rate reduction factor.
                i.e. new_lr = factor * old_lr
            min_epochs: int, number of epochs to train the model before early
                stopping begins monitoring.
            save_optimizer: bool, save optimizer and scaler (if provided) state dict to checkpoint.
            save_scheduler: bool, save scheduler state dict to checkpoint.
            module_names: t.List[str], a list of module names in the model to
                save in the checkpoint, save all modules if None.
        """
        assert mode in ("min", "max"), f"mode must be either min or max but not {mode}."
        assert (
            not save_optimizer or optimizer is not None
        ), "Optimizer must be provided when save_optimizer=True"
        self.mode = mode
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.module_names = module_names
        self.max_reduce = max_reduce
        self.num_reduce = 0
        self.lr_patience = lr_patience
        self.lr_wait = 0
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor
        self.min_epochs = min_epochs
        self.best_value = torch.inf if mode == "min" else -torch.inf
        self.checkpoint_dir = os.path.join(args.output_dir, "ckpt")
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.device = args.device
        self.verbose = args.verbose

    def _parameters2save(self):
        state_dict = self.model.state_dict()
        parameters = OrderedDict()
        if self.module_names is None:
            parameters = state_dict
        else:
            for parameter in state_dict.keys():
                if parameter.split(".")[0] in self.module_names:
                    parameters[parameter] = state_dict[parameter]
        return parameters

    def save_checkpoint(
        self, value: t.Union[float, np.ndarray, torch.Tensor], epoch: int
    ):
        """Save current model as best_model.pt"""
        filename = os.path.join(self.checkpoint_dir, "model_state.pt")
        ckpt = {
            "epoch": epoch,
            "value": float(value),
            "model": self._parameters2save(),
        }
        if self.save_optimizer:
            ckpt["optimizer"] = self.optimizer.state_dict()
            if self.scaler is not None:
                ckpt["scaler"] = self.scaler.state_dict()
        if self.save_scheduler:
            ckpt["scheduler"] = self.state_dict()
        torch.save(ckpt, f=filename)
        if self.verbose:
            print(f"\nCheckpoint saved to {filename}.")

    def restore(
        self,
        force: bool = False,
        load_optimizer: bool = False,
        load_scheduler: bool = False,
    ) -> int:
        """
        Load the best model in self.checkpoint_dir and return the epoch
        Args:
            force: bool, raise an error if checkpoint is not found.
            load_optimizer: bool, load optimizer and scaler (if exists) from checkpoint.
            load_scheduler: bool, load scheduler from checkpoint.
        Return:
            epoch: int, the number of epoch the model has been trained for,
                return 0 if no checkpoint was found.
        """
        epoch = 0
        filename = os.path.join(self.checkpoint_dir, "model_state.pt")
        if os.path.exists(filename):
            ckpt = torch.load(filename, map_location=self.device)
            epoch = ckpt["epoch"]
            # it is possible that the checkpoint only contains part of a model
            # hence we update the current state_dict of the model instead of
            # directly calling model.load_state_dict(ckpt['model'])
            state_dict = self.model.state_dict()
            state_dict.update(ckpt["model"])
            self.model.load_state_dict(state_dict)
            if load_optimizer and "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                if self.scaler is not None and "scaler" in ckpt:
                    self.scaler.load_state_dict(ckpt["scaler"])
            if load_scheduler and "scheduler" in ckpt:
                self.load_state_dict(ckpt["scheduler"])
            if self.verbose:
                print(
                    f"\nLoaded checkpoint from epoch {epoch} "
                    f"(correlation: {ckpt['value']:.04f}).\n"
                )
        elif force:
            raise FileNotFoundError(f"Cannot find checkpoint in {self.checkpoint_dir}.")
        return epoch

    def state_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "model")
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def is_better(self, value: t.Union[float, torch.Tensor]):
        if self.mode == "min":
            return value < self.best_value
        else:
            return value > self.best_value

    def reduce_lr(self):
        """Reduce the learning rates for each param_group by the defined factor"""
        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = self.factor * float(param_group["lr"])
            param_group["lr"] = new_lr
            if self.verbose:
                print(
                    f"Reduce learning rate of {param_group['name']} to "
                    f"{new_lr:.04e} (num. reduce: {self.num_reduce})."
                )

    def step(self, value: t.Union[float, np.ndarray, torch.Tensor], epoch: int):
        terminate = False
        if self.is_better(value):
            self.best_value = value
            self.best_epoch = epoch
            self.lr_wait = 0
            self.num_reduce = 0
            self.save_checkpoint(value=value, epoch=epoch)
        elif epoch > self.min_epochs:
            if self.lr_wait >= self.lr_patience:
                if self.num_reduce >= self.max_reduce:
                    terminate = True
                    if self.verbose:
                        print(
                            f"\nModel has not improved after {self.num_reduce} "
                            f"LR reductions."
                        )
                else:
                    self.num_reduce += 1
                    self.restore()
                    self.reduce_lr()
                    self.lr_wait = 0
            else:
                self.lr_wait += 1
        return terminate