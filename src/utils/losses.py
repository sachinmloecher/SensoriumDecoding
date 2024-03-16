import torch
import typing as t
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss

_REDUCATION = t.Literal["sum", "mean"]
EPS = torch.finfo(torch.float32).eps


class MSELoss(_Loss):
    """Basic MSE Criterion class"""

    def __init__(
        self,
        args,
        ds: t.Dict[str, DataLoader],
        size_average: bool = None,
        reduce: bool = None,
        reduction = "sum",
    ):
        super(MSELoss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction
        )
        self.ds_scale = args.ds_scale
        self.ds_sizes = {
            mouse_id: torch.tensor(len(mouse_ds.dataset), dtype=torch.float32)
            for mouse_id, mouse_ds in ds.items()
        }

    def scale_ds(self, loss: torch.Tensor, mouse_id: str, batch_size: int):
        """Scale loss based on the size of the dataset"""
        if self.ds_scale:
            scale = torch.sqrt(self.ds_sizes[mouse_id] / batch_size)
            loss = scale * loss
        return loss

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mouse_id: str,
        reduction = "sum",
        batch_size: int = None,
    ):
        if batch_size is None:
            batch_size = y_true.size(0)
        loss = torch.mean(torch.square(y_true - y_pred))
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=batch_size)
        return loss


def get_criterion(args, ds: t.Dict[str, DataLoader]):
    criterion = MSELoss(args, ds=ds)
    criterion.to(args.device)
    return criterion