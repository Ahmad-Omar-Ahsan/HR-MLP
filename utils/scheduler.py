from torch import optim
from torch.optim import lr_scheduler


class WarmUpLR(lr_scheduler._LRScheduler):
    """WarmUp learning rate scheduler.
    Args:
        optimizer (optim.Optimizer): Optimizer instance
        total_iters (int): steps_per_epoch * n_warmup_epochs
        last_epoch (int): Final epoch. Defaults to -1.
    """

    def __init__(
        self, optimizer: optim.Optimizer, total_iters: int, last_epoch: int = -1
    ):
        """Initializer for WarmUpLR"""

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Learning rate will be set to base_lr * last_epoch / total_iters."""

        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]


def get_scheduler(
    optimizer: optim.Optimizer, scheduler_type: str, T_max: int, max_lr: float = 1.0
) -> lr_scheduler._LRScheduler:
    """Gets scheduler.
    Args:
        optimizer (optim.Optimizer): Optimizer instance.
        scheduler_type (str): Specified scheduler.
        T_max (int):  Maximum number of iterations.
        max_lr(float) : Max learning rate. Optional

    Raises:
        ValueError: Unsupported scheduler type.
    Returns:
        lr_scheduler._LRScheduler: Scheduler instance.
    """

    if scheduler_type == "cosine_annealing":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=1e-8)
    elif scheduler_type == "one_cycle_lr":
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps=T_max)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return scheduler
