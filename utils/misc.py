"""Miscellaneous helper functions."""

import torch
from torch import nn, optim
import numpy as np
import random
import os
import wandb
from models.resnet import ResNet, ResNet_small
from models.Simple_CNN import SimpleConvNet
from models.MLP_mixer import MLPMixer
from models.gMLP import gMLPVision
from models.vgg import VGG_net
from models.inception_net import GoogLeNet
from models.lorentz_iso_vig import Isotropic_VIG_lorentz, Isotropic_VIG_lorentz_complete
from models.iso_vig import Isotropic_VIG, poincare_iso_VIG, Isotropic_VIG_Lorentz_head
from models.pyramid_vig import IVGN
from models.Lorentz_resmlp import Lorentz_resmlp
from models.Lorentz_ViT import Lorentz_ViT
from models.Resmlp import ResMLP, Poincare_ResMLP, Lor_ResMLP_ablation
from models.ViT import ViT


def seed_everything(seed: str) -> None:
    """Set manual seed.
    Args:
        seed (int): Supplied seed.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Set seed {seed}")


def count_params(model: nn.Module) -> int:
    """Counts number of parameters in a model.
    Args:
        model (torch.nn.Module): Model instance for which number of params is to be counted.
    Returns:
        int: Parameter count.
    """

    return sum(map(lambda p: p.data.numel(), model.parameters()))


def calc_step(epoch: int, n_batches: int, batch_index: int) -> int:
    """Calculates current step.
    Args:
        epoch (int): Current epoch.
        n_batches (int): Number of batches in dataloader.
        batch_index (int): Current batch index.
    Returns:
        int: Current step.
    """
    return (epoch - 1) * n_batches + (1 + batch_index)


def log(log_dict: dict, step: int, config: dict) -> None:
    """Handles logging for metric tracking server, local disk and stdout.
    Args:
        log_dict (dict): Log metric dict.
        step (int): Current step.
        config (dict): Config dict.
    """

    # send logs to wandb tracking server
    if config["exp"]["wandb"]:
        wandb.log(log_dict, step=step)

    log_message = f"Step: {step} | " + " | ".join(
        [f"{k}: {v}" for k, v in log_dict.items()]
    )

    # write logs to disk
    if config["exp"]["log_to_file"]:
        log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")

        with open(log_file, "a+") as f:
            f.write(log_message + "\n")

    # show logs in stdout
    if config["exp"]["log_to_stdout"]:
        print(log_message)


def get_model(model_config: dict) -> nn.Module:
    """Creates model from config dict.
    Args:
        model_config (dict): Dict containing model config params. If the "name" key is not None, other params are ignored.
    Returns:
        nn.Module: Model instance.
    """

    if model_config["type"] == "ResNet_small":
        return ResNet_small(**model_config["ResNet_small"])
    elif model_config['type'] == "Lor_ResMLP_ablation":
        return Lor_ResMLP_ablation(**model_config["Lor_ResMLP_ablation"])
    elif model_config["type"] == "ViT":
        return ViT(**model_config["ViT"])
    elif model_config["type"] == "simple_cnn":
        return SimpleConvNet(**model_config["simple_cnn"])
    elif model_config["type"] == "gMLP":
        return gMLPVision(**model_config["gMLP"])
    elif model_config["type"] == "MLP_mixer":
        return MLPMixer(**model_config["MLP_mixer"])
    elif model_config["type"] == "VGG":
        return VGG_net(**model_config["VGG"])
    elif model_config["type"] == "Inception":
        return GoogLeNet(**model_config["Inception"])
    elif model_config["type"] == "lorentz":
        return Isotropic_VIG_lorentz(**model_config["lorentz"])
    elif model_config["type"] == "lorentz_complete":
        return Isotropic_VIG_lorentz_complete(**model_config["lorentz_complete"])
    elif model_config["type"] == "iso_vig":
        return Isotropic_VIG(**model_config["iso_vig"])
    elif model_config["type"] == "ivgn":
        return IVGN(**model_config["ivgn"])
    elif model_config["type"] == "poincare_iso_vig":
        return poincare_iso_VIG(**model_config["poincare_iso_vig"])
    elif model_config["type"] == "iso_vig_l_head":
        return Isotropic_VIG_Lorentz_head(**model_config["iso_vig_l_head"])
    elif model_config["type"] == "lorentz_resmlp":
        return Lorentz_resmlp(**model_config["lorentz_resmlp"])
    elif model_config["type"] == "lorentz_vit":
        return Lorentz_ViT(**model_config["lorentz_vit"])
    elif model_config["type"] == "resmlp":
        return ResMLP(**model_config["resmlp"])
    elif model_config["type"] == "poincare_resmlp":
        return Poincare_ResMLP(**model_config["poincare_resmlp"])
    # elif model_config['type'] == 'lorentz_poincare_resmlp':
    #     return Lorentz_poincare_resmlp(**model_config['lorentz_poincare_resmlp'])
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")


def save_model(
    epoch: int,
    val_acc: float,
    save_path: str,
    net: nn.Module,
    optimizer: optim.Optimizer = None,
    scheduler: optim.lr_scheduler._LRScheduler = None,
    log_file: str = None,
) -> None:
    """Saves checkpoint.
    Args:
        epoch (int): Current epoch.
        val_acc (float): Validation accuracy.
        save_path (str): Checkpoint path.
        net (nn.Module): Model instance.
        optimizer (optim.Optimizer, optional): Optimizer. Defaults to None.
        scheduler (optim.lr_scheduler._LRScheduler): Scheduler. Defaults to None.
        log_file (str, optional): Log file. Defaults to None.
    """

    ckpt_dict = {
        "epoch": epoch,
        "val_acc": val_acc,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
        if optimizer is not None
        else optimizer,
        "scheduler_state_dict": scheduler.state_dict()
        if scheduler is not None
        else scheduler,
    }

    torch.save(ckpt_dict, save_path)

    log_message = f"Saved {save_path} with accuracy {val_acc}."
    print(log_message)

    if log_file is not None:
        with open(log_file, "a+") as f:
            f.write(log_message + "\n")
