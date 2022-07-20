from argparse import ArgumentParser
from config import get_config

from utils.loss import LabelSmoothingLoss
from utils.optim import get_optimizer
from utils.scheduler import WarmUpLR, get_scheduler
from utils.train import train, evaluate
from utils.dataset import get_train_valid_loader_CIFAR100, get_test_loader
from utils.misc import seed_everything, count_params, get_model, calc_step, log


import torch
from torch import nn
import numpy as np
import wandb

import os
import yaml
import random
import time


def training_pipeline(config):
    """Initiates and executes all the steps involved with model training.
    Args:
        config (dict) - Dict containing various settings for the training run.
    """

    config["exp"]["save_dir"] = os.path.join(
        config["exp"]["exp_dir"], config["exp"]["exp_name"]
    )
    os.makedirs(config["exp"]["save_dir"], exist_ok=True)

    ######################################
    # save hyperparameters for current run
    ######################################

    config_str = yaml.dump(config)
    print("Using settings:\n", config_str)

    with open(os.path.join(config["exp"]["save_dir"], "settings.txt"), "w+") as f:
        f.write(config_str)

    #####################################
    # initialize training items
    #####################################

    # data

    trainloader, valloader = get_train_valid_loader_CIFAR100(
        data_dir=config["data_dir"],
        random_seed=config['hparams']['seed'],
        batch_size=config["hparams"]["batch_size"],
        augment=config["hparams"]["augment"],
        num_workers=config["exp"]["n_workers"],
        pin_memory=config["exp"]["pin_memory"],
    )
    testloader = get_test_loader(
        data_dir=config["data_dir"],
        batch_size=config["hparams"]["batch_size"],
        num_workers=config["exp"]["n_workers"],
        pin_memory=config["exp"]["pin_memory"],
    )

    # model
    model = get_model(config["hparams"]["model"])
    model = model.to(config["hparams"]["device"])
    print(f"Created model with {count_params(model)} parameters.")

    # loss
    if config["hparams"]["l_smooth"]:
        criterion = LabelSmoothingLoss(
            num_classes=config["hparams"]["model"]["num_classes"],
            smoothing=config["hparams"]["l_smooth"],
        )
    else:
        criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = get_optimizer(model, config["hparams"]["optimizer"])

    # scheduler
    schedulers = {"warmup": None, "scheduler": None}

    if config["hparams"]["scheduler"]["n_warmup"]:
        schedulers["warmup"] = WarmUpLR(
            optimizer,
            total_iters=len(trainloader) * config["hparams"]["scheduler"]["n_warmup"],
        )

    if config["hparams"]["scheduler"]["scheduler_type"] is not None:
        total_iters = len(trainloader) * max(
            1,
            (
                config["hparams"]["scheduler"]["max_epochs"]
                - config["hparams"]["scheduler"]["n_warmup"]
            ),
        )
        schedulers["scheduler"] = get_scheduler(
            optimizer, config["hparams"]["scheduler"]["scheduler_type"], total_iters
        )

    #####################################
    # Resume run
    #####################################

    if config["hparams"]["restore_ckpt"]:
        ckpt = torch.load(config["hparams"]["restore_ckpt"])

        config["hparams"]["start_epoch"] = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if schedulers["scheduler"]:
            schedulers["scheduler"].load_state_dict(ckpt["scheduler_state_dict"])

        print(f'Restored state from {config["hparams"]["restore_ckpt"]} successfully.')

    #####################################
    # Training
    #####################################

    print("Initiating training.")
    train(model, optimizer, criterion, trainloader, valloader, schedulers, config)

    #####################################
    # Final Test
    #####################################

    final_step = calc_step(
        config["hparams"]["n_epochs"] + 1, len(trainloader), len(trainloader) - 1
    )

    # evaluating the final state (last.pth)
    test_acc, test_loss = evaluate(
        model, criterion, testloader, config["hparams"]["device"]
    )
    log_dict = {"test_loss_last": test_loss, "test_acc_last": test_acc}
    log(log_dict, final_step, config)

    # evaluating the best validation state (best.pth)
    ckpt = torch.load(os.path.join(config["exp"]["save_dir"], "best.pth"))
    model.load_state_dict(ckpt["model_state_dict"])
    print("Best ckpt loaded.")

    test_acc, test_loss = evaluate(
        model, criterion, testloader, config["hparams"]["device"]
    )
    log_dict = {"test_loss_best": test_loss, "test_acc_best": test_acc}
    log(log_dict, final_step, config)


def main(args):
    config = get_config(args.conf)
    seed_everything(config["hparams"]["seed"])

    if config["exp"]["wandb"]:
        if config["exp"]["wandb_api_key"] is not None:
            with open(config["exp"]["wandb_api_key"], "r") as f:
                os.environ["WANDB_API_KEY"] = f.read()
                print(os.environ["WANDB_API_KEY"] )

        elif os.environ.get("WANDB_API_KEY", False):
            print(f"Found API key from env variable.")

        else:
            wandb.login()

        with wandb.init(
            project=config["exp"]["proj_name"],
            name=config["exp"]["exp_name"],
            config=config["hparams"],
        ):
            training_pipeline(config)

    else:
        training_pipeline(config)


if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument(
        "--conf", type=str, required=True, help="Path to config.yaml file."
    )
    args = parser.parse_args()

    main(args)