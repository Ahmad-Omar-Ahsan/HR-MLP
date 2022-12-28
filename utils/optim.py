from torch import nn, optim
from optim_h.radam import RiemannianAdam
from optim_h.rsgd import RiemannianSGD

def get_optimizer(net: nn.Module, opt_config: dict) -> optim.Optimizer:
    """Creates optimizer based on config.
    Args:
        net (nn.Module): Model instance.
        opt_config (dict): Dict containing optimizer settings.
    Raises:
        ValueError: Unsupported optimizer type.
    Returns:
        optim.Optimizer: Optimizer instance.
    """

    if opt_config["opt_type"] == "adamw":
        optimizer = optim.AdamW(net.parameters(), **opt_config["opt_kwargs"])
    elif opt_config["opt_type"] == 'radam':
        optimizer = RiemannianAdam(net.parameters(), **opt_config["opt_kwargs"])
    elif opt_config["opt_type"] == 'rsgd':
        optimizer = RiemannianSGD(net.parameters(), **opt_config["opt_kwargs"])
    else:
        raise ValueError(f'Unsupported optimizer {opt_config["opt_type"]}')

    return optimizer

def get_optimizer_classifier(net: nn.Module, opt_config: dict) -> optim.Optimizer:
    """Creates optimizer based on config to train classifier.
    Args:
        net (nn.Module): Model instance.
        opt_config (dict): Dict containing optimizer settings.
    Raises:
        ValueError: Unsupported optimizer type.
    Returns:
        optim.Optimizer: Optimizer instance.
    """

    if opt_config["opt_type"] == "adamw":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad==True, net.parameters()), **opt_config["opt_kwargs"])
    elif opt_config["opt_type"] == 'radam':
        optimizer = RiemannianAdam(filter(lambda p: p.requires_grad==True, net.parameters()), **opt_config["opt_kwargs"])
    elif opt_config["opt_type"] == 'rsgd':
        optimizer = RiemannianSGD(filter(lambda p: p.requires_grad==True, net.parameters()), **opt_config["opt_kwargs"])
    else:
        raise ValueError(f'Unsupported optimizer {opt_config["opt_type"]}')

    return optimizer