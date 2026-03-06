
import numpy as np
import torch
from timm.scheduler import CosineLRScheduler

_optimizers_factory = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    'RMSprop': torch.optim.RMSprop
}

def make_optimizer(cfg, args, model):
    """
    Initialize an optimizer based on the configs of cfg and args

    Parameters:
    -----------
    cfg, args: config and argument parser from the command line
    model: torch.nn

    Returns:
    ---------
    optimizer: torch.optim.SGD

    """
    optimizer_type = cfg['SOLVER']['optimizer']
    if optimizer_type == 'SGD':
        optimizer = _optimizers_factory[optimizer_type](
            model.parameters(),
            lr=cfg['SOLVER']['lr'],
            weight_decay=cfg['SOLVER']['weight_decay'],
            )
    elif optimizer_type == 'RMSprop':
        optimizer = _optimizers_factory[optimizer_type](
            model.parameters(),
            lr=cfg['SOLVER']['lr'],
            weight_decay=cfg['SOLVER']['weight_decay'],
            alpha=cfg['SOLVER']['alpha'],  # RMSprop-specific parameter
            momentum=cfg['SOLVER']['momentum']  # RMSprop-specific parameter
        )
    else:
        optimizer = _optimizers_factory[optimizer_type](
            model.parameters(),
            lr=cfg['SOLVER']['lr'],
            weight_decay=cfg['SOLVER']['weight_decay'],
            betas=(cfg['SOLVER']['beta1'], cfg['SOLVER']['beta2'])
            )
        
        print("weight decay: ", cfg['SOLVER']['weight_decay'])
    return optimizer

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return 
