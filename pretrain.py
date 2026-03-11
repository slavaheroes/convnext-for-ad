# -*- coding: utf-8 -*-
"""
Created on Wed Mar 7 2026

@author: qasymjomart and slavaheroes
"""

import numpy as np

import os
import time
import datetime
import argparse
import random
from omegaconf import OmegaConf
from loguru import logger as logger
import wandb
import monai

import torch

from dataloaders.make_dataloaders import prepare_pt_data
from models.make_models import make_pt_model
from utils.optimizers import cosine_scheduler
from utils.clip_grad import clip_gradients
from utils.logger import MetricLogger

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['WANDB_API_KEY'] = ""

# Set the seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    monai.utils.set_determinism(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print('Seed is set.')

def save_output_dir(*args, **kwargs):
    torch.save(*args, **kwargs)
        
def dir_exists(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('./monai_cache', exist_ok=True)

def get_args():
    # Parse some variable configs
    parser = argparse.ArgumentParser(description='Pre-train the models (ConvNeXT-based MAE, ViT-based MAE)')
    parser.add_argument('--config_file', type=str, default='./configs/pretrain_convnext_sparse.yaml', help='Name of the config file')
    parser.add_argument('--savename', type=str, help='Experiment name (used for saving files)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/', help='Directory to save pre-training outputs')
    parser.add_argument('--datasets', nargs='+', type=str, default=['IXI'], help='Datasets to use for pre-training MAE')
    parser.add_argument('--seed', type=int, default=7885, help='Experiment seed (for reproducible results)')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Mask ratio used for MAE')
    parser.add_argument('--devices', type=str, help='GPU devices to use')
    parser.add_argument('--size', default='base', type=str, help='Model size (small, base, large)')
    parser.add_argument('--use_aug', action='store_true', help='Augmentations')
    parser.add_argument('--kernel_size', type=int, default=7, help='kernel_size = 7, 3, 5')
    parser.add_argument('--downsampling', type=str, default='avgpool3d', help='conv, avgpool3d, maxpool3d')
    parser.add_argument('--decoder_dim', type=int, default=256, help='Decoder embedding dimension')
    args = parser.parse_args()
    return args

def pretrain_model():
    
    args = get_args()
    cfg = OmegaConf.load(args.config_file)
    set_seed(args.seed)
    dir_exists(args)

    print('-----------------------------')
    print('Selected devices: %s'%(args.devices))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    print('Process number: %d'%(os.getpid()))
    print('-----------------------------')

    aug_prefix = 'aug' if args.use_aug else 'noaug'
    FILENAME = f"MAE_pt_{int(args.mask_ratio*100)}_{args.savename}_{aug_prefix}_d_{'_'.join(args.datasets)}_seed_{args.seed}"

    os.environ["WANDB_API_KEY"] = OmegaConf.load("keys.yaml")["WANDB_API_KEY"]
    run = wandb.init(project="AD-NEXT", name=FILENAME, config=OmegaConf.to_container(cfg), dir=os.path.join(args.output_dir, FILENAME),
                tags=['PT'], group=FILENAME)
    run.log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"))
    
    # Prepare the model and data
    model = make_pt_model(cfg, args)
    model = model.cuda()
    data_loader, dataset = prepare_pt_data(cfg, args)
    logger.success(f"Data loaded: there are {len(dataset)} images.")
    
    # ============ init schedulers ... ============
    lr = cfg.optimizer.lr
    min_lr = cfg.optimizer.min_lr
    batch_size = cfg.training.batch_size
    warmup_epochs = cfg.training.warmup_epochs
    weight_decay = cfg.optimizer.weight_decay
    epochs = cfg.training.epochs
    
    coeff_lr_div = 256.0
    lr_schedule = cosine_scheduler(
        lr * batch_size / coeff_lr_div,  # linear scaling rule
        min_lr,
        epochs, len(data_loader),
        warmup_epochs=warmup_epochs,
    )    
    if cfg.optimizer.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
                model.parameters(),
                weight_decay=weight_decay,
                betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
            )
    assert optimizer is not None, "No optimizer is created."
    
    fp16_scaler = None
    if cfg.training.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    logger.success("Loss, optimizer and schedulers ready.")
        
    ### LOG CHORES ###
    cfg.transforms.cache_dir_train = f'./monai_cache/pretrain_{FILENAME}'
    os.makedirs(os.path.join(args.output_dir, FILENAME), exist_ok=True)
    logger.add(os.path.join(args.output_dir, FILENAME, 'log.txt'))
    logger.info(cfg)
    logger.info(args)

    os.environ["WANDB_API_KEY"] = OmegaConf.load("keys.yaml")["WANDB_API_KEY"]
    run = wandb.init(project="AD-NEXT", name=FILENAME, config=OmegaConf.to_container(cfg), dir=os.path.join(args.output_dir, FILENAME),
                tags=['PT'], group=FILENAME)
    run.log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"))
    
    # Prepare the model and data
    model = make_pt_model(cfg, args)
    model = model.cuda()
    data_loader, dataset = prepare_pt_data(cfg, args)
    logger.success(f"Data loaded: there are {len(dataset)} images.")
    
    # ============ init schedulers ... ============
    lr = cfg.optimizer.lr
    min_lr = cfg.optimizer.min_lr
    batch_size = cfg.training.batch_size
    warmup_epochs = cfg.training.warmup_epochs
    weight_decay = cfg.optimizer.weight_decay
    epochs = cfg.training.epochs
    
    coeff_lr_div = 256.0
    lr_schedule = cosine_scheduler(
        lr * batch_size / coeff_lr_div,  # linear scaling rule
        min_lr,
        epochs, len(data_loader),
        warmup_epochs=warmup_epochs,
    )    
    if cfg.optimizer.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
                model.parameters(),
                weight_decay=weight_decay,
                betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
            )
    assert optimizer is not None, "No optimizer is created."
    
    fp16_scaler = None
    if cfg.training.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    logger.success("Loss, optimizer and schedulers ready.")
    
    start_epoch = 0
    steps = 0
    start_time = time.time()
    logger.warning("Starting pre-training!")
    
    for epoch in range(start_epoch, epochs):

        # ============ training one epoch ... ============
        train_stats, steps = train_one_epoch(model, data_loader, optimizer,
            epoch, run, lr_schedule, fp16_scaler, steps, cfg, args)
        
        # ============ save checkpoint ============
        save_dict = {
            'net': model.state_dict(),
            'args': args,
            'cfg': cfg,
        }
        
        save_output_dir(save_dict, os.path.join(args.output_dir, FILENAME, 'checkpoint.pth'))
        if cfg.training.saveckp_freq and epoch % cfg.training.saveckp_freq == 0:
            save_output_dir(save_dict, os.path.join(args.output_dir, FILENAME, f'checkpoint{epoch:04}.pth'))
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    'epoch': epoch}
        
        run.log(log_stats)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str}')
    save_output_dir(save_dict, os.path.join(args.output_dir, FILENAME, 'checkpoint_last.pth'))
    
    if run is not None:
        run.finish()
    logger.success('Training is finished.')

def train_one_epoch(model, data_loader, optimizer, epoch, run, lr_schedule, fp_scaler, steps, cfg, args):
    model.train()
    # steps = epoch * len(data_loader)
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, cfg.training.epochs)
    for it, images in enumerate(metric_logger.log_every(data_loader, 10, header)):
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
        
        optimizer.zero_grad()
        images = images['image'].cuda()
        
        if fp_scaler is not None:
            with torch.cuda.amp.autocast(enabled=False):
                loss, _, _ = model(images, mask_ratio=args.mask_ratio)
            fp_scaler.scale(loss).backward()
            # fp_scaler.unscale_(optimizer)
            if cfg.optimizer.clip_grad:
                fp_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = clip_gradients(model, cfg.optimizer.clip_grad)
            fp_scaler.step(optimizer)
            fp_scaler.update()
            
        else:
            loss, _, _ = model(images, mask_ratio=args.mask_ratio)
            loss.backward()
            if cfg.optimizer.clip_grad:
                param_norms = clip_gradients(model, cfg.optimizer.clip_grad)
            optimizer.step()
        
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        
        steps += 1
        if run is not None:
            run.log({"iter_loss": loss.item(),
                     "iter_lr": optimizer.param_groups[0]["lr"]
                     }, step=it)
            
    torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats: " + str(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, steps

if __name__ == '__main__':
    # Process number
    logger.info(f"Process number: {os.getpid()}")
    pretrain_model()