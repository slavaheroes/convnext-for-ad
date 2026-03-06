# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 2023

@author: qasymjomart and slavaheroes
"""

import numpy as np

import os
from datetime import datetime
import argparse
import random
from omegaconf import OmegaConf

import torch
from torch.cuda import amp

from dataloaders.make_dataloaders import prepare_pt_data
from models.make_models import make_pt_model
from do_pretrain import do_pretrain
from utils.optimizers import make_optimizer

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Set the seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print('Seed is set.')

def get_args():
    # Parse some variable configs
    parser = argparse.ArgumentParser(description='Pre-train the models (ConvNeXT-based MAE, ViT-based MAE)')
    parser.add_argument('--config_file', type=str, default='./configs/pretrain_convnext_sparse.yaml', help='Name of the config file')
    parser.add_argument('--savename', type=str, help='Experiment name (used for saving files)')
    parser.add_argument('--datasets_to_use', nargs='+', type=str, default=['IXI'], help='Datasets to use for pre-training MAE')
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

if __name__ == '__main__':
    
    args = get_args()
    cfg = OmegaConf.load(args.config_file)
    set_seed(args.seed)

    print('-----------------------------')
    print('Selected devices: %s'%(args.devices))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    print('Process number: %d'%(os.getpid()))
    print('-----------------------------')

    FILENAME_POSTFIX = args.savename + '_seed_' + str(args.seed)
    timestamp_current = datetime.now()
    timestamp_current = timestamp_current.strftime("%Y%m%d_%H%M")

    # Monai logs foldernames
    cfg.transforms.cache_dir_train = './monai_logs/train_' + FILENAME_POSTFIX
    
    cfg.model.kernel_size = args.kernel_size
    cfg.model.downsampling = args.downsampling
    cfg.model.decoder_embed_dim = args.decoder_dim
    
    model = make_pt_model(cfg, args)
    dataloader, dataset = prepare_pt_data(cfg, args)
      
    # Move model to GPU
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Num of parameters in the model: ', params)
    model.cuda()

    optimizer = make_optimizer(cfg, args, model)
    scaler = amp.GradScaler()
    
    # Split the devices string into a list of integers
    # device_ids = list(map(int, args.devices.split(",")))
    # Check if more than one GPU is selected
    # if len(device_ids) > 1:
    #     print(f"Multi-GPU training on devices: {device_ids}")
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)
    #     print(f"Model is on device: {next(model.parameters()).device}")

    do_pretrain(cfg=cfg,
                args=args,
                FILENAME_POSTFIX=FILENAME_POSTFIX,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                pretrain_loader=dataloader,
                pretrain_dataset=dataset,
                )

