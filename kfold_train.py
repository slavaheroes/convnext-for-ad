# -*- coding: utf-8 -*-
"""
Created on Sun Aug 9 2025
@author: qasymjomart
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import shutil
from datetime import datetime
import argparse
from loguru import logger
from tqdm import tqdm
import random
import yaml
from omegaconf import OmegaConf

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from monai.utils import set_determinism as monai_set_determinism

import torch
import torch.nn as nn
import time
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from lits import LitViT
from dataloaders.make_dataloaders import make_kfold_dataloaders, replace_data_path
from models.make_models import make_vanilla_model
from utils.prepare_model import prepare_model_for_training
from test_fn import test_adni2, test_aibl

import warnings 
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['WANDB_API_KEY'] = ""

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set the seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    monai_set_determinism(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print('Seed is set.')

def train():
    # Parse some variable configs
    parser = argparse.ArgumentParser(description='Train AD ViT model for MRI imaging for classification of AD')
    parser.add_argument('--config_file', type=str, default='configs/vitb_mae.yaml', help='Name of the config file')
    parser.add_argument('--savename', type=str, help='Experiment name (used for saving files)')
    parser.add_argument('--classes_to_use', nargs='+', type=str, help='Classes to use (enter by separating by space, e.g. CN AD MCI)')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--seed', type=int, help='Experiment seed (for reproducible results)')
    parser.add_argument('--devices', type=str, help='GPU devices to use')
    parser.add_argument('--patch_embed_fun', type=str, default='conv3d', help='Patch embed function to use')
    parser.add_argument('--checkpoint', default='./checkpoints/', type=str, help='Checkpoint model path')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate to use')
    parser.add_argument('--drop_path', type=float, default=0.1, help='Drop_path to use')
    parser.add_argument('--attn_p', type=float, default=0.1, help='Attn_p dropout to use')
    parser.add_argument('--p', type=float, default=0.0, help='Dropout rate to use')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size for ViT transformer')
    parser.add_argument('--scheduler', type=str, default='cosine', help='Learning rate scheduler')
    parser.add_argument('--train_size', type=str, default='all', help='Train size: [0.2, 0.4, 0.6, 0.8, all]')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--model_size', type=str, default='base', help='ViT base, small, large')
    parser.add_argument('--disable_qkv_bias', action='store_true', help='If set, will not use qkv bias in ViT model')
    parser.add_argument('--use_aug', action='store_true')
    parser.add_argument('--use_pretrained', type=str, help='Path to pre-trained model checkpoint to load')
    parser.add_argument('--mode', type=str, default='full', help='Mode: full, linear, etc.')
    parser.add_argument('--kernel_size', type=int, default=7, help='Kernel size for patch embedding conv layer (if patch_embed_fun is conv)')
    parser.add_argument('--downsampling', type=str, default='avgpool3d', help='Downsampling method for ViT (if patch_embed_fun is conv): conv or pool')
    args = parser.parse_args()

    # Loads config file for fixed configs
    cfg = OmegaConf.load(args.config_file)
    
    os.environ["WANDB_API_KEY"] = OmegaConf.load("keys.yaml")["WANDB_API_KEY"]
    
    # Set mode and modify model architecture accordingly
    cfg['mode'] = args.mode
    if cfg['mode'] in ['adapter', 'side', 'vpt']:
        cfg['model']['arch'] += f"_{cfg['mode']}"
        logger.info(f'Model architecture changed to {cfg["model"]["arch"]} due to mode {cfg["mode"]}')
        if cfg['mode'] == 'vpt':
            # load cfg of vpt in configs/config_vpt.yaml
            f_config_vpt = open('configs/config_vpt.yaml','rb')
            cfg_vpt = yaml.load(f_config_vpt, Loader=yaml.FullLoader)
            # every key-value under 'vpt' in cfg_vpt needs to be added to cfg['model']
            for key, value in cfg_vpt['vpt'].items():
                cfg['model'][key] = value
            logger.info(f'VPT configs added to model cfg: {cfg["model"]}')
            del cfg_vpt, f_config_vpt

    # Set seed
    set_seed(args.seed)

    # Set up GPU devices to use
    print(f'Using GPU {args.devices}')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    args.devices = [int(i) for i in args.devices.split(',')]
    print(f'Process number: {os.getpid()} \n')

    df = pd.read_csv(replace_data_path(cfg[args.dataset]['labelsroot']))
    df = df[df['Group'].isin(args.classes_to_use)]

    if cfg.model.arch == 'ConvNeXtV2_3D':
        cfg.model.downsampling = args.downsampling
        cfg.model.kernel_size = args.kernel_size
        cfg.model.padding = args.kernel_size // 2
    
    cfg['model']['patch_embed_fun'] = args.patch_embed_fun
    cfg['model']['patch_size'] = args.patch_size
    cfg['training']['epochs'] = args.epochs
    cfg['solver']['optimizer'] = args.optimizer
    cfg['solver']['lr'] = args.lr
    cfg['solver']['scheduler'] = args.scheduler
    cfg['training']['train_size'] = args.train_size if args.train_size == 'all' else float(args.train_size)
    cfg['training']['batch_size'] = args.batch_size
    cfg['model']['drop_path_rate'] =args.drop_path
    cfg['model']['attn_p'] = args.attn_p
    cfg['model']['p'] = args.p
    # cfg['model']['model_size'] = args.model_size
    cfg['model']['disable_qkv_bias'] = not args.disable_qkv_bias
    cfg['model']['n_classes'] = len(args.classes_to_use)
    cfg['mode'] = args.mode

    kfold_results = {"val_accs": [],
                     "best_epoch": [],
                     "recalls": [],
                     "f1s": [],
                     "corrects": [],
                     "n_datapoints": [],
                     "ratios": []}

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=args.seed)
    
    FILENAME = f'{args.savename}_{args.dataset}_seed_{args.seed}'
    if len(glob.glob(f'./checkpoints/{FILENAME}/fold_3/last.ckpt')) == 1:
        f = open(f'./results/kfold_results_{FILENAME}.txt', 'a+')
        f.seek(0)
        lines = f.readlines()
        if len(lines) == 0:
            logger.critical(f"No validation results found in the existing results file. Needs training: {FILENAME}")
            sys.exit(0)
        if not any('ADNI2 Test results' in line for line in lines):
            cfg['transforms']['cache_dir_test'] = f'./monai_cache/ADNI2_{FILENAME}'
            logger.warning(f'All folds for {FILENAME} already trained. Testing on ADNI2 now.')
            test_results_dict_adni2 = test_adni2(cfg, args, f'{FILENAME}')
            logger.success("Test results on ADNI2:", test_results_dict_adni2.values())
            f.write(f'ADNI2 Test results str:\n {str(test_results_dict_adni2)}\n')
            f.write('ADNI2 Test results:\n')
            for key, value in test_results_dict_adni2.items():
                f.write(f'{key}: {value}\n')
            
        if not any('AIBL Test results' in line for line in lines):
            cfg['transforms']['cache_dir_test'] = f'./monai_cache/AIBL_{FILENAME}'
            logger.warning(f'All folds for {FILENAME} already trained. No results on AIBL dataset. Testing on AIBL now.')
            test_results_dict_aibl = test_aibl(cfg, args, f'{FILENAME}')
            logger.success("Test results on AIBL:", test_results_dict_aibl.values())
            f.write(f'AIBL Test results str:\n {str(test_results_dict_aibl)}\n')
            f.write('AIBL Test results:\n')
            for key, value in test_results_dict_aibl.items():
                f.write(f'{key}: {value}\n')
        # if not any('AIBL Test results' in line for line in lines):
        #     
        #     logger.warning(f'All folds for {FILENAME} already trained. Testing on AIBL now.')
        f.close()
        sys.exit(0)

    # Init wandb
    wandb_logger = WandbLogger(project="AD-NEXT", 
                               name=FILENAME, 
                               tags=f'{args.savename}_{args.dataset}', config=OmegaConf.to_container(cfg))
    # wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"))

    for i, (train_index, test_index) in enumerate(skf.split(df, df['Group'])):
        # FILENAME_POSTFIX = args.savename + '_' + args.mode + '_seed_' + str(args.seed)
        timestamp_current = datetime.now()
        timestamp_current = timestamp_current.strftime("%Y%m%d_%H%M")
        
        FILENAME_POSTFIX = f'{FILENAME}_fold_{i}'
        logger.add(f'./logs/{FILENAME_POSTFIX}_{timestamp_current}.log', rotation="10 MB", level='TRACE')

        # Monai logs foldernames
        cfg['transforms']['cache_dir_train'] = f'./monai_cache/train_{FILENAME_POSTFIX}'
        cfg['transforms']['cache_dir_test'] = f'./monai_cache/test_{FILENAME_POSTFIX}'
        # Number of classes to use
        
        # Set up logger file
        logger.info(f'Process number: {os.getpid()}')
        logger.info(f"Started training. Savename : {args.savename}")
        logger.info(f"Seed : {args.seed}")
        logger.info(f"dataset dataset : {args.dataset}")

        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]

        logger.info(f'Fold {i}: ')
        logger.info(f'Train: {len(df_train)}, Test: {len(df_test)}')
        logger.info(f'Train balance: {df_train["Group"].value_counts()}')
        logger.info(f'Test balance: {df_test["Group"].value_counts()}')

        train_dataloader, test_dataloader, train_dataset, test_dataset, ratios_train, ratios_test = make_kfold_dataloaders(cfg, args, df_train, df_test)

        del df_train, df_test
    
        logger.info(f'Number of classes to be used: {args.classes_to_use}, {cfg["model"]["n_classes"]}')
        logger.info(f'Train labels ratio: {ratios_train}, Test labels ratio: {ratios_test}')
        logger.info(f'Train ratios (%): {[round(100*x/sum(ratios_train.values()), 2) for x in ratios_train.values()]}, Test ratios (%): {[round(100*x/sum(ratios_test.values()), 2) for x in ratios_test.values()]}')
        logger.info(f'Train set labels ratio: {ratios_train}')
        logger.info(f'Test set labels ratio: {ratios_test}')

        ### MODEL ####
        model = make_vanilla_model(cfg, args)
        
        model = prepare_model_for_training(model, cfg)
        
        # Initialize loss function (with weight balance)
        class_numbers = []
        for class_name in args.classes_to_use:
            class_numbers += [args.classes_to_use.index(class_name)] * ratios_train[class_name]
        class_weights = torch.Tensor(compute_class_weight(class_weight='balanced', classes=np.unique(class_numbers), y=class_numbers)).cuda()
        logger.info(f'Class weights: {class_weights}')
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Save all configs and args just in case
        logger.info(cfg)
        logger.info(args)

        lit_vit = LitViT(model, cfg['model']['n_classes'],
                         pretrained_model_path=args.use_pretrained,
                         loss_fn=criterion, 
                         learning_rate=cfg['solver']['lr'],
                         weight_decay=cfg['solver']['weight_decay'], 
                         betas=tuple(cfg['solver']['betas']),
                         epochs=cfg['training']['epochs'],
                         mode=cfg['mode'],
                         fold_i=i)

        if os.path.exists(f'checkpoints/{FILENAME}/fold_{i}'):
            logger.info(f'Checkpoint folder for fold {i} already exists. Removing it.')
            # pause for 5 sec
            time.sleep(5)
            shutil.rmtree(f'checkpoints/{FILENAME}/fold_{i}')
        
        os.makedirs(f'checkpoints/{FILENAME}/fold_{i}')
        checkpoint_callback = ModelCheckpoint(
            monitor=f'val_acc_fold{i}',
            dirpath=f'checkpoints/{FILENAME}/fold_{i}/',
            filename='best-{epoch:02d}',
            save_weights_only=True,
            save_top_k=1,
            mode='max',
            save_last=True,
            verbose=True
        )
                
        trainer = L.Trainer(max_epochs=cfg['training']['epochs'],
                            # default_root_dir=f'checkpoints/{FILENAME}/fold_{i}/',
                            accelerator='gpu',
                            devices=[0],
                            num_sanity_val_steps=0,
                            # accumulate_grad_batches=4//cfg['training']['batch_size'],
                            logger=wandb_logger,
                            callbacks=[checkpoint_callback],
                            )
        
        trainer.fit(lit_vit, train_dataloader, val_dataloaders=test_dataloader)
        
        kfold_results["val_accs"].append(max(lit_vit.val_accs))
        # recall, f1, and corrects will be selected from the best epoch which is max val_acc
        best_epoch = lit_vit.val_accs.index(max(lit_vit.val_accs))
        kfold_results["best_epoch"].append(best_epoch)
        kfold_results["recalls"].append(lit_vit.recalls[best_epoch])
        kfold_results["f1s"].append(lit_vit.f1s[best_epoch])
        kfold_results["corrects"].append(lit_vit.corrects[best_epoch])
        n_datapoints = len(test_dataset)
        kfold_results["n_datapoints"].append(n_datapoints)
        kfold_results["ratios"].append(ratios_test)
        
        logger.info(kfold_results)

        del model, train_dataloader, test_dataloader, train_dataset, test_dataset

        if os.path.exists(f'./monai_cache/train_{FILENAME_POSTFIX}'):
            shutil.rmtree(f'./monai_cache/train_{FILENAME_POSTFIX}')
            shutil.rmtree(f'./monai_cache/test_{FILENAME_POSTFIX}')
    
    if not any(sweep in FILENAME for sweep in ['LR', 'fullaug', 'mT0d', 'wTT', 'patchembed', 'posembed', 'predratio', 'EPS', 'cM0d', 'NoKoleo', 'NoiBOT']):
        test_results_dict_adni2 = test_adni2(cfg, args, f'{FILENAME}')
        test_results_dict_aibl = test_aibl(cfg, args, f'{FILENAME}')
    
    # after all folds are done, print the results
    print(kfold_results)
    print(f'k-fold val acc: {sum(kfold_results["corrects"])/sum(kfold_results["n_datapoints"]):.2f}')
    print(f'avg of val accs {round(sum(kfold_results["val_accs"])/4, 2)} ± {round(np.std(kfold_results["val_accs"]), 2)}')
    print(f'avg of val recalls {round(sum(kfold_results["recalls"])/4, 2)} ± {round(np.std(kfold_results["recalls"]), 2)}')
    print(f'avg of val f1s {round(sum(kfold_results["f1s"])/4, 2)} ± {round(np.std(kfold_results["f1s"]), 2)}')

    print("FOR COPY-PASTE:")
    print(f"VAL ACC: {kfold_results['val_accs']}")
    if not any(sweep in FILENAME for sweep in ['LR', 'fullaug', 'mT0d', 'wTT', 'patchembed', 'posembed', 'predratio', 'EPS', 'cM0d', 'NoKoleo', 'NoiBOT']):
        print(f"ADNI2 TEST RESULTS: {test_results_dict_adni2.values()}")
        print(f"AIBL TEST RESULTS: {test_results_dict_aibl.values()}")
    
    # save kfold results acc into a file
    os.makedirs('./results', exist_ok=True)
    with open(f'./results/kfold_results_{FILENAME}.txt', 'w') as f:
        f.write(f'{FILENAME}\n')
        f.write(str(kfold_results) + '\n')
        f.write(f'k-fold acc: {sum(kfold_results["corrects"])/sum(kfold_results["n_datapoints"]):.2f}')
        f.write(f'avg of val accs {round(sum(kfold_results["val_accs"])/4, 2)} ± {round(np.std(kfold_results["val_accs"]), 2)}')
        f.write(f'avg of recalls {round(sum(kfold_results["recalls"])/4, 2)} ± {round(np.std(kfold_results["recalls"]), 2)}')
        f.write(f'avg of f1s {round(sum(kfold_results["f1s"])/4, 2)} ± {round(np.std(kfold_results["f1s"]), 2)}\n\n')
        if not any(sweep in FILENAME for sweep in ['LR', 'fullaug', 'mT0d', 'wTT', 'patchembed', 'posembed', 'predratio', 'EPS', 'cM0d', 'NoKoleo', 'NoiBOT']):
            f.write(f'ADNI2 Test results str:\n {str(test_results_dict_adni2)}\n')
            f.write('ADNI2 Test results:\n')
            for key, value in test_results_dict_adni2.items():
                f.write(f'{key}: {value}\n')
            f.write(f'\nAIBL Test results str:\n {str(test_results_dict_aibl)}\n')
            f.write('AIBL Test results:\n')
            for key, value in test_results_dict_aibl.items():
                f.write(f'{key}: {value}\n')
    
    # save results into results_bank.csv
    results_bank_df = pd.DataFrame({
        'Experiment': [FILENAME],
        'kfold_acc': [sum(kfold_results["corrects"])/sum(kfold_results["n_datapoints"])],
        'fold1_val_acc': [kfold_results["val_accs"][0]],
        'fold2_val_acc': [kfold_results["val_accs"][1]],
        'fold3_val_acc': [kfold_results["val_accs"][2]],
        'fold4_val_acc': [kfold_results["val_accs"][3]],
        'val_acc_mean': [round(sum(kfold_results["val_accs"])/4, 4)],
        'val_acc_std': [round(np.std(kfold_results["val_accs"]), 4)],
        'val_recall_mean': [round(sum(kfold_results["recalls"])/4, 4)],
        'val_recall_std': [round(np.std(kfold_results["recalls"]), 4)],
        'val_f1_mean': [round(sum(kfold_results["f1s"])/4, 4)],
        'val_f1_std': [round(np.std(kfold_results["f1s"]), 4)],
    })
    if not any(sweep in FILENAME for sweep in ['LR', 'fullaug', 'mT0d', 'wTT', 'patchembed', 'posembed', 'predratio', 'EPS', 'cM0d', 'NoKoleo', 'NoiBOT']):
        results_bank_df['ADNI2_acc_soft'] = test_results_dict_adni2['acc_soft']
        results_bank_df['ADNI2_acc_hard'] = test_results_dict_adni2['acc_hard']
        results_bank_df['ADNI2_recall_soft'] = test_results_dict_adni2['recall_soft']
        results_bank_df['ADNI2_recall_hard'] = test_results_dict_adni2['recall_hard']
        results_bank_df['ADNI2_f1_soft'] = test_results_dict_adni2['f1_soft']
        results_bank_df['ADNI2_f1_hard'] = test_results_dict_adni2['f1_hard']
        results_bank_df['AIBL_acc_soft'] = test_results_dict_aibl['acc_soft']
        results_bank_df['AIBL_acc_hard'] = test_results_dict_aibl['acc_hard']
        results_bank_df['AIBL_recall_soft'] = test_results_dict_aibl['recall_soft']
        results_bank_df['AIBL_recall_hard'] = test_results_dict_aibl['recall_hard']
        results_bank_df['AIBL_f1_soft'] = test_results_dict_aibl['f1_soft']
        results_bank_df['AIBL_f1_hard'] = test_results_dict_aibl['f1_hard']
    
    # check if results_bank.csv exists
    if os.path.exists('./results/results_bank.csv'):
        try:
            results_bank_df_existing = pd.read_csv('./results/results_bank.csv')
            results_bank_df = pd.concat([results_bank_df_existing, results_bank_df], ignore_index=True)
        except Exception as e:
            print(f'Error reading existing results_bank.csv: {e}')
    results_bank_df.to_csv('./results/results_bank.csv', index=False)

if __name__ == '__main__':
    train()
    