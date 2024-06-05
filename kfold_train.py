from __future__ import print_function, division
import numpy as np
import pandas as pd
import shutil

import os, gc
import glob
from datetime import datetime
import argparse
import random
import yaml
import logging

from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.cuda import amp
import monai

from dataloaders.make_dataloaders import make_kfold_dataloaders
from models.make_models import make_vanilla_model
from do_train import do_train, do_inference
from utils.utils import EarlyStopping, load_pretrained_checkpoint
from utils.optimizers import make_optimizer

from ptflops import get_model_complexity_info

torch.multiprocessing.set_sharing_strategy('file_system')

# Set the seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    monai.utils.set_determinism(seed=seed)
    print('Seed is set.')


def setup_logger(FILENAME_POSTFIX, timestamp_current, args):
    # Set up the logger
    filename = glob.glob('./logs/' + FILENAME_POSTFIX + '*')[-1] if len(glob.glob('./logs/' + FILENAME_POSTFIX + '*')) > 0 else './logs/' + FILENAME_POSTFIX + '_' + timestamp_current + '.log'
    filemode = 'a' if len(glob.glob('./logs/' + FILENAME_POSTFIX + '*')) > 0 else 'w'
    logging.basicConfig(filename=filename,
                        format='%(asctime)s %(message)s',
                        filemode=filemode,
                        level=logging.DEBUG, 
                        force=True)
    logger = logging.getLogger()
    return logger

if __name__ == '__main__':

    # Parse some variable configs
    parser = argparse.ArgumentParser(description='Train UDA model for MRI imaging for classification of AD')
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Name of the config file')
    parser.add_argument('--savename', type=str, help='Experiment name (used for saving files)')
    parser.add_argument('--classes_to_use', nargs='+', type=str, help='Classes to use (enter by separating by space, e.g. CN AD MCI)')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--seed', type=int, default=42, help='Experiment seed (for reproducible results)')
    parser.add_argument('--devices', type=str, help='GPU devices to use')
    parser.add_argument('--checkpoint', default='./checkpoints/', type=str, help='Checkpoint model path')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate to use')
    parser.add_argument('--drop_path', type=float, default=0.1, help='Drop_path to use')
    parser.add_argument('--scheduler', type=str, default='cosine', help='Learning rate scheduler')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--size', type=str, default='base', help='base, small, large, tiny')
    parser.add_argument('--kernel_size', type=int, default=7, help='kernel_size = 7, 3, 5')
    parser.add_argument('--downsampling', type=str, default='avgpool3d', help='conv, avgpool3d, maxpool3d')
    parser.add_argument('--train_size', type=str, default='all', help='Train size: [0.2, 0.4, 0.6, 0.8, all]')
    parser.add_argument('--use_pretrained', type=str, help='Path to pre-trained model checkpoint to load')
    args = parser.parse_args()

    # Loads config file for fixed configs
    f_config = open(args.config_file,'rb')
    cfg = yaml.load(f_config, Loader=yaml.FullLoader) 

    # Set seed
    set_seed(args.seed)

    # Set up GPU devices to use
    if cfg['TRAINING']['USE_GPU']:
        print(f'Using GPU {args.devices}')
        os.environ["CUDA_DEVICE_ORDER"]=cfg['TRAINING']['CUDA_DEVICE_ORDER']
        os.environ["CUDA_VISIBLE_DEVICES"]=args.devices
    else:
        print('CPU mode')
    print(f'Process number: {os.getpid()} \n')

    df = pd.read_csv(cfg[args.dataset]['labelsroot'])
    df = df[df['Group'].isin(args.classes_to_use)]

    cfg['TRAINING']['EPOCHS'] = args.epochs
    cfg['SOLVER']['optimizer'] = args.optimizer
    cfg['SOLVER']['lr'] = args.lr
    cfg['SOLVER']['scheduler'] = args.scheduler
    cfg['DATALOADER']['train_size'] = args.train_size if args.train_size == 'all' else float(args.train_size)
    cfg['DATALOADER']['BATCH_SIZE'] = args.batch_size
    cfg['MODEL']['drop_path_rate'] = args.drop_path

    kfold_results = {"acc": [],
                     "bal_acc": [],
                     "auc": [],
                     "specificity": [],
                     "sensitivity": [],
                     "corrects": [],
                     "n_datapoints": [],
                     "ratios": []}

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=args.seed)

    for i, (train_index, test_index) in enumerate(skf.split(df, df['Group'])):
        FILENAME_POSTFIX = f'{args.savename}_{args.dataset}_seed_{args.seed}_fold_{i}'

        # Monai logs foldernames
        cfg['TRANSFORMS']['cache_dir_train'] = f'./monai_logs/train_{FILENAME_POSTFIX}'
        cfg['TRANSFORMS']['cache_dir_test'] = f'./monai_logs/test_{FILENAME_POSTFIX}'
        # Number of classes to use
        cfg['MODEL']['n_classes'] = len(args.classes_to_use)
        cfg['MODEL']['kernel_size'] = args.kernel_size
        cfg['MODEL']['downsampling'] = args.downsampling

        timestamp_current = datetime.now()
        timestamp_current = timestamp_current.strftime("%Y%m%d_%H%M")
        # Set up logger file
        logger = setup_logger(FILENAME_POSTFIX, timestamp_current, args)
        logger.setLevel(logging.DEBUG)
        logger.info('Process number: %d'%(os.getpid()))
        logger.info("Started training. Savename : " + args.savename)
        logger.info("Seed : " + str(args.seed))
        logger.info("Source dataset : " + args.dataset + ", Target dataset : " + args.dataset)

        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]

        print(f'Fold {i}: ')
        print(f'Train: {len(df_train)}, Test: {len(df_test)}')
        print(f'Train balance: {df_train["Group"].value_counts()}')
        print(f'Test balance: {df_test["Group"].value_counts()}')

        train_dataloader, test_dataloader, train_dataset, test_dataset, ratios_train, ratios_test = make_kfold_dataloaders(cfg, args, df_train, df_test)

        del df_train, df_test
        gc.collect()
    
        print(f'Number of classes to be used: {args.classes_to_use}, {cfg["MODEL"]["n_classes"]}')
        print(f'Train labels ratio: {ratios_train}, Test labels ratio: {ratios_test}')
        print(f'Train ratios (%): {[round(100*x/sum(ratios_train.values()), 2) for x in ratios_train.values()]}, Test ratios (%): {[round(100*x/sum(ratios_test.values()), 2) for x in ratios_test.values()]}')
        logger.info('Train set labels ratio: ' + str(ratios_train))
        logger.info('Test set labels ratio: ' + str(ratios_test))

        model = make_vanilla_model(cfg, args)
        
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(model, (1, 128, 128, 128), as_strings=True, print_per_layer_stat=False, verbose=False)

        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # Load pre-trained model weights (MAE)
        if args.use_pretrained is not None:
            model = load_pretrained_checkpoint(model, args.use_pretrained, cfg['TRAINING']['CHECKPOINT_TYPE'])
        
        # Move model to GPU
        if cfg['TRAINING']['USE_GPU']:
            model.cuda()

        optimizer = make_optimizer(cfg, args, model)
        scaler = amp.GradScaler()
        
        # Initialize loss function (with weight balance) and optimizer
        weight_balance = torch.Tensor(list(ratios_train.values())).cuda()
        criterion = nn.CrossEntropyLoss(weight=weight_balance)

        if len([x for x in args.devices.split(",")]) > 1: # if more than 1 GPU selected
            model = torch.nn.DataParallel(model)
            print('Multi-GPU training enabled.')
    
        # Early Stopper
        early_stopper = None
        if cfg['TRAINING']['EARLY_STOPPING']:
            early_stopper = EarlyStopping(patience=cfg['TRAINING']['EARLY_STOPPING_PATIENCE'], 
                                        min_delta=cfg['TRAINING']['EARLY_STOPPING_DELTA'])        
        
        # Save all configs and args just in case
        logger.info(cfg)
        logger.info(args)

        # Init wandb
        # wandb_setup(cfg, args, FILENAME_POSTFIX)

        trained_model = do_train(
            cfg, args, FILENAME_POSTFIX, model, criterion, optimizer, scaler, train_dataloader,
            train_dataset, logger, early_stopper, True, 
            test_dataloader
        )
        
        test_acc, bal_acc, test_auc, test_spec, test_sens, corrects, n_datapoints = do_inference(
            cfg, args, trained_model, test_dataloader, logger, False
        )

        kfold_results["acc"].append(test_acc)
        kfold_results["bal_acc"].append(round(bal_acc, 2))
        kfold_results["specificity"].append(round(test_spec, 2))
        kfold_results["sensitivity"].append(round(test_sens, 2))
        kfold_results["auc"].append(round(test_auc, 2))
        kfold_results["corrects"].append(corrects)
        kfold_results["n_datapoints"].append(n_datapoints)
        kfold_results["ratios"].append(ratios_test)

        logger.info(kfold_results)

        del model, trained_model, train_dataloader, test_dataloader, train_dataset, test_dataset

        shutil.rmtree(f'./monai_logs/train_{FILENAME_POSTFIX}')
        shutil.rmtree(f'./monai_logs/test_{FILENAME_POSTFIX}')

        # if i == 3:
        #     try:
        #         wandb.log({"k_fold acc": round(100*sum(kfold_results["corrects"])/sum(kfold_results["n_datapoints"]), 2)})
        #     except:
        #         pass

        # wandb.finish()


print(kfold_results)
print(f'k-fold acc: {round(100*sum(kfold_results["corrects"])/sum(kfold_results["n_datapoints"]), 2)}')
print(f'avg of test accs {round(sum(kfold_results["acc"])/4, 2)}')
print(f'avg bal acc {round(sum(kfold_results["bal_acc"])/4, 2)}')
print(f'avg auc {round(sum(kfold_results["auc"])/4, 2)}')
print(f'avg specificity {round(sum(kfold_results["specificity"])/4, 2)}')
print(f'avg sensitivity {round(sum(kfold_results["sensitivity"])/4, 2)}')





    






