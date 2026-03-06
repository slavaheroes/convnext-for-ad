# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 2023

@author: qasymjomart
"""

from __future__ import print_function, division
from sklearn.metrics import balanced_accuracy_score

import os, gc
from time import time
# import wandb

import torch
import torch.nn.functional as F
from torch.cuda import amp
import torchmetrics

from utils.utils import save_model, load_checkpoint


def do_train(cfg, args, FILENAME_POSTFIX, model, criterion, optimizer, scaler, source_loader, 
             source_dataset, logger, early_stopper, do_valid, 
             test_dataloader):
    """
    Do vanilla mode training

    """
    # Read batch size
    batch_size = cfg['DATALOADER']['BATCH_SIZE']

    # Calculate iter per epoch
    N_src = source_dataset.__len__()
    iter_per_epoch = source_dataset.__len__()/batch_size

    # Read epochs
    epochs = cfg['TRAINING']['EPOCHS']

    # Train the Model
    batch_time, net_time = [], []

    iter_start = 0
    steps = 0
    
    # performance metrics helpers
    train_acc, val_acc = 0, 0
    average_loss = 0
    correct = 0

    if cfg['SOLVER']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(int(iter_start/iter_per_epoch), epochs):
        model.train()
        end = time()

        # Set the learning rate for each layer based on the decay factor
        # if cfg['TRAINING']['LAYERWISE_LR_DECAY']:
        #     for idx, param_group in enumerate(optimizer.param_groups):
        #         param_group['lr'] = optimizer.param_groups[idx]['lr'] * cfg['TRAINING']['LAYERWISE_LR_DECAY']

        for batch_data in source_loader:
            batch_time.append(time()-end)
            if len(batch_time)>100:
                del batch_time[0]
            
            # adjust_learning_rate_halfcosine(optimizer, steps / len(source_loader) + epoch, cfg)

            if cfg['TRAINING']['USE_GPU']:
                images, labels = (
                    batch_data["image"].cuda(non_blocking=True),
                    batch_data["label"].cuda(non_blocking=True)
                )
                                    
            t = time()
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels.long())

            # Forward + Backward + Optimize
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            lr = optimizer.param_groups[0]["lr"]
            torch.cuda.synchronize()

            net_time.append(time()-t)
            if len(net_time)>100:
                del net_time[0]
            
            # other way to calculate accuracy
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            average_loss += float(loss.item())
            
            steps += 1
            # wandb.log({"train loss": loss.item()})

            end = time()
        
        # save_model(args, cfg, model, cfg['TRAINING']['CHECKPOINT'] + FILENAME_POSTFIX, epoch, steps)
        # print(f'Saved: {cfg["TRAINING"]["CHECKPOINT"]}{FILENAME_POSTFIX}_{epoch}_{steps}')

        if do_valid:
            test_acc, balanced_acc, test_auc, test_spec, test_sens, _, _ = do_inference(cfg, args, model, test_dataloader, logger, is_validation=True)
            model.train()
            if val_acc < test_acc:
                val_acc = test_acc
                # saving the best model by first removing previous best models (for saving memory)
                save_model(args, cfg, model, cfg['TRAINING']['CHECKPOINT'] + 'BEST_MODEL_' + FILENAME_POSTFIX, epoch, steps, remove_previous=True)
                print(f'Best model saved: {cfg["TRAINING"]["CHECKPOINT"]}{FILENAME_POSTFIX}_{epoch}_{steps}')

        # Printing train stats and logging
        print(f'[{epoch+1}/{epochs}] {steps}) LR {lr:.7f}, Loss: {average_loss/iter_per_epoch:.3f}, Acc {100*correct/N_src:.2f}%, N {N_src}, Test Acc {test_acc:.2f}%, Bal acc {balanced_acc:.2f}%')
        print(f'[{epoch+1}/{epochs}] {steps}) Test Spec {test_spec:.2f}%, Test Sens {test_sens:.2f}%, Test Auc {test_auc:.2f}%')
        
        logger.info(f'[{epoch+1}/{epochs}] {steps}) LR {lr:.7f}, Loss: {average_loss/iter_per_epoch:.3f}, Acc {100*correct/N_src:.2f}%, N {N_src}, Test Acc {test_acc:.2f}%, Bal acc {balanced_acc:.2f}%')
        logger.info(f'[{epoch+1}/{epochs}] {steps}) Test Spec {test_spec:.2f}%, Test Sens {test_sens:.2f}%, Test Auc {test_auc:.2f}%')
        
        torch.cuda.empty_cache()

        # wandb.log({"lr": lr, "epoch": epoch+1, "avg loss": average_loss/iter_per_epoch, 
        #            "train acc": 100*correct/N_src, "test acc": test_acc, "bal acc": balanced_acc, "test auc": test_auc})
        average_loss, correct = 0, 0
                
        if epoch > 20 and cfg['TRAINING']['EARLY_STOPPING']:
            early_stopper(test_acc)
            if early_stopper.early_stop:
                print('Early stopping and saving the model...')
                # saving the last model by first removing previous models (for saving memory)
                break
        
        if cfg['SOLVER']['scheduler'] == 'cosine':                
            scheduler.step()
        
        if os.path.exists(cfg['TRAINING']['CHECKPOINT']+'/stop.txt'):
            # break without using CTRL+C
            # just create stop.txt file in cfg['TRAINING']['CHECKPOINT']
            break
        
        if os.path.exists(cfg['TRAINING']['CHECKPOINT']+'/pdb.txt'):
            import pdb; pdb.set_trace()
    
    # Load the best-performing model
    model = load_checkpoint(args, cfg, model, 'BEST_MODEL_' + FILENAME_POSTFIX)

    return model

@torch.no_grad()
def do_inference(cfg, args, model, test_loader, logger, is_validation=True):
    """

    Parameters
    ----------
    cfg : config yaml
        Config read in yaml format file
    args : argument parser
        Arguments red from command line
    test_loader : torch.Dataloader
        test dataset loader

    Returns
    -------
    acc : float
        accuracy of inference
    """
    
    corrects = 0
    auroc_calc = torchmetrics.AUROC(task="multiclass", num_classes=len(args.classes_to_use))
    specificity = torchmetrics.Specificity(task="binary", num_classes=len(args.classes_to_use))
    sensitivity = torchmetrics.Recall(task="binary", num_classes=len(args.classes_to_use))
    
    logits_arr, preds_arr, labels_arr = [], [], []
    n_datapoints = 0
    model.eval()
    for batch_data in test_loader:

        if cfg['TRAINING']['USE_GPU']:
            images = batch_data["image"].cuda(non_blocking=True)
            
        labels = batch_data["label"] # (1)
        
        outputs = model(images) # (1,2))
        logits = F.softmax(outputs, dim=-1).cpu().data # (1,2)

        logits_arr.append(logits.view(-1).detach().cpu().tolist())
        labels_arr.append(labels.view(-1).detach().cpu().tolist())
        
        preds = torch.argmax(logits, dim=1)
        corrects += torch.sum(preds == labels)
        n_datapoints += outputs.shape[0]

        preds_arr.append(preds.view(-1).detach().cpu().tolist())
        
    test_acc = corrects.item()/n_datapoints
    test_auc = auroc_calc(torch.tensor(logits_arr), torch.tensor(labels_arr).squeeze()).item()
    test_spec = specificity(torch.tensor(logits_arr).argmax(dim=1), torch.tensor(labels_arr).squeeze()).item()
    test_sens = sensitivity(torch.tensor(logits_arr).argmax(dim=1), torch.tensor(labels_arr).squeeze()).item()
    
    balanced_acc = balanced_accuracy_score(labels_arr, preds_arr)
    
    logger.info(f'TESTING: Number of datapoints: {n_datapoints}), Accuracy {100*test_acc:.2f}%, Bal ACC {100*balanced_acc:.2f}%, AUC {100*test_auc:.2f}%\
                 Spec {100*test_spec:.2f}%, Sens {100*test_sens:.2f}%')

    if not is_validation:
        print(f'TESTING: Number of datapoints: {n_datapoints}), Accuracy {100*test_acc:.2f}%, Bal ACC {100*balanced_acc:.2f}%, AUC {100*test_auc:.2f}%\
                 Spec {100*test_spec:.2f}%, Sens {100*test_sens:.2f}%')

    del logits_arr, labels_arr, preds_arr, auroc_calc
    gc.collect()

    # test_acc, balanced_acc, test_auc, test_spec, test_sens, _, _
    return 100*test_acc, 100*balanced_acc, 100*test_auc, 100*test_spec, 100*test_sens, corrects.item(), n_datapoints
