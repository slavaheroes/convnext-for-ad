import glob
import os
import shutil
import torch
from torchmetrics.classification import Accuracy, F1Score, Recall
from torchmetrics import ConfusionMatrix
from tqdm import tqdm

from dataloaders.make_dataloaders import make_adni2_test_dataloader, make_aibl_test_dataloader
from lits import LitViT 

def test_aibl(cfg, args, checkpoints_dir):
    models_files = glob.glob(f'checkpoints/{checkpoints_dir}/fold_*/best-*.ckpt')
    assert len(models_files) == 4, "There should be 4 model files for 4 folds."
    
    test_dataloader, test_dataset, ratios_test = make_aibl_test_dataloader(cfg, args, verbose=True)
    batch_size = test_dataloader.batch_size
    
    # corrects = 0
    # acc_fn, recall_fn, f1_fn =  Accuracy(task='multiclass', num_classes=cfg['model']['n_classes'], average='macro'), \
    #                             Recall(task='multiclass', num_classes=cfg['model']['n_classes'], average='macro'), \
    #                             F1Score(task='multiclass', num_classes=cfg['model']['n_classes'], average='macro')
    acc_fn, recall_fn, f1_fn =  Accuracy(task='binary'), \
                                Recall(task='binary'), \
                                F1Score(task='binary')
    
    confmat = ConfusionMatrix(task='binary', num_classes=cfg['model']['n_classes'])
    
    outputs_softmax = torch.zeros((len(models_files), len(test_dataset), cfg['model']['n_classes']))
    outputs_pred_labels = torch.zeros(len(models_files), len(test_dataset)).long()
    gt_labels = torch.zeros(len(test_dataset)).long()
    for ii in range(len(models_files)):
        model = LitViT.load_from_checkpoint(models_files[ii], pretrained_model_path=None).eval()
        print(f'Loaded model from {models_files[ii]}')
        for idx, batch in enumerate(tqdm(test_dataloader, desc=f'Testing AIBL with model fold {ii}')):
            images, labels = batch['image'].cuda(model.device), batch['label'].cuda(model.device)
            assert images is not None and labels is not None, "Images or labels in the batch are None."
            # Average predictions from all models
            with torch.no_grad():
                out = model(images)
                out_softmax = torch.nn.functional.softmax(out, dim=1)
                outputs_softmax[ii, idx*batch_size:idx*batch_size+images.shape[0], :] = out_softmax.cpu()
                
                _, preds = torch.max(out, 1)
                outputs_pred_labels[ii, idx*batch_size:idx*batch_size+images.shape[0]] = preds.cpu()
                
                gt_labels[idx*batch_size:idx*batch_size+images.shape[0]] = labels.cpu()
            
    # Now average the outputs_softmax over models
    outputs_softmax_mean = torch.mean(outputs_softmax, dim=0) # shape: (n_samples, n_classes)
    outputs_final_preds = torch.argmax(outputs_softmax_mean, dim=1) # shape:
    
    # hard voting
    outputs_hard_preds = torch.mode(outputs_pred_labels, dim=0).values # shape: (n_samples,)
        
    # Compare soft and hard voting results
    acc_soft = acc_fn(outputs_final_preds, gt_labels).item()
    recall_soft = recall_fn(outputs_final_preds, gt_labels).item()
    f1_soft = f1_fn(outputs_final_preds, gt_labels).item()
    acc_fn.reset(), recall_fn.reset(), f1_fn.reset()
    acc_hard = acc_fn(outputs_hard_preds, gt_labels).item()
    recall_hard = recall_fn(outputs_hard_preds, gt_labels).item()
    f1_hard = f1_fn(outputs_hard_preds, gt_labels).item()
    
    conf_mat_soft = confmat(outputs_final_preds, gt_labels)
    conf_mat_hard = confmat(outputs_hard_preds, gt_labels)
    
    print(f'AIBL Test Accuracy (soft voting): {acc_soft:.4f}, Recall (soft voting): {recall_soft:.4f}, F1 (soft voting): {f1_soft:.4f}')
    print(f'AIBL Test Accuracy (hard voting): {acc_hard:.4f}, Recall (hard voting): {recall_hard:.4f}, F1 (hard voting): {f1_hard:.4f}')
    print('Confusion Matrix (soft voting):', conf_mat_soft)
    print('Confusion Matrix (hard voting):', conf_mat_hard)
    
    # delete monai cache
    if os.path.exists(cfg['transforms']['cache_dir_test']):
        shutil.rmtree(cfg['transforms']['cache_dir_test'])
    
    return {"acc_soft": acc_soft, "acc_hard": acc_hard, "recall_soft": recall_soft, "recall_hard": recall_hard, "f1_soft": f1_soft, "f1_hard": f1_hard, "conf_mat_soft": conf_mat_soft, "conf_mat_hard": conf_mat_hard}

def test_adni2(cfg, args, checkpoints_dir):
    models_files = glob.glob(f'checkpoints/{checkpoints_dir}/fold_*/best-*.ckpt')
    assert len(models_files) == 4, "There should be 4 model files for 4 folds."
    
    test_dataloader, test_dataset, ratios_test = make_adni2_test_dataloader(cfg, args, verbose=True)
    batch_size = test_dataloader.batch_size
    
    # corrects = 0
    acc_fn, recall_fn, f1_fn =  Accuracy(task='binary'), \
                                Recall(task='binary'), \
                                F1Score(task='binary')
    
    confmat = ConfusionMatrix(task='binary', num_classes=cfg['model']['n_classes'])
    
    outputs_softmax = torch.zeros((len(models_files), len(test_dataset), cfg['model']['n_classes']))
    outputs_pred_labels = torch.zeros(len(models_files), len(test_dataset)).long()
    gt_labels = torch.zeros(len(test_dataset)).long()
    for ii in range(len(models_files)):
        model = LitViT.load_from_checkpoint(models_files[ii], pretrained_model_path=None).eval()
        print(f'Loaded model from {models_files[ii]}')
        for idx, batch in enumerate(tqdm(test_dataloader, desc=f'Testing ADNI2 with model fold {ii}')):
            images, labels = batch['image'].cuda(model.device), batch['label'].cuda(model.device)
            assert images is not None and labels is not None, "Images or labels in the batch are None."
            # Average predictions from all models
            with torch.no_grad():
                out = model(images)
                out_softmax = torch.nn.functional.softmax(out, dim=1)
                outputs_softmax[ii, idx*batch_size:idx*batch_size+images.shape[0], :] = out_softmax.cpu()
                
                _, preds = torch.max(out, 1)
                outputs_pred_labels[ii, idx*batch_size:idx*batch_size+images.shape[0]] = preds.cpu()
                
                gt_labels[idx*batch_size:idx*batch_size+images.shape[0]] = labels.cpu()
            
    # Now average the outputs_softmax over models
    outputs_softmax_mean = torch.mean(outputs_softmax, dim=0) # shape: (n_samples, n_classes)
    outputs_final_preds = torch.argmax(outputs_softmax_mean, dim=1) # shape: (n_samples,)
    
    # hard voting
    outputs_hard_preds = torch.mode(outputs_pred_labels, dim=0).values # shape: (n_samples,)
        
    # Compare soft and hard voting results
    acc_soft = acc_fn(outputs_final_preds, gt_labels).item()
    recall_soft = recall_fn(outputs_final_preds, gt_labels).item()
    f1_soft = f1_fn(outputs_final_preds, gt_labels).item()
    acc_fn.reset(), recall_fn.reset(), f1_fn.reset()
    acc_hard = acc_fn(outputs_hard_preds, gt_labels).item()
    recall_hard = recall_fn(outputs_hard_preds, gt_labels).item()
    f1_hard = f1_fn(outputs_hard_preds, gt_labels).item()
    
    conf_mat_soft = confmat(outputs_final_preds, gt_labels)
    conf_mat_hard = confmat(outputs_hard_preds, gt_labels)
    
    print(f'ADNI2 Test Accuracy (soft voting): {acc_soft:.4f}, Recall (soft voting): {recall_soft:.4f}, F1 (soft voting): {f1_soft:.4f}')
    print(f'ADNI2 Test Accuracy (hard voting): {acc_hard:.4f}, Recall (hard voting): {recall_hard:.4f}, F1 (hard voting): {f1_hard:.4f}')
    print('Confusion Matrix (soft voting):', conf_mat_soft)
    print('Confusion Matrix (hard voting):', conf_mat_hard)
    
    # delete monai cache
    if os.path.exists(cfg['transforms']['cache_dir_test']):
        shutil.rmtree(cfg['transforms']['cache_dir_test'])
    
    return {"acc_soft": acc_soft, "acc_hard": acc_hard, "recall_soft": recall_soft, "recall_hard": recall_hard, "f1_soft": f1_soft, "f1_hard": f1_hard, "conf_mat_soft": conf_mat_soft, "conf_mat_hard": conf_mat_hard}