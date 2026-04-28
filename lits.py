from loguru import logger
import wandb
import torch

import lightning as L
from torchmetrics.classification import Accuracy, F1Score, Recall
    
class LitViT(L.LightningModule):
    def __init__(self, 
                 model,
                 num_classes,
                 loss_fn,
                 optimizer_type='adamw',
                 learning_rate=0.0001,
                 mode='full', # 'full' for finetuning, 'linear' for linear evaluation, etc.
                 fold_i=None,
                 **kwargs):
        super().__init__()
        self.model = model
        self.num_classes = num_classes if num_classes is not None else 2
        if kwargs.get('pretrained_model_path') is not None:
            self.copy_pretrained_weights(kwargs['pretrained_model_path'])
        # self.copy_pretrained_weights() ##
        # optimizer parameters
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = kwargs.get('weight_decay', 0.0001)
        self.betas = kwargs.get('betas', (0.9, 0.999))
        self.epochs = kwargs.get('epochs', 50)
        self.fold_i = fold_i
        
        self.mode = mode
        
        self.save_hyperparameters()
        # loss
        self.loss_fn = loss_fn
        self.train_acc_fn = Accuracy(task='multiclass', num_classes=self.num_classes, average='micro')
        self.val_acc_fn = Accuracy(task='multiclass', num_classes=self.num_classes, average='micro') # same as binary
        self.f1_fn = F1Score(task='multiclass', num_classes=self.num_classes, average=None) # same as binary
        self.recall_fn = Recall(task='multiclass', num_classes=self.num_classes, average=None) # same as binary
        
        # logs
        self.correct = []
        self.val_accs, self.recalls, self.f1s, self.corrects = [], [], [], []
        
    def copy_pretrained_weights(self, pre_trained_model_path):
        # load pretrained weights
        try:
            checkpoint = torch.load(pre_trained_model_path, map_location='cpu')
        except:
            checkpoint = torch.load(pre_trained_model_path, map_location='cpu', weights_only=False)
        
        # if 'SimCLR' in pre_trained_model_path and pre_trained_model_path.endswith('.pth.tar'):
        #     checkpoint_model = checkpoint
        #     for key in list(checkpoint_model.keys()):
        #         if key.startswith('backbone.'):
        #             checkpoint_model[key[len('backbone.'):]] = checkpoint_model.pop(key)
        #     logger.success(f"Loading SimCLR pretrained weights from {pre_trained_model_path}...")
        if pre_trained_model_path.endswith('.pth.tar'):
            checkpoint_model = checkpoint['net']
        elif any(keyword in pre_trained_model_path for keyword in ['FCMAE_3D']) and pre_trained_model_path.endswith('.pth'):
            from utils.utils import load_pretrained_checkpoint
            if 'NoSparse' in pre_trained_model_path:
                self.model = load_pretrained_checkpoint(self.model, pre_trained_model_path, 'ConvNext')
            else:
                self.model = load_pretrained_checkpoint(self.model, pre_trained_model_path, 'ConvNext_sparse')
            logger.critical(f'Loaded pretrained weights from {pre_trained_model_path} using custom loading function for ConvNext')
            return
        elif any(keyword in pre_trained_model_path for keyword in ['MAE', 'SimMIM', 'RubiksCube', 'UNETR']) and pre_trained_model_path.endswith('.pth'):
            checkpoint_model = checkpoint['net']
            for key in list(checkpoint_model.keys()):
                if key.startswith('module.'):
                    checkpoint_model[key[len('module.'):]] = checkpoint_model.pop(key)
            for key in list(checkpoint_model.keys()):
                if key.startswith('encoder.'):
                    checkpoint_model[key[len('encoder.'):]] = checkpoint_model.pop(key)
            for key in list(checkpoint_model.keys()):
                if key.startswith('backbone.'):
                    checkpoint_model[key[len('backbone.'):]] = checkpoint_model.pop(key)
            for key in list(checkpoint_model.keys()):
                if key.startswith('net.'):
                    checkpoint_model[key[len('net.'):]] = checkpoint_model.pop(key)
            # keys_to_remove = ['head.weight', 'head.bias']
            # for k in keys_to_remove:
            #     if k in checkpoint_model and k in self.model.state_dict() and checkpoint_model[k].shape != self.model.state_dict()[k].shape:
            #         logger.warning(f"Removing key {k} from pretrained checkpoint")
            #         del checkpoint_model[k]
        elif any(keyword in pre_trained_model_path for keyword in ['DINO', 'DINOv2', 'MOCOv3', 'BYOL', 'iBOT', 'MoCov3', 'SwAV', 'SimCLR', 'SimSiam', 'BarlowTwins', 'VICReg']) and pre_trained_model_path.endswith('.pth'):
            checkpoint_model = checkpoint['student']
            for key in list(checkpoint_model.keys()):
                if key.startswith('module.'):
                    checkpoint_model[key[len('module.'):]] = checkpoint_model.pop(key)
            for key in list(checkpoint_model.keys()):
                if key.startswith('backbone.'):
                    checkpoint_model[key[len('backbone.'):]] = checkpoint_model.pop(key)
            for key in list(checkpoint_model.keys()):
                if key.startswith('net.'):
                    checkpoint_model[key[len('net.'):]] = checkpoint_model.pop(key)
            # import ipdb; ipdb.set_trace()
        elif pre_trained_model_path.endswith('.ckpt'):
            checkpoint_model = checkpoint['state_dict']
            for key in list(checkpoint_model.keys()):
                if key.startswith('student.'):
                    checkpoint_model[key[len('student.'):]] = checkpoint_model.pop(key)
            for key in list(checkpoint_model.keys()):
                if key.startswith('backbone.'):
                    checkpoint_model[key[len('backbone.'):]] = checkpoint_model.pop(key)
        # remove head keys that are not in the model because head sizes may differ (e.g. SimMIM)
        keys_to_remove = ['head.weight', 'head.bias']
        state_dict = self.model.state_dict()
        for key in keys_to_remove:
            if key in checkpoint_model and key in state_dict and state_dict[key].shape != checkpoint_model[key].shape:
                print(f"Removing key {key} from pretrained checkpoint")
                del checkpoint_model[key]
        # load the model
        msg = self.model.load_state_dict(checkpoint_model, strict=False)
        logger.critical(f'Loaded pretrained weights from {pre_trained_model_path}')
        logger.critical(f'Keys not loaded (missing keys): {msg.missing_keys}')        
        logger.critical(f'Keys not used (unexpected keys): {msg.unexpected_keys}')    
    
    def on_train_epoch_end(self):
        # get max hard dice score across steps
        train_acc = self.train_acc_fn.compute().item()
        self.log(f'train_acc_{self.fold_i}', train_acc, prog_bar=True)
        self.train_acc_fn.reset()
        
    def on_validation_epoch_end(self):
        # get max hard dice score from validation
        val_acc = self.val_acc_fn.compute().item()
        recall = self.recall_fn.compute()[1].item() # [1] for class 1
        f1 = self.f1_fn.compute()[1].item() # [1] for class 1
        
        self.log_dict({f'val_acc_fold{self.fold_i}': val_acc,
                        f'recall_fold{self.fold_i}': recall,
                        f'f1_fold{self.fold_i}': f1}, prog_bar=True)
        # log metrics to array
        self.val_accs.append(val_acc)
        self.recalls.append(recall)
        self.f1s.append(f1)
        self.corrects.append(sum(self.correct))  # sum of corrects across all batches for this epoch
        
        # we have to reset the metrics after logging
        self.val_acc_fn.reset()
        self.recall_fn.reset()
        self.f1_fn.reset()
        self.correct = []  # reset correct for the next epoch
    
    def on_train_end(self):
        # log the best metrics
        if len(self.val_accs) > 0:
            best_val_acc = max(self.val_accs)
            best_epoch = self.val_accs.index(best_val_acc) + 1
            best_recall = self.recalls[best_epoch - 1]
            best_f1 = self.f1s[best_epoch - 1]
            logger.critical(f'Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}')
            logger.critical(f'Best recall: {best_recall:.4f} at epoch {best_epoch}')
            logger.critical(f'Best f1: {best_f1:.4f} at epoch {best_epoch}')
            
            wandb.run.summary[f'best_val_acc_fold{self.fold_i}'] = best_val_acc
            wandb.run.summary[f'best_recall_fold{self.fold_i}'] = best_recall
            wandb.run.summary[f'best_f1_fold{self.fold_i}']= best_f1
            wandb.run.summary[f'best_epoch_fold{self.fold_i}'] = best_epoch
        else:
            logger.warning('No validation metrics logged.')
    
    def forward(self, inputs, return_attn=False, return_cls_token=False):
        return self.model(inputs, return_attn=return_attn, return_cls_token=return_cls_token)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        logits = self(x)
        y_hat = torch.argmax(logits, dim=1)
        # compute loss
        loss = self.loss_fn(logits, y)
        # accumulate accuracy
        self.train_acc_fn.update(y_hat, y)
        self.log_dict({f'loss_fold{self.fold_i}': loss,
                       f'lr_fold{self.fold_i}': self.optimizers().param_groups[0]['lr']}, 
                      prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        logits = self(x)
        y_hat = torch.argmax(logits, dim=1)
        # compute loss
        loss = self.loss_fn(logits, y)
        # accumulate accuracy, f1, recall
        self.val_acc_fn.update(y_hat, y)
        self.recall_fn.update(y_hat, y)
        self.f1_fn.update(y_hat, y)
        self.correct.append(y_hat.eq(y).sum().item())
        self.log(f'val_loss_fold{self.fold_i}', loss, prog_bar=True)
                
    def configure_optimizers(self):
        
        if self.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), 
                                        lr=self.learning_rate, 
                                        betas=self.betas)
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), 
                                        lr=self.learning_rate, 
                                        # weight_decay=self.weight_decay,
                                        momentum=0.9)
        elif self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), 
                                        lr=self.learning_rate, 
                                        # weight_decay=self.weight_decay,
                                        betas=self.betas)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs),
                'interval': 'epoch',
                'frequency': 1
            }
        }