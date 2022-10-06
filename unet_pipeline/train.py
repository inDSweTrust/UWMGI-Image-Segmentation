import numpy as np
import gc

import torch
import torch.nn as nn
from torch.cuda import amp
import torch.optim as optim
import segmentation_models_pytorch as smp

from tqdm import tqdm
from pathlib import Path

import time
import copy
import gc
import wandb
from tqdm import tqdm
from collections import defaultdict

# For colored terminal text
from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

from cfg import CFG
from loss import criterion, dice_coef, iou_coef
from data import prepare_loaders
from scheduler import fetch_scheduler

class Training(nn.Module):

  def __init__(self, 
              optimizer,
              scheduler,
              device, 
              num_epochs
    ):

    self.optimizer = optimizer
    self.scheduler = scheduler
    self.device = device
    self.num_epochs = num_epochs


  def train_one_epoch(self, model, dataloader):

    model.train()
    scaler = amp.GradScaler()   # what is this?
    
    dataset_size = 0
    running_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, masks) in pbar:         
        images = images.to(self.device, dtype=torch.float)
        masks  = masks.to(self.device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        with amp.autocast(enabled=True):    #what is that?
            y_pred = model(images)
            loss   = criterion(y_pred, masks)
            loss   = loss / CFG['n_accumulate']
            
        scaler.scale(loss).backward()
    
        if (step + 1) % CFG['n_accumulate'] == 0:
            scaler.step(self.optimizer)
            scaler.update()

            # zero the parameter gradients
            self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = self.optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_mem=f'{mem:0.2f} GB')
        torch.cuda.empty_cache()
        gc.collect()
    
    return epoch_loss
    
  @torch.no_grad()
  def valid_one_epoch(self, model, dataloader):
      model.eval()
      
      dataset_size = 0
      running_loss = 0.0
      
      val_scores = []
      
      pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
      for step, (images, masks) in pbar:        
          images = images.to(self.device, dtype=torch.float)
          masks = masks.to(self.device, dtype=torch.float)
          
          batch_size = images.size(0)
          
          y_pred = model(images)
          loss = criterion(y_pred, masks)
          
          running_loss += (loss.item() * batch_size)
          dataset_size += batch_size
          
          epoch_loss = running_loss / dataset_size
          
          y_pred = nn.Sigmoid()(y_pred)
          val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
          val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
          val_scores.append([val_dice, val_jaccard])
          
          mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
          current_lr = self.optimizer.param_groups[0]['lr']
          pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                          lr=f'{current_lr:0.5f}',
                          gpu_memory=f'{mem:0.2f} GB')

      val_scores  = np.mean(val_scores, axis=0)
      torch.cuda.empty_cache()
      gc.collect()
      
      return epoch_loss, val_scores

  def model_checkpoint(self, 
                     model, 
                     epoch=0,
                     best_dice=0,
                     best_epoch=0,
                     best_model=0
                     ):    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_dice': best_dice, 
            'best_epoch': best_epoch,
            'best_model': best_model
            }, CFG['PATH_weights'] + f'model-checkpoint.pt')

  def run_training(self, model, train_loader, valid_loader, fold):
    # To automatically log gradients
    # wandb.watch(self.model, log_freq=100)
    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    # train_loader, valid_loader = prepare_loaders(debug=CFG['debug'])
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = -np.inf
    best_epoch = -1
    last_epoch = 0
    history = defaultdict(list)

    # if self.checkpoint:
    #   cp = torch.load(CFG['PATH_weights'] + f'model-checkpoint.pt')
    #   model.load_state_dict(cp['model_state_dict'])
    #   self.optimizer.load_state_dict(cp['optimizer_state_dict'])
    #   best_model_wts = cp['best_model']
    #   best_dice = cp['best_dice']
    #   last_epoch = cp['epoch']

    
    for epoch in range(last_epoch + 1, self.num_epochs + 1): 
        gc.collect()
        print(f'Epoch {epoch}/{self.num_epochs}', end='')

        train_loss = self.train_one_epoch(model,
                                          dataloader=train_loader
                                          )
        
        val_loss, val_scores = self.valid_one_epoch(model, valid_loader, 
                                                 device=CFG['device'], 
                                                 epoch=epoch)
        val_dice, val_jaccard = val_scores
    
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)
        
        # Log the metrics
        wandb.log({"Train Loss": train_loss, 
                   "Valid Loss": val_loss,
                   "Valid Dice": val_dice,
                   "Valid Jaccard": val_jaccard,
                   "LR": self.scheduler.get_last_lr()[0]})
        
        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')
        
        # deep copy the model
        if val_dice >= best_dice:
            print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice = val_dice
            best_jaccard = val_jaccard
            best_epoch = epoch
            run.summary["Best Dice"]    = best_dice
            run.summary["Best Jaccard"] = best_jaccard
            run.summary["Best Epoch"]   = best_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            
            torch.save(model.state_dict(), CFG['PATH_weights'] + f"best_epoch-{fold:02d}.bin")
            # Save a model file from the current directory
            wandb.save(CFG['PATH_weights'])
            print(f"Model Saved{sr_}")
            
        last_model_wts = copy.deepcopy(model.state_dict())

        torch.save(model.state_dict(), CFG['PATH_weights'] + f"last_epoch-{epoch:02d}.bin")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_dice': best_dice
            }, Path(CFG['CHECKPOINTS_FOLDER'], f'model-checkpoint.pt'))

        # self.model_checkpoint(model, 
        #                       self.optimizer,
        #                       self.epoch,
        #                       best_dice=best_dice,
        #                       best_epoch=best_epoch,
        #                       best_model=best_model_wts
        #                       )
        # print(); print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Score: {:.4f}".format(best_jaccard))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history


  def load_model(self, path):
    model = self.build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

    