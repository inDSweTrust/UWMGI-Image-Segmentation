import argparse
import yaml

from pickle import NONE
import numpy as np
from timm.models.ghostnet import _cfg

import torch 
import torch.nn as nn
from torch.cuda import amp
import torch.optim as optim
import segmentation_models_pytorch as smp
from pathlib import Path
import yaml

import wandb

from train import Training
from data import prepare_loaders
from scheduler import fetch_scheduler
from train import Training
from utils import load_yaml


def argparser():
  parser = argparse.ArgumentParser(description='uwmgi segmentation pipeline')
  parser.add_argument('CFG_TRAIN', type=str, help='train config path')
  return parser.parse_args()

def build_model(CFG):
  model = smp.Unet(encoder_name=CFG["backbone"],   # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                   encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                   in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                   classes=CFG['num_classes'],     # model output channels (number of classes in your dataset)
                   activation=None,
  )
  model.to(CFG['device'])
  return model

def train_fold(CFG, fold, train_loader, valid_loader):
  print(f'#'*15)
  print(f'### Fold: {fold}')
  print(f'#'*15)

  checkpoint_folder = Path(CFG["DATA_DIR"], CFG["CHECKPOINTS_FOLDER"])
  checkpoint_folder.mkdir(exist_ok=True, parents=True)

  model = build_model(CFG)
  optimizer = optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
  scheduler = fetch_scheduler(CFG, optimizer)
  device = CFG["device"]
  num_epochs = CFG["epochs"]

  checkpoint_file = Path(CFG['CHECKPOINTS_FOLDER'], f'model-checkpoint.pt').is_file()

  if checkpoint_file:
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
  

  model, history = Training(optimizer,
                            scheduler,
                            device, 
                            num_epochs, 
                            CFG
                    ).run_training(model, train_loader, valid_loader, fold)

def main():
  args = argparser()
  config_folder = Path(args.CFG_TRAIN)
  CFG = load_yaml(config_folder)

  input_path = CFG['ROOT_DIR'] + '/input'
  data_path = CFG['DATA_DIR']

  CFG['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  for fold in CFG['folds']:
    train_loader, valid_loader = prepare_loaders(CFG, input_path, data_path, debug=CFG['debug'])
    train_fold(CFG, fold, train_loader, valid_loader)

if __name__=="__main__":
  main()
  

