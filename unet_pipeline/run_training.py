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

from cfg import CFG
from train import Training
from data import prepare_loaders
from scheduler import fetch_scheduler
from train import Training


def build_model():
  model = smp.Unet(
      encoder_name=CFG["backbone"],      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
      encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
      in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
      classes=CFG['num_classes'],        # model output channels (number of classes in your dataset)
      activation=None,
  )
  model.to(CFG['device'])
  return model


def train_fold(fold, train_loader, valid_loader):
  print(f'#'*15)
  print(f'### Fold: {fold}')
  print(f'#'*15)
  # run = wandb.init(project='uwmgi-final', 
  #                 config={k:v for k, v in dict(vars(CFG)).items() if '__' not in k},
  #                 anonymous=anonymous,
  #                 name=f"fold-{fold}|dim-{CFG['img_size'][0]}x{CFG['img_size'][1]}|model-{CFG['model_name']}",
  #                 group=CFG['comment'],
  #                 )
  checkpoint_folder = Path(CFG["DATA_DIR"], CFG["CHECKPOINTS_FOLDER"])
  checkpoint_folder.mkdir(exist_ok=True, parents=True)

  model = build_model()
  optimizer = optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
  scheduler = fetch_scheduler(optimizer)
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
                            num_epochs
                    ).run_training(model, train_loader, valid_loader, fold)
  # run.finish()
  # display(ipd.IFrame(run.url, width=1000, height=720))

def main() -> None:
  #argparser here for path to config 
  config_dict = open("/content/uwmgi_repo/unet_pipeline/configs/config.yaml", 'r')
  CFG = yaml.load(config_dict, yaml.Loader)

  input_path = CFG['ROOT_DIR'] + '/input'
  data_path = CFG['DATA_DIR']

  CFG['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  for fold in CFG['folds']:
    train_loader, valid_loader = prepare_loaders(input_path, data_path, debug=CFG['debug'])
    train_fold(fold, train_loader, valid_loader)

if __name__=="__main__":
  main()
  

