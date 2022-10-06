#UMWGI config
from os import truncate
import torch 

CFG = {
    'seed': 101,
    'debug': True,
    'checkpoint': True, # load model checkpoint
    'exp_name': '2.5D',
    'comment': 'unet-efficientnet-b5',
    'model_name': 'Unet',
    'backbone': 'efficientnet-b5',
    'train_bs': 16,
    'valid_bs': 32,
    'img_size': [320, 320],
    'epochs': 5,
    'lr': 2e-3,
    'scheduler': 'CosineAnnealingLR',
    'optimizer': 'Adam',
    'min_lr': 1e-6,
    'T_max': 0,
    'T_0': 25,
    'warmup_epochs': 0,
    'wd': 1e-6,
    'n_accumulate': 0,
    'n_fold': 5,
    'folds': [0],
    'num_classes': 3,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'DATA_DIR': f"/content",
    'ROOT_DIR': f"/content/uwmgi_repo",
    'WEIGHTS_FOLDER': f"weights",
    'CHECKPOINTS_FOLDER': f"checkpoints"
}

CFG['T_max'] = int(30000/CFG['train_bs']*CFG['epochs']) + 50
CFG['n_accumulate'] = max(1, 32//CFG['train_bs'])