import torch
import numpy as np
import segmentation_models_pytorch as smp


JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss    = smp.losses.DiceLoss(mode='multilabel')
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

# BCE + Tversky
# BCE + Dice

def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

def hd_dist(y_true, y_pred):
    preds_coords = np.argwhere(y_pred) / np.array(y_pred.shape)
    targets_coords = np.argwhere(y_true) / np.array(y_pred.shape)
    haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]
    return haussdorf_dist

def criterion(y_pred, y_true):
    return 0.5*DiceLoss(y_pred, y_true) + 0.5*JaccardLoss(y_pred, y_true)