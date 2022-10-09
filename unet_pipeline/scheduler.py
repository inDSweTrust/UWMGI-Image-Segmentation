from torch.optim import lr_scheduler


def fetch_scheduler(CFG, optimizer):
    if CFG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG['T_max'], 
                                                   eta_min=CFG['min_lr'])
    elif CFG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CFG['T_0'], 
                                                             eta_min=CFG['min_lr'])
    elif CFG['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=CFG['min_lr'],)
    elif CFG['scheduler'] == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif CFG['scheduler'] == None:
        return None
        
    return scheduler