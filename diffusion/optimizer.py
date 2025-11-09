from torch.optim import lr_scheduler

def get_scheduler(optimizer, config):
    if config.lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + config.epoch_count - config.n_epochs) / float(config.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif config.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iters, gamma=0.1)
    elif config.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif config.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs, eta_min=0)
    else:
        raise NotImplementedError()
    
    return scheduler