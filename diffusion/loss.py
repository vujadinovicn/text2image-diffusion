import torch.nn.functional as F

def get_loss(loss_type):
    if loss_type == 'l1':
          loss = F.l1_loss
    elif loss_type == 'l2':
        loss = F.mse_loss
    else:
        raise NotImplementedError()

    return loss