import torch

# TODO: implement get_useful_values
def get_useful_values(**args):
    pass

def get_constants(device, beta_1= 0.001, beta_T= 0.02, T=1000):
    """
    Compute alpha_t, alpha_bar_t and sigma_t for given time step t.
    """
    beta_t = torch.linspace(beta_1, beta_T, T, device=device)
    alpha_t = 1.0 - beta_t
    alpha_bar_t = torch.cumprod(alpha_t, dim=0)
    alpha_bar_t_minus_1 = torch.cat([torch.ones(1, device=device), alpha_bar_t[:-1]])
    sigma_square_t = (1.0 - alpha_bar_t_minus_1)*(1 - alpha_t)/(1.0 - alpha_bar_t)
     # build clipped posterior variance like OpenAI; since variance can be 0 for t=0, and log(0) is undefined
    sigma_square_t_clipped = torch.cat([sigma_square_t[1].unsqueeze(0), sigma_square_t[1:]],dim=0)
    log_sigma_square_t_clipped = torch.log(sigma_square_t_clipped.clamp(min=1e-20))

    return alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t, log_sigma_square_t_clipped

def noise_predictor_loss(estimated_noise,
                         true_noise):
    loss = ((estimated_noise - true_noise)**2)
    loss = loss.view(loss.shape[0], -1).sum(dim=-1).mean()
    return loss, 0.0, 0.0 

def variational_lower_bound_loss(mu_theta,
                                 original_x,
                                 noisy_x,
                                 batch_t,
                                 alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t):
    
    #mu theta: B x C x H x W
    #original_x: B x C x H x W
    #noisy_x: B x C x H x W
    #batch_t: B
    
    alpha_t = alpha_t[batch_t].view(-1, 1, 1, 1)
    alpha_bar_t = alpha_bar_t[batch_t].view(-1, 1, 1, 1)
    alpha_bar_t_minus_1 = alpha_bar_t_minus_1[batch_t].view(-1, 1, 1, 1)
    sigma_square_t = sigma_square_t[batch_t].view(-1,1,1,1)   
    
    # compute mu_q
    mu_q = torch.sqrt(alpha_t)*(1.0 - alpha_bar_t_minus_1)*noisy_x
    mu_q += torch.sqrt(alpha_bar_t_minus_1)*(1 - alpha_t)*original_x
    mu_q = mu_q/(1 - alpha_bar_t)
    
    t_0_indices = (batch_t==0)    
    t_non0_indices = (batch_t!=0)

    # denoising matching term for t > 
    if not t_non0_indices.any():
        loss_non0 = torch.tensor(0.0).to(mu_theta.device)
    else:
        mu_theta_non0 = mu_theta[t_non0_indices]
        sigma_square_t_non0 = sigma_square_t[t_non0_indices]
        mu_q_non0 = mu_q[t_non0_indices]
        loss_non0 = ((mu_theta_non0 - mu_q_non0)**2)/(2*(sigma_square_t_non0 + 1e-12))
        loss_non0 = loss_non0.view(loss_non0.shape[0],-1).sum(dim=-1).mean()

    # Reconstruction term for t = 0
    if not t_0_indices.any():
        loss_0 = torch.tensor(0.0).to(mu_theta.device)
    else:
        mu_theta_0 = mu_theta[t_0_indices]
        original_x_0 = original_x[t_0_indices]
        alpha_bar_t_0 = alpha_bar_t[t_0_indices]
        loss_0 = ((mu_theta_0 - original_x_0)**2)
        loss_0 /= (1 - alpha_bar_t_0)
        loss_0 = loss_0.view(loss_0.shape[0],-1).sum(dim=-1).mean()

    return loss_non0 + loss_0, loss_non0.item(), loss_0.item()


def variational_lower_bound_loss2(mu_theta,
                                 original_x,
                                 noisy_x,
                                 batch_t,
                                 alpha_t, alpha_bar_t, alpha_bar_t_minus_1,
                                 log_sigma_square_t_clipped, log_sigma_square):
    
    #mu theta: B x C x H x W
    #original_x: B x C x H x W
    #noisy_x: B x C x H x W
    #batch_t: B
    
    alpha_t = alpha_t[batch_t].view(-1, 1, 1, 1)
    alpha_bar_t = alpha_bar_t[batch_t].view(-1, 1, 1, 1)
    alpha_bar_t_minus_1 = alpha_bar_t_minus_1[batch_t].view(-1, 1, 1, 1)
    log_sigma_square_t_clipped = log_sigma_square_t_clipped[batch_t].view(-1, 1, 1, 1)
    
    # compute mu_q
    mu_q = torch.sqrt(alpha_t)*(1.0 - alpha_bar_t_minus_1)*noisy_x
    mu_q += torch.sqrt(alpha_bar_t_minus_1)*(1 - alpha_t)*original_x
    mu_q = mu_q/(1 - alpha_bar_t)
    
    t_0_indices = (batch_t==0)    
    t_non0_indices = (batch_t!=0)

    # end
    import numpy as np

    kl = normal_kl(mean1=mu_q, logvar1=log_sigma_square_t_clipped, mean2=mu_theta, logvar2=log_sigma_square)
    kl = kl.mean(dim=list(range(1, len(kl.shape)))) / np.log(2.0)

    import math
    decoder_log_var = 0.0
    decoder_var = 1.0
    decoder_nll = 0.5 * (((original_x - mu_theta) ** 2) / decoder_var + decoder_log_var + math.log(2 * math.pi))
    decoder_nll = decoder_nll.mean(dim=list(range(1, len(decoder_nll.shape)))) / np.log(2.0)

    loss = torch.where(t_0_indices, decoder_nll, kl).mean()

    if t_non0_indices.any():
        loss_non0 = kl[t_non0_indices].mean()
    else:
        loss_non0 = torch.tensor(0.0, device=mu_theta.device)

    if t_0_indices.any():
        loss_0 = decoder_nll[t_0_indices].mean()
    else:
        loss_0 = torch.tensor(0.0, device=mu_theta.device)

    return loss, loss_non0.item(), loss_0.item()

def normal_kl(mean1, logvar1, mean2, logvar2):
    var1, var2 = torch.exp(logvar1), torch.exp(logvar2)
    return 0.5 * (logvar2 - logvar1 + (var1 + (mean1 - mean2) ** 2) / var2 - 1.0)


def compute_log_sigma_square(var_theta, t_batch, log_sigma_square_t_clipped, alpha_t, use_single_batch=True):
    batch_size = 1 if use_single_batch else t_batch.shape[0]

    min_log = log_sigma_square_t_clipped[t_batch].view(batch_size, 1, 1, 1)
    max_log = torch.log(1.0 - alpha_t[t_batch].clamp(min=1e-20)).view(batch_size, 1, 1, 1)

    frac = (var_theta.clamp(-1.0, 1.0) + 1.0) / 2.0
    interpolated = frac * max_log + (1.0 - frac) * min_log
    return interpolated