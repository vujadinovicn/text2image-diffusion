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
    return alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t

def noise_predictor_loss(estimated_noise,
                         true_noise):
    loss = ((estimated_noise - true_noise)**2)
    loss = loss.view(loss.shape[0], -1).sum(dim=-1).mean()
    return loss, 0.0, 0.0 

def mean_predictor_loss(mu_theta,
                        noisy_x,
                        original_x,
                        alpha_t,
                        alpha_bar_t_minus_1,
                        alpha_bar_t):
    
    mu_q = torch.sqrt(alpha_t)*(1.0 - alpha_bar_t_minus_1)*noisy_x
    mu_q += torch.sqrt(alpha_bar_t_minus_1)*(1 - alpha_t)*original_x
    mu_q = mu_q/(1 - alpha_bar_t)

    loss = ((mu_theta - mu_q)**2)
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