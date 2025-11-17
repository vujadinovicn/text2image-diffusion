import torch
import math

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

def compute_log_sigma_square(var_theta, t_batch, log_sigma_square_t_clipped, alpha_t, use_single_batch=True):
    batch_size = 1 if use_single_batch else t_batch.shape[0]

    min_log = log_sigma_square_t_clipped[t_batch].view(batch_size, 1, 1, 1)
    max_log = torch.log(1.0 - alpha_t[t_batch].clamp(min=1e-20)).view(batch_size, 1, 1, 1)

    frac = (var_theta.clamp(-1.0, 1.0) + 1.0) / 2.0
    interpolated = frac * max_log + (1.0 - frac) * min_log
    return interpolated

def noise_predictor_loss(estimated_noise,
                         true_noise):
    loss = ((estimated_noise - true_noise)**2)
    loss = loss.view(loss.shape[0], -1).sum(dim=-1).mean()
    return loss, 0.0, 0.0 

# maybe try the version of this loss which matches exactly the denoising loss
# this is equivalent to multiplying the loss by (2_sigma_t^2)*(1 - alpha_bar_t)*alpha_t/(1 - alpha_t)^2
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

def score_matching_loss(estimated_score,
                         original_x,
                         noisy_x,
                         alpha_bar_t):
    true_score = -(noisy_x - original_x*torch.sqrt(alpha_bar_t))
    true_score /= (1 - alpha_bar_t + 1e-12)
    loss = ((estimated_score - true_score)**2)
    loss = loss * (1 - alpha_bar_t)
    loss = loss.view(loss.shape[0], -1).sum(dim=-1).mean()
    return loss, 0.0, 0.0

def denoising_loss(estimated_x0,
                   original_x):
    loss = ((estimated_x0 - original_x)**2)
    loss = loss.view(loss.shape[0], -1).sum(dim=-1).mean()
    return loss, 0.0, 0.0

# denoising loss which is equivalent to the noise predictor loss
# def denoising_loss(estimated_x0,
    #                original_x,
    #                  alpha_bar_t,
    #                 alpha_t):
    # loss = ((estimated_x0 - original_x)**2)
    # loss = loss * alpha_t/(1 - alpha_bar_t + 1e-12)
    # loss = loss.view(loss.shape[0], -1).sum(dim=-1).mean()
    # return loss, 0.0, 0.0


def transform_timestep_data(batch_t, *data):
    return [d[batch_t].view(-1, 1, 1, 1) for d in data]

def compute_mu_q(original_x, noisy_x, alpha_t, alpha_bar_t, alpha_bar_t_minus_1):
    mu_q = torch.sqrt(alpha_t) * (1.0 - alpha_bar_t_minus_1) * noisy_x
    mu_q += torch.sqrt(alpha_bar_t_minus_1) * (1.0 - alpha_t) * original_x
    mu_q = mu_q / (1.0 - alpha_bar_t)
    return mu_q, alpha_bar_t

def variational_lower_bound_loss(mu_theta,
                                 original_x,
                                 noisy_x,
                                 batch_t,
                                 alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t):
    
    #mu theta: B x C x H x W
    #original_x: B x C x H x W
    #noisy_x: B x C x H x W
    #batch_t: B

    alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t = transform_timestep_data(batch_t, alpha_t, 
                                                                                        alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t)  
    
    mu_q, alpha_bar_t = compute_mu_q(original_x, noisy_x, alpha_t, alpha_bar_t, alpha_bar_t_minus_1)
    
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

def compute_normal_kl(mean1, logvar1, mean2, logvar2):
    var1, var2 = torch.exp(logvar1), torch.exp(logvar2)
    return 0.5 * (logvar2 - logvar1 + (var1 + (mean1 - mean2) ** 2) / var2 - 1.0)

def compute_nll(x, mean):
    return 0.5 * ((x - mean) ** 2 + math.log(2.0))
    
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
def vlb_openai_like(mu_theta, original_x, noisy_x, batch_t,
                    alpha_t, alpha_bar_t, alpha_bar_t_minus_1,
                    log_sigma_square_t_clipped, log_sigma_square):
    
    alpha_t, alpha_bar_t, alpha_bar_t_minus_1, log_sigma_square_t_clipped = transform_timestep_data(batch_t, alpha_t, 
                                                                                        alpha_bar_t, alpha_bar_t_minus_1, log_sigma_square_t_clipped)  
    
    mu_q, alpha_bar_t = compute_mu_q(original_x, noisy_x, alpha_t, alpha_bar_t, alpha_bar_t_minus_1)

    t_0_indices = (batch_t==0)    
    t_non0_indices = (batch_t!=0)

    kl = compute_normal_kl(mean1=mu_q, logvar1=log_sigma_square_t_clipped, mean2=mu_theta, logvar2=log_sigma_square)
    kl = kl.mean(dim=list(range(1, len(kl.shape)))) / math.log(2.0)

    nll = compute_nll(x=original_x, mean=mu_theta)
    nll = nll.mean(dim=list(range(1, nll.dim()))) / math.log(2.0)

    loss = torch.where(t_0_indices, nll, kl).mean()
    loss_non0 = kl[t_non0_indices].mean() if t_non0_indices.any() else torch.tensor(0.0, device=mu_theta.device)
    loss_0 = nll[t_0_indices].mean() if t_0_indices.any() else torch.tensor(0.0, device=mu_theta.device)

    return loss, loss_non0.item(), loss_0.item()