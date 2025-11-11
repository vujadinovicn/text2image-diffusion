import torch

def get_useful_values(t_batch, beta_1=10e-4, beta_T=0.02, T=1000):
    # t = {0, 1, ..., T-1}
    device = t_batch.device
    beta_schedule = torch.linspace(beta_1, beta_T, T).to(device)  # shape: [T]
    beta_batch = beta_schedule[t_batch].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # shape: [B, 1, 1, 1]
    alpha_batch = 1 - beta_batch
    alpha_bar_batch = torch.cumprod((1-beta_schedule), dim=0)[t_batch].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # shape: [B, 1, 1, 1]
    sigma_batch = torch.sqrt(beta_batch)
    return beta_batch, alpha_batch, alpha_bar_batch, sigma_batch

def mean_predictor_loss(mu_theta, t_batch, x_0, x_t, beta_batch, alpha_t_batch, alpha_bar_batch, sigma_batch):

    alpha_bar_t_minus_1_batch = alpha_bar_batch / alpha_t_batch

    mu_q = torch.sqrt(alpha_t_batch)*(1 - alpha_bar_t_minus_1_batch)*x_t    
    mu_q += torch.sqrt(alpha_bar_t_minus_1_batch)*(1-alpha_t_batch)*x_0
    mu_q /= (1 - alpha_bar_batch)

    t_not_0_indices = (t_batch != 0)
    if t_not_0_indices.sum() > 0:
        matching_loss = torch.mean((1/(2*(sigma_batch[t_not_0_indices]**2)))*(mu_theta[t_not_0_indices].view(-1) - mu_q[t_not_0_indices].view(-1))**2)
    else:
        matching_loss = torch.tensor(0.0, device=x_0.device)
    
    t_1_indices = (t_batch == 0)
    if t_1_indices.sum() > 0:
        x_0_t1 = x_0[t_1_indices]
        mu_theta_t1 = mu_theta[t_1_indices]
        reconstruction_loss = torch.mean((x_0_t1.view(-1) - mu_theta_t1.view(-1))**2)
        total_loss = matching_loss + reconstruction_loss
    else:
        total_loss = matching_loss
    
    return total_loss

    