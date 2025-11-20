import torch
from loss.losses import get_constants
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils import parse_config, load_pretrained_model
import argparse

def mean_predictor_step(i, T, x, model, generated_images, sigma_current):
    t_current = torch.tensor([i], device=x.device)    
    mu_theta = model(x, t_current)
    sigma_current = sigma_current[t_current]
    if i > 0:
        noise = torch.randn_like(x).to(x.device)
        x_new = mu_theta + torch.sqrt(sigma_current) * noise
    else:
        x_new = mu_theta
    
    if i % 100 == 0 or i == T-1:
        to_append = x_new.detach().clone()
        generated_images.append(to_append)
    
    x = x_new
    return x

def noise_predictor_step(i, T, x, model, 
                         alpha_t, alpha_bar_t, sigma_square_t, generated_images):
    t_current = torch.tensor([i], device=x.device)
        
    sigma_current = sigma_square_t[t_current]
    alpha_t_current = alpha_t[t_current]
    alpha_bar_t_current = alpha_bar_t[t_current]

    eps_theta = model(x, t_current)

    if i > 0:
        noise = torch.randn_like(x).to(x.device)
        x_new = (x  - (1 - alpha_t_current)/torch.sqrt(1 - alpha_bar_t_current) * eps_theta)/torch.sqrt(alpha_t_current)
        x_new += torch.sqrt(sigma_current) * noise
    else:
        x_new = (x  - (1 - alpha_t_current)/torch.sqrt(1 - alpha_bar_t_current) * eps_theta)/torch.sqrt(alpha_t_current)        
    
    if i % 100 == 0 or i == T-1:
        to_append = x_new.detach().clone()
        generated_images.append(to_append)
    
    x = x_new
    return x

def denoising_step(i, T, x, model, generated_images,
                   alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t):
    t_current = torch.tensor([i], device=x.device)
    
    alpha_t_current = alpha_t[t_current]
    alpha_bar_t_current = alpha_bar_t[t_current]
    sigma_square_current = sigma_square_t[t_current]
    alpha_bar_t_minus_1_current = alpha_bar_t_minus_1[t_current]

    x0_pred = model(x, t_current)
    m1 = (1 - alpha_bar_t_minus_1_current)*torch.sqrt(alpha_t_current)*x
    m2 = (1 - alpha_t_current)*torch.sqrt(alpha_bar_t_minus_1_current)*x0_pred
    mu_theta = (m1+m2)/(1 - alpha_bar_t_current)
    if i > 0:
        noise = torch.randn_like(x).to(x.device)
        x_new = mu_theta + torch.sqrt(sigma_square_current)*noise
    else:
        x_new = mu_theta
    
    if i % 100 == 0 or i == T-1:
        to_append = x_new.detach().clone()
        generated_images.append(to_append)
    
    return x_new

def conditional_score_matching_step(i, T, x, model, diffusion_params, generated_images,
                        alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t, y, w=1.0):
    
    t_current = torch.tensor([i], device=x.device)

    alpha_t_current = alpha_t[t_current]
    alpha_bar_t_current = alpha_bar_t[t_current]
    sigma_square_current = sigma_square_t[t_current]
    alpha_bar_t_minus_1_current = alpha_bar_t_minus_1[t_current]
    b = x.shape[0]

    y = y.repeat(b).to(x.device) # SHAPE: (b,)
    score_theta = model(x, t_current, y)

    un_y = torch.tensor([3], device=x.device)
    un_y = un_y.repeat(b)  # SHAPE: (b,)

    uncond_score_theta = model(x, t_current, un_y)  

    score_theta = (w+1)*score_theta - w*uncond_score_theta
    
    mu_q = x + (1 - alpha_t_current) * score_theta 
    mu_q = mu_q / torch.sqrt(alpha_t_current)
    if i > 0:
        noise = torch.randn_like(x).to(x.device)
        x_new = mu_q + torch.sqrt(sigma_square_current)*noise
    else:
        x_new = mu_q
    if i % 100 == 0 or i == T-1:
        to_append = x_new.detach().clone()
        generated_images.append(to_append)
    return x_new

def score_matching_step(i, T, x, model, diffusion_params, generated_images,
                        alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t):
    t_current = torch.tensor([i], device=x.device)
    
    alpha_t_current = alpha_t[t_current]
    alpha_bar_t_current = alpha_bar_t[t_current]
    sigma_square_current = sigma_square_t[t_current]
    alpha_bar_t_minus_1_current = alpha_bar_t_minus_1[t_current]

    score_theta = model(x, t_current)

    mu_q = x + (1 - alpha_t_current) * score_theta 
    mu_q = mu_q / torch.sqrt(alpha_t_current)
    if i > 0:
        noise = torch.randn_like(x).to(x.device)
        x_new = mu_q + torch.sqrt(sigma_square_current)*noise
    else:
        x_new = mu_q
    if i % 100 == 0 or i == T-1:
        to_append = x_new.detach().clone()
        generated_images.append(to_append)
    return x_new

def sample(config, method):
    diffusion_params = config['diffusion_params']
    T = diffusion_params['T'] # TODO: check

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_pretrained_model(config).to(device)
    model.eval()

    generated_images = []
    x = torch.randn(1, 1, 32, 32).to(device)

    
    alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t = get_constants(device, **diffusion_params)

    with torch.no_grad():
        for i in tqdm(reversed(range(T)), total=T):
            if method == 'mean_predictor': # this does not work well
                x = mean_predictor_step(i, T, x, model, generated_images, sigma_square_t)
            elif method == 'noise_predictor':
                x = noise_predictor_step(i, T, x, model, alpha_t, alpha_bar_t, sigma_square_t, generated_images)
            elif method == 'denoising':
                x = denoising_step(i, T, x, model, diffusion_params, generated_images, alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t)
            elif method == 'score_matching':
                x = score_matching_step(i, T, x, model, diffusion_params, generated_images, alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t)
            elif method == 'conditional_score_matching':
                y = torch.tensor([2], device=device)  # generate class '2'
                x = conditional_score_matching_step(i, T, x, model, diffusion_params, generated_images, alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t, y, w=1.0)
            else:
                raise ValueError(f"Unknown sampling method: {method}")
            
    return x, generated_images

def plot_generated_images(final_image, generated_images):
    final_image = final_image.squeeze().cpu().numpy()
    final_image = (final_image + 1) / 2  
    plt.imshow(final_image, cmap='gray')

    # plot_images = generated_images[-10:]
    # take equally spaced 10 images from generated_images
    plot_images = []
    num_images = len(generated_images)
    indices = torch.linspace(0, num_images - 1, steps=10).long()
    for idx in indices:
        plot_images.append(generated_images[idx])

    fig, axes = plt.subplots(1, len(plot_images), figsize=(15, 3))
    for ax, img in zip(axes, plot_images):
        img = img.squeeze().cpu().numpy()
        img = (img + 1) / 2  # Rescale to [0, 1]
        # img = img.clip(0, 1)
        ax.imshow(img)
        ax.axis('off')
    plt.show()


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--config_path', type=str, default='config/mnist.yml', help='Path to the configuration file.')
    argparse.add_argument('--sample_method', type=str, default='mean_predictor', choices=["noise_predictor", "mean_predictor", "denoising", "score_matching","conditional_score_matching"], help="Sampling method.")
    args = argparse.parse_args()

    config = parse_config(args.config_path)
    final_image, generated_images = sample(config=config, method=args.sample_method)
    plot_generated_images(final_image, generated_images)