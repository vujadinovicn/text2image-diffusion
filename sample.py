import torch
from loss.losses import get_constants, get_useful_values
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils import parse_config, load_pretrained_model
import argparse

def mean_predictor_step(i, T, x, model, diffusion_params, generated_images):
    t_current = torch.tensor([i], device=x.device)
    _, _, _, sigma_current = get_useful_values(t_current, **diffusion_params)
    
    mu_theta = model(x, t_current)

    if i > 0:
        noise = torch.randn_like(x).to(x.device)
        x_new = mu_theta + sigma_current.squeeze() * noise
    else:
        x_new = mu_theta
    
    if i % 100 == 0 or i == T-1:
        to_append = x_new.detach().clone()
        generated_images.append(to_append)
    
    x = x_new.clamp(-1, 1)
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
    
    x = x_new.clamp(-1, 1)
    return x


def sample(config, method):
    diffusion_params = config['diffusion_params']
    T = diffusion_params['T'] # TODO: check

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_pretrained_model(config).to(device)
    model.eval()

    generated_images = []
    x = torch.randn(1, 1, 32, 32).to(device)

    if method == 'noise_predictor':
        alpha_t, alpha_bar_t, _, sigma_square_t, log_sigma_square_t_clipped = get_constants(device, **diffusion_params)

    with torch.no_grad():
        for i in tqdm(reversed(range(T)), total=T):
            if method == 'mean_predictor':
                x = mean_predictor_step(i, T, x, model, diffusion_params, generated_images)
            elif method == 'noise_predictor':
                x = noise_predictor_step(i, T, x, model, alpha_t, alpha_bar_t, sigma_square_t, generated_images)
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
    argparse.add_argument('--sample_method', type=str, default='mean_predictor', choices=["noise_predictor", "mean_predictor"], help="Sampling method.")
    args = argparse.parse_args()

    config = parse_config(args.config_path)
    final_image, generated_images = sample(config=config, method=args.sample_method)
    plot_generated_images(final_image, generated_images)