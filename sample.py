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
                        alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t, y, w=-1.0):
    
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

def sample(config, method, batch_size=8):
    diffusion_params = config['diffusion_params']
    T = diffusion_params['T'] # TODO: check
    B = batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_pretrained_model(config).to(device)
    model.eval()

    generated_images = []
    x = torch.randn(B, 1, 32, 32).to(device)
    
    alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t = get_constants(device, **diffusion_params)

    with torch.no_grad():
        for i in tqdm(reversed(range(T)), total=T):
            if method == 'mean_predictor': # this does not work well
                x = mean_predictor_step(i, T, x, model, generated_images, sigma_square_t)
            elif method == 'noise_predictor':
                x = noise_predictor_step(i, T, x, model, alpha_t, alpha_bar_t, sigma_square_t, generated_images)
            elif method == 'denoising':
                x = denoising_step(i, T, x, model, generated_images, alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t)
                # i, T, x, model, generated_images,
                #    alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t
            elif method == 'score_matching':
                x = score_matching_step(i, T, x, model, diffusion_params, generated_images, alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t)
            elif method == 'conditional_score_matching':
                y = torch.tensor([0], device=device)  # generate class '2'
                x = conditional_score_matching_step(i, T, x, model, diffusion_params, generated_images, alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t, y, w=-1.0)
            else:
                raise ValueError(f"Unknown sampling method: {method}")
            
    return x, generated_images

def plot_generated_images(final_image, generated_images, n_show=8):
    """
    Show up to `n_show` images from the batch in final_image.
    final_image: tensor of shape (B, 1, H, W)
    """
    final_image = final_image.detach().cpu()
    B = final_image.shape[0]
    n_show = min(n_show, B)

    # select first n_show images and remove channel dim
    imgs = final_image[:n_show].squeeze(1).numpy()  # shape: (n_show, H, W)
    imgs = (imgs + 1.0) / 2.0  # rescale to [0,1]

    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 2, 2))
    if n_show == 1:
        axes = [axes]
    for ax, img in zip(axes, imgs):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--config_path', type=str, default='config/mnist.yml', help='Path to the configuration file.')
    argparse.add_argument('--sample_method', type=str, default='mean_predictor', choices=["noise_predictor", "mean_predictor", "denoising", "score_matching","conditional_score_matching"], help="Sampling method.")
    argparse.add_argument('--save_folder', type=str, default='../checkpoints/images_gen', help='Path to the configuration file.')
    argparse.add_argument('--show_images', action='store_true', help='If set, display images; otherwise save them to --save_folder.')
    argparse.add_argument('--batch_size', type=int, default=8, help='Batch size for sampling.')
    args = argparse.parse_args()

    config = parse_config(args.config_path)
    final_image, generated_images = sample(config=config, method=args.sample_method, batch_size=args.batch_size)
    if args.show_images:
        plot_generated_images(final_image, generated_images)
    else:
        import os
        os.makedirs(args.save_folder, exist_ok=True)
        # save final image
        final = final_image.squeeze().cpu().numpy()
        final = (final + 1) / 2
        for i in range(args.batch_size):
            plt.imsave(os.path.join(args.save_folder, f'final_image_{i}.png'), final[i], cmap='gray')
        