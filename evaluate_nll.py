import torch
from data.mnist_dataloader import get_mnist_dataloader
from loss.losses import vlb_openai_like, prior_bpd, get_constants, compute_log_sigma_square
from tqdm import tqdm
from utils.utils import parse_config, load_pretrained_model, load_model
import argparse

def evaluate_nll(config):
    batch_size = config['eval']['batch_size']
    allowed_classes=config['train']['allowed_classes']
    diffusion_params = config['diffusion_params']
    T = diffusion_params['T']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = get_mnist_dataloader(batch_size=batch_size, split="test", allowed_classes=allowed_classes)

    model = load_pretrained_model(config).to(device)
    model.eval()

    all_bpd = []
    with torch.no_grad():
        for images, _ in tqdm(test_loader):
            images = images.to(device)
            batch_size = images.size(0)
            alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t, log_sigma_square_t_clipped = get_constants(device, **diffusion_params)

            vbs = []

            for t in reversed(range(T)):
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

                alpha_bar_t_batch = alpha_bar_t[t_batch].view(-1, 1, 1, 1)
                x_t = torch.sqrt(alpha_bar_t_batch)*images + torch.sqrt(1 - alpha_bar_t_batch)*torch.randn_like(images)

                with torch.no_grad():
                    if not config['model']['learned_variance']:
                        eps_theta = model(x_t, t_batch)
                        alpha_t_batch = alpha_t[t_batch].view(-1, 1, 1, 1)
                        sigma_square = sigma_square_t[t_batch].view(-1, 1, 1, 1)
                        log_sigma_square = torch.log(sigma_square.clamp(min=1e-20))
                    else:
                        eps_theta, var_theta = model(x_t, t_batch)
                        log_sigma_square = compute_log_sigma_square(var_theta, t_batch, log_sigma_square_t_clipped, alpha_t, use_single_batch=False)
                        alpha_t_batch = alpha_t[t_batch].view(-1, 1, 1, 1)
                        
                    mu_theta = (x_t - (1 - alpha_t_batch) / torch.sqrt(1 - alpha_bar_t_batch) * eps_theta) / torch.sqrt(alpha_t_batch)
                        
                    vb_t, _, _ = vlb_openai_like(
                        mu_theta=mu_theta,
                        original_x=images,
                        noisy_x=x_t,
                        batch_t=t_batch,
                        alpha_t=alpha_t,
                        alpha_bar_t=alpha_bar_t,
                        alpha_bar_t_minus_1=alpha_bar_t_minus_1,
                        log_sigma_square_t_clipped=log_sigma_square_t_clipped,
                        log_sigma_square=log_sigma_square,
                        fixed_var=not config['model']['learned_variance'],
                    )

                vbs.append(vb_t)
            vbs = torch.stack(vbs, dim=1)

            alpha_bar_T = alpha_bar_t[-1]
            prior = prior_bpd(images, alpha_bar_T)

            bpd = vbs.sum(dim=1) + prior
            all_bpd.append(bpd.cpu())

    all_bpd = torch.cat(all_bpd, dim=0)
    mean_bpd = all_bpd.mean().item()

    print(f"{mean_bpd:.4f} bits/dim")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--config_path', type=str, default='config/mnist.yml', help='Path to the configuration file.')
    args = argparse.parse_args()

    config = parse_config(args.config_path)
    evaluate_nll(config)