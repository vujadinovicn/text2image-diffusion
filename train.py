import torch
from data.mnist_dataloader import get_mnist_dataloader
from loss.losses import variational_lower_bound_loss, get_constants, noise_predictor_loss, mean_predictor_loss, denoising_loss, score_matching_loss, compute_log_sigma_square, vlb_openai_like, compute_mu_q
from tqdm import tqdm
from utils.utils import parse_config, load_model
import argparse

def train(config):
    batch_size = config['train']['batch_size']
    num_epochs = config['train']['epochs']
    learning_rate = config['train']['lr']
    checkpoint_folder = config['train']['checkpoint_folder']
    allowed_classes = config['train']['allowed_classes']
    learned_variance = config['model']['learned_variance']
    print("Using learned variance: ", learned_variance)
    T = config['diffusion_params']['T']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_mnist_dataloader(batch_size=batch_size, split="train", allowed_classes=allowed_classes)

    model = load_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t, log_sigma_square_t_clipped = get_constants(device, **config['diffusion_params'])

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_loss_non0 = 0
        total_loss_0 = 0
        total_loss_vlb = 0
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            batch_size = images.size(0)
            t_batch = torch.randint(0, T, (batch_size,), device=device)
            
            # get the diffusion params for this batch
            alpha_bar_t_batch = alpha_bar_t[t_batch].view(-1, 1, 1, 1)
            alpha_t_batch = alpha_t[t_batch].view(-1, 1, 1, 1)
            alpha_bar_t_minus_1_batch = alpha_bar_t_minus_1[t_batch].view(-1, 1, 1, 1)

            # create the noisy image
            x_t = torch.sqrt(alpha_bar_t_batch)*images + torch.sqrt(1 - alpha_bar_t_batch)*torch.randn_like(images)

            optimizer.zero_grad()

            if config["train"]["loss"] == "variational_lower_bound_loss":
                # this is basically a weighted mean predictor loss 
                mu_theta = model(x_t, t_batch)
                loss, loss_non0, loss_0 = variational_lower_bound_loss(mu_theta,
                                    original_x = images,
                                    noisy_x = x_t,
                                    batch_t = t_batch,
                                    alpha_t = alpha_t, 
                                    alpha_bar_t = alpha_bar_t, 
                                    alpha_bar_t_minus_1 = alpha_bar_t_minus_1, 
                                    sigma_square_t = sigma_square_t)
                
            elif config["train"]["loss"] == "noise_predictor_loss":
                # this loss supports both fixed and learned variance
                if not learned_variance:
                    mu_theta = model(x_t, t_batch)
                else:
                    mu_theta, var_theta = model(x_t, t_batch)

                true_noise = (x_t - torch.sqrt(alpha_bar_t_batch)*images)/torch.sqrt(1 - alpha_bar_t_batch)
                loss_theta, loss_non0, loss_0 = noise_predictor_loss(mu_theta, true_noise)
            
                # compute reverse variance loss
                if learned_variance:
                    log_sigma_square = compute_log_sigma_square(var_theta, t_batch, log_sigma_square_t_clipped, alpha_t, use_single_batch=False)
                    x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t_batch) * mu_theta) / torch.sqrt(alpha_bar_t_batch)

                    mu_theta_pred, _ = compute_mu_q(
                        original_x = x0_pred,
                        noisy_x = x_t,
                        alpha_t = alpha_t_batch,
                        alpha_bar_t = alpha_bar_t_batch,
                        alpha_bar_t_minus_1 = alpha_bar_t_minus_1_batch,
                    )

                    loss_vlb, _, _ = vlb_openai_like(
                        mu_theta = mu_theta_pred.detach(),
                        original_x = images,
                        noisy_x = x_t,
                        batch_t = t_batch,
                        alpha_t = alpha_t,
                        alpha_bar_t = alpha_bar_t,
                        alpha_bar_t_minus_1 = alpha_bar_t_minus_1,
                        log_sigma_square_t_clipped = log_sigma_square_t_clipped,
                        log_sigma_square = log_sigma_square,
                        fixed_var = False,
                    )

                    lambda_vlb = 1000.0
                    loss_vlb *= lambda_vlb
                
                loss = loss_theta if not learned_variance else loss_theta + loss_vlb

            elif config["train"]["loss"] == "mean_predictor_loss":
                # this is the VLB loss without the weighting
                mu_theta = model(x_t, t_batch)
                loss, loss_non0, loss_0 = mean_predictor_loss(mu_theta,
                                        noisy_x = x_t,
                                        original_x = images,
                                        alpha_t = alpha_t_batch,
                                        alpha_bar_t_minus_1 = alpha_bar_t_minus_1_batch,
                                        alpha_bar_t = alpha_bar_t_batch)
            
            elif config["train"]["loss"] == "denoising_loss":
                # predict x0 directly and compute MSE
                x0_pred = model(x_t, t_batch)
                loss, loss_non0, loss_0 = denoising_loss(x0_pred, images)

            elif config["train"]["loss"] == "score_matching_loss":
                score_pred = model(x_t, t_batch)
                loss, loss_non0, loss_0 = score_matching_loss(score_pred, images, x_t, alpha_bar_t_batch)

            else:
                raise ValueError(f"Unknown loss function: {config['train']['loss']}")
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_loss_non0 += loss_non0
            total_loss_0 += loss_0

            if learned_variance:
                total_loss_vlb += loss_vlb.item()

        avg_loss = total_loss / len(train_loader)
        avg_loss_non0 = total_loss_non0 / len(train_loader)
        avg_loss_0 = total_loss_0 / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        if learned_variance:
            avg_loss_vlb = total_loss_vlb / len(train_loader)
            print(f"Var Loss: {avg_loss_vlb:.4f}")
        print(f"Denoising Loss (t>0): {avg_loss_non0:.4f}, Reconstruction Loss (t=0): {avg_loss_0:.4f}")
        print()
        if (epoch+1)%10 == 0:
            torch.save(model.state_dict(), f"{checkpoint_folder}/model_epoch_sm_{epoch+1}.pth")

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--config_path', type=str, default='config/mnist.yml', help='Path to the configuration file.')
    args = argparse.parse_args()

    config = parse_config(args.config_path)
    train(config)