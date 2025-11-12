import torch
from data.mnist_dataloader import get_mnist_dataloader
from model.unet import UNet
from loss.losses import variational_lower_bound_loss, get_constants, noise_predictor_loss
import yaml
from tqdm import tqdm

def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(config):
    model = UNet(**config['model'])
    return model

def train(config):
    batch_size = config['train']['batch_size']
    num_epochs = config['train']['epochs']
    learning_rate = config['train']['lr']
    checkpoint_folder = config['train']['checkpoint_folder']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_mnist_dataloader(batch_size=batch_size, split="train")
    model = load_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t = get_constants(device, **config['diffusion_params'])

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_loss_non0 = 0
        total_loss_0 = 0
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            batch_size = images.size(0)
            t_batch = torch.randint(0, config['diffusion_params']['T'], (batch_size,), device=device)
            
            # get the diffusion params for this batch
            alpha_bar_t_batch = alpha_bar_t[t_batch].view(-1, 1, 1, 1)

            # create the noisy image
            x_t = torch.sqrt(alpha_bar_t_batch)*images + torch.sqrt(1 - alpha_bar_t_batch)*torch.randn_like(images)

            optimizer.zero_grad()
            mu_theta = model(x_t, t_batch)
            if config["train"]["loss"] == "mean_predictor_loss":
                loss, loss_non0, loss_0 = variational_lower_bound_loss(mu_theta,
                                    original_x = images,
                                    noisy_x = x_t,
                                    batch_t = t_batch,
                                    alpha_t = alpha_t, 
                                    alpha_bar_t = alpha_bar_t, 
                                    alpha_bar_t_minus_1 = alpha_bar_t_minus_1, 
                                    sigma_square_t = sigma_square_t)
                
            elif config["train"]["loss"] == "noise_predictor_loss":
                true_noise = (x_t - torch.sqrt(alpha_bar_t_batch)*images)/torch.sqrt(1 - alpha_bar_t_batch)
                loss, loss_non0, loss_0 = noise_predictor_loss(mu_theta, true_noise)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_loss_non0 += loss_non0
            total_loss_0 += loss_0

        avg_loss = total_loss / len(train_loader)
        avg_loss_non0 = total_loss_non0 / len(train_loader)
        avg_loss_0 = total_loss_0 / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        print(f"Denoising Loss (t>0): {avg_loss_non0:.4f}, Reconstruction Loss (t=0): {avg_loss_0:.4f}")
        print()
        if (epoch+1)%10 == 0:
            torch.save(model.state_dict(), f"{checkpoint_folder}/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    config = parse_config('config/mnist.yml')
    train(config)