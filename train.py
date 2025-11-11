import torch
from data.mnist_dataloader import get_mnist_dataloader
from model.unet import UNet
from loss.losses import mean_predictor_loss, get_useful_values
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

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            batch_size = images.size(0)
            t_batch = torch.randint(0, config['diffusion_params']['T'], (batch_size,), device=device)

            # get the diffusion params for this batch
            beta_batch, alpha_t_batch, alpha_bar_batch, sigma_batch = get_useful_values(t_batch, **config['diffusion_params'])

            # create the noisy image
            x_t = torch.sqrt(alpha_bar_batch)*images + torch.sqrt(1 - alpha_bar_batch)*torch.randn_like(images)

            optimizer.zero_grad()
            mu_theta = model(x_t, t_batch)
            loss = mean_predictor_loss(mu_theta=mu_theta, 
                                       t_batch=t_batch, 
                                       x_0=images, 
                                       x_t=x_t, 
                                       beta_batch=beta_batch, 
                                       alpha_t_batch=alpha_t_batch, 
                                       alpha_bar_batch=alpha_bar_batch, 
                                       sigma_batch=sigma_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"{checkpoint_folder}/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    config = parse_config('config/mnist.yml')
    train(config)