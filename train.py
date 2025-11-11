import torch
from data.mnist_dataloader import get_mnist_dataloader
from model.unet import UNet
from loss.losses import mean_predictor_loss
import yaml

def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(config):
    model = UNet(**config['model'])
    return model

def train(config):
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['epochs']
    learning_rate = config['training']['lr']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_mnist_dataloader(batch_size=batch_size, split="train")
    model = load_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            batch_size = images.size(0)
            t_batch = torch.randint(0, config['diffusion']['T'], (batch_size,), device=device)

            optimizer.zero_grad()
            mu_theta = model(images, t_batch)
            loss = mean_predictor_loss(mu_theta, t_batch, images,**config['diffusion'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    config = parse_config('config/mnist.yml')
    model = load_model(config)