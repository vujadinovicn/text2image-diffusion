import torch
from loss.losses import get_constants
import yaml
from model.unet import UNet
import matplotlib.pyplot as plt
from tqdm import tqdm

config_path = 'config/mnist.yml'
checkpoint_path = '../checkpoints/model_epoch_mp_30.pth'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

T = config['diffusion_params']['T']
alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t = get_constants(device, **config['diffusion_params'])

model = UNet(**config['model']).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

generated_images = []

x = torch.randn(1, 1, 32, 32, device=device)

with torch.no_grad():
    for t in tqdm(reversed(range(0, T)), total=T-1):
        t_current = torch.tensor([t], device=device)
        mu_theta = model(x, t_current)  # predicts mean of p(x_{t-1} | x_t)

        if t > 0:
            sigma_t = torch.sqrt(sigma_square_t[t_current]).view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            x_prev = mu_theta + sigma_t * noise
        else:
            x_prev = mu_theta

        if t % 100 == 0 or t == T-1 or t == 1:
            generated_images.append(x_prev.detach().clone())

        # x = x_prev.clamp(-1, 1)
        x = x_prev

# Final sample x_0
x0 = x.squeeze().cpu().numpy()
x0 = (x0 + 1) / 2
plt.imshow(x0, cmap='gray')
plt.axis('off')
plt.show()

# Visualization of trajectory (pick up to 10 spaced snapshots)
plot_images = []
num_images = len(generated_images)
indices = torch.linspace(0, num_images - 1, steps=min(10, num_images)).long()
for idx in indices:
    plot_images.append(generated_images[idx])

fig, axes = plt.subplots(1, len(plot_images), figsize=(15, 3))
for ax, img in zip(axes, plot_images):
    arr = img.squeeze().cpu().numpy()
    arr = (arr + 1) / 2
    ax.imshow(arr, cmap='gray')
    ax.axis('off')
plt.show()