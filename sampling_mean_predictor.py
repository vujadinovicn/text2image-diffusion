import torch
from loss.losses import get_useful_values
import yaml
from model.unet import UNet
import matplotlib.pyplot as plt
from tqdm import tqdm

T = 1000
config_path = 'config/mnist.yml'
checkpoint_path = '../checkpoints/model_epoch_7.pth'
device = "cuda" if torch.cuda.is_available() else "cpu"

t = torch.linspace(0, T-1, steps=T).long().to(device)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

model = UNet(**config['model']).to(device)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

beta_t, alpha_t, alpha_bar_t, sigma_t = get_useful_values(t, **config['diffusion_params'])

generated_images = []
x = torch.randn(1, 1, 32, 32) 
for i in tqdm(reversed(range(T)), total=T):
    mu_theta = model(x, t[i].unsqueeze(0))
    if i>0:
        x = mu_theta + sigma_t[i] * torch.randn_like(x)
    else:
        x = mu_theta
    if i % 100 == 0 or i == T-1:
        generated_images.append(x.detach().clone())

plot_images = generated_images[-10:]
fig, axes = plt.subplots(1, len(plot_images), figsize=(15, 3))
for ax, img in zip(axes, plot_images):
    img = img.squeeze().cpu().numpy()
    img = (img + 1) / 2  
    ax.imshow(img, cmap='gray')
plt.show()