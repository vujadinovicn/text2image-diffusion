import torch
from loss.losses import get_useful_values
import yaml
from model.unet import UNet
import matplotlib.pyplot as plt
from tqdm import tqdm

T = 1000
config_path = 'config/mnist.yml'
checkpoint_path = '../checkpoints/model_sigmafix_epoch_10.pth'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

t = torch.linspace(0, T-1, steps=T).long().to(device)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

model = UNet(**config['model']).to(device)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

beta_t, alpha_t, alpha_bar_t, sigma_t = get_useful_values(t, **config['diffusion_params'])
sigma_t = sigma_t.to(device)

generated_images = []
x = torch.randn(1, 1, 32, 32).to(device) 
with torch.no_grad():
    for i in tqdm(reversed(range(T)), total=T):
        mu_theta = model(x, t[i].unsqueeze(0))

        if i>0:
            noise = torch.randn_like(x).to(device)
            x_new = mu_theta + sigma_t[i] * noise
        else:
            x_new = mu_theta
        
        if i % 100 == 0 or i == T-1:
            to_append = mu_theta.detach().clone()
            generated_images.append(to_append)
        
        x = x_new

x = x.squeeze().cpu().numpy()
x = (x + 1) / 2  
plt.imshow(x, cmap='gray')

plot_images = generated_images[-10:]
fig, axes = plt.subplots(1, len(plot_images), figsize=(15, 3))
for ax, img in zip(axes, plot_images):
    img = img.squeeze().cpu().numpy()
    img = (img + 1) / 2  # Rescale to [0, 1]
    # img = img.clip(0, 1)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
plt.show()