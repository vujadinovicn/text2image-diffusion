import torch
from loss.losses import get_constants
import yaml
from model.unet import UNet
import matplotlib.pyplot as plt
from tqdm import tqdm

T = 1000
B = 8

config_path = 'config/mnist.yml'
checkpoint_path = '../checkpoints/model_epoch_50.pth'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

t = torch.linspace(0, T, steps=T).long().to(device)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

model = UNet(**config['model']).to(device)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

alpha_t, alpha_bar_t, alpha_bar_t_minus_1, sigma_square_t = get_constants(device, **config['diffusion_params'])

generated_images = []
x = torch.randn(B, 1, 32, 32).to(device) 

with torch.no_grad():
    for i in tqdm(reversed(range(1000)), total=T):
        t_current = torch.tensor([i], device=device)
        
        sigma_current = sigma_square_t[t_current]
        alpha_t_current = alpha_t[t_current]
        alpha_bar_t_current = alpha_bar_t[t_current]

        eps_theta = model(x, t_current)

        if i > 0:
            noise = torch.randn_like(x).to(device)
            x_new = (x  - (1 - alpha_t_current)/torch.sqrt(1 - alpha_bar_t_current) * eps_theta)/torch.sqrt(alpha_t_current)
            x_new += torch.sqrt(sigma_current) * noise
        else:
            x_new = (x  - (1 - alpha_t_current)/torch.sqrt(1 - alpha_bar_t_current) * eps_theta)/torch.sqrt(alpha_t_current)        
        
        if i % 100 == 0 or i == T-1:
            to_append = x_new.detach().clone()
            generated_images.append(to_append)
        
        # x = x_new.clamp(-1, 1)
        x = x_new

x = x.squeeze().cpu().numpy()
print(x.shape)
x = (x + 1) / 2  
# plt.imshow(x, cmap='gray')

# plot_images = generated_images[-10:]
# take equally spaced 10 images from generated_images
num_images = len(generated_images)
n_cols = min(10, num_images)
indices = torch.linspace(0, num_images - 1, steps=n_cols).long().tolist()

fig, axes = plt.subplots(B, n_cols, figsize=(n_cols * 2, B * 2))
for i in range(B):
    for j, idx in enumerate(indices):
        img = generated_images[int(idx)][i]
        img = img.squeeze().cpu().numpy()
        img = (img + 1) / 2  # Rescale to [0,1]
        # choose correct axis handle depending on shape of axes
        if B == 1 and n_cols == 1:
            ax = axes
        elif B == 1:
            ax = axes[j]
        elif n_cols == 1:
            ax = axes[i]
        else:
            ax = axes[i, j]
        ax.imshow(img, cmap='gray')
        ax.axis('off')

plt.tight_layout()
plt.show()