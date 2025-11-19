import torch
from data.mnist_dataloader import get_mnist_dataloader
from loss.losses import variational_lower_bound_loss, get_constants, noise_predictor_loss, mean_predictor_loss, denoising_loss, score_matching_loss
from tqdm import tqdm
from utils.utils import parse_config, load_model
import argparse

def train(config):

    # Extract training parameters from config
    batch_size = config['train']['batch_size']
    num_epochs = config['train']['epochs']
    learning_rate = config['train']['lr']
    checkpoint_folder = config['train']['checkpoint_folder']
    allowed_classes = config['train']['allowed_classes']
    label_drop_prob = config['train'].get('label_drop_prob') # probability of dropping label for classifier-free guidance

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    null_class_label = len(allowed_classes) + 1

    train_loader = get_mnist_dataloader(batch_size=batch_size, split="train", allowed_classes=allowed_classes)

    model = load_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    _, alpha_bar_t, _, _ = get_constants(device, **config['diffusion_params'])

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            label = label.to(device)

            ##################################################################
            # remove this line when training on all classes starting from 0 to 9
            label = label - 1 # since we are training on classes 1,2,3
            ##################################################################

            # randomly drop labels for classifier-free guidance
            mask = torch.rand(label.shape, device=device) < label_drop_prob
            label[mask] = null_class_label

            t_batch = torch.randint(0, config['diffusion_params']['T'], (batch_size,), device=device)
            
            # get the diffusion params for this batch
            alpha_bar_t_batch = alpha_bar_t[t_batch].view(-1, 1, 1, 1)

            # create the noisy image
            x_t = torch.sqrt(alpha_bar_t_batch)*images + torch.sqrt(1 - alpha_bar_t_batch)*torch.randn_like(images)

            optimizer.zero_grad()
            score = model(x_t, t_batch, label)
            
            assert config["train"]["loss"] == "score_matching_loss", "Use score_matching_loss for conditional generation."
            loss, _, _ = score_matching_loss(score, images, x_t, alpha_bar_t_batch)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        print()
        if (epoch+1)%10 == 0:
            torch.save(model.state_dict(), f"{checkpoint_folder}/model_epoch_sm_{epoch+1}.pth")

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--config_path', type=str, default='config/mnist.yml', help='Path to the configuration file.')
    args = argparse.parse_args()

    config = parse_config(args.config_path)
    train(config)