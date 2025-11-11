from data_loader import dataset
from trainer.trainer import Trainer
import yaml
import torch
from models.network import UNet

if __name__ == "__main__":
    transform = dataset.get_transforms()
    train_dataset, test_dataset = dataset.get_datasets(transform)
    train_loader, test_loader = dataset.get_dataloaders(train_dataset, test_dataset, slice=True)
    
    config_path = "configs/config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = UNet(
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        img_resolution=int(config["image_size"]),
    ).to('cpu')

    trainer = Trainer(config, train_loader, test_loader, model)
    trainer.train()

    
    