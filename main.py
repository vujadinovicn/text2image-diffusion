from data_loader import dataset
from trainer.trainer import Trainer
import yaml

if __name__ == "__main__":
    transform = dataset.get_transforms()
    train_dataset, test_dataset = dataset.get_datasets(transform)
    train_loader, test_loader = dataset.get_dataloaders(train_dataset, test_dataset, slice=True)
    
    config_path = "configs/config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    trainer = Trainer(train_loader, test_loader, config)
    trainer._testing()
    