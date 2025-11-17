from model.unet import UNet
import yaml
import torch

def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(config):
    model = UNet(**config['model'])
    return model

def load_pretrained_model(config):
    model = UNet(**config['model'])
    model.load_state_dict(torch.load(config['train']['checkpoint_path']))
    return model