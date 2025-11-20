from model.unet import UNet
from model.unet_conditional import UNetCond as UNetConditional
import yaml
import torch

def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(config):
    if config['train']['use_guidance']:
        print("loaded conditional")
        model = UNetConditional(**config['model'])
    else:
        print("loaded UNCONDITIONAL")
        model = UNet(**config['model'])
    return model

def load_pretrained_model(config):
    if config['train']['use_guidance']:
        print("loaded conditional")
        model = UNetConditional(**config['model'])
    else:
        print("loaded UNCONDITIONAL")
        model = UNet(**config['model'])
    
    if config['train']['checkpoint_path']:
        try:
            model.load_state_dict(torch.load(config['train']['checkpoint_path']))
        except:
            print("COULDN'T LOAD TRAINED MODEL, RETURNING UNTRAINED MODEL")
            return model
    return model