import torch
import yaml
import os
import pandas as pd
import argparse
import socket


def save_checkpoint(epoch, model, model_name, optimizer):
    ckpt = {'epoch': epoch, 'model_weights': model.state_dict(
    ), 'optimizer_state': optimizer.state_dict()}
    file_name = f"{model_name}_ckpt_{str(epoch)}.pth"

    directory_name = 'weights'
    os.makedirs(directory_name, exist_ok=True)
    torch.save(ckpt, os.path.join(directory_name, file_name))


def load_checkpoint(model, file_name):
    ckpt = torch.load(os.path.join('weights', file_name))
    model_weights = ckpt['model_weights']
    model.load_state_dict(model_weights)
    print("Model's pretrained weights loaded!")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Process settings from a YAML file.')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML configuration file')
    return parser.parse_args()


def read_settings(config_path):
    with open(config_path, 'r') as file:
        settings = yaml.safe_load(file)
    hostname = socket.gethostname()

    if hostname.endswith('local'):  # Example check for local machine names
        print("Running on Macbook locally")
        settings['dataset']['root_dir'] = settings['dataset']['root_dir_local']

    del settings['dataset']['root_dir_local']
    return settings
