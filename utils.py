import torch
import yaml
import os
import pandas as pd
import argparse
import socket
import wandb
from torchtext.data.metrics import bleu_score


def save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, logger=None):
    ckpt = {
        'epoch': epoch,
        'encoder_weights': encoder.state_dict(),
        'decoder_weights': decoder.state_dict(),
        'encoder_optimizer_state': encoder_optimizer.state_dict(),
        'decoder_optimizer_state': encoder_optimizer.state_dict()
    }

    file_name = f"{str(encoder)}_{str(decoder)}_ckpt_{str(epoch)}.pth"

    directory_name = 'weights'
    os.makedirs(directory_name, exist_ok=True)
    save_path = os.path.join(directory_name, file_name)
    torch.save(ckpt, save_path)
    if logger:
        artifact = wandb.Artifact(
            name=file_name, type="model")
        # Add dataset file to artifact
        artifact.add_file(local_path=save_path)
        logger.log_artifact(artifact)
    return save_path


def load_checkpoint(model, file_name):
    ckpt = torch.load(os.path.join('weights', file_name))
    model_weights = ckpt['model_weights']
    model.load_state_dict(model_weights)
    print("Model's pretrained weights loaded!")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Process settings from a YAML file.')
    parser.add_argument('--config', type=str, default='configs/configFC.yaml',
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


def calculate_bleu_scores(candidate_corpus, reference_corpus, max_n=4):
    bleu_scores = []
    for n in range(1, max_n+1):
        weights = [1/n] * n
        bleu = bleu_score(candidate_corpus,
                          reference_corpus, max_n=n, weights=weights)
        bleu_scores.append(bleu)

    return bleu_scores
