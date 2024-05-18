import torch
import wandb
import time
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dataset import ImageCaptioningDataset
from logger import Logger
from models import VanillaDecoderRNN, ImageEncoderFC, ImageEncoderAttention, DecoderWithAttention, ImageCaptioningModel, ViTImageEncoder
from utils import parse_arguments, read_settings, save_checkpoint, calculate_bleu_scores, load_checkpoint
from torch.utils.data import DataLoader
from torchvision.models import ResNet101_Weights
import numpy as np
from tqdm import tqdm


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
if torch.backends.mps.is_available():
    device = torch.device('mps')

print(f'{device = }')

torch.backends.cudnn.benchmark = True


def evaluate(encoder, decoder, dataloader):
    encoder.eval()
    decoder.eval()
    all_ground_truths_captions = []
    decoded_sentences = []
    with torch.no_grad():
        for image_tensor, tokenized_caption, caption_texts in tqdm(dataloader):
            encoder_outputs = encoder(image_tensor.to(device))
            decoder_outputs, decoder_hidden, _ = decoder(
                encoder_outputs)

            if decoder_outputs.ndim == 3:
                _, topi = decoder_outputs.topk(1)
                decoded_ids = topi.squeeze()
            else:
                decoded_ids = decoder_outputs
            for index, decoded_sentence in enumerate(decoded_ids):
                sentence = dataloader.dataset.tokens_to_sentence(
                    decoded_sentence)
                decoded_sentences.append(sentence)
                ground_truths = np.array(caption_texts)[:, index]
                all_ground_truths_captions.append(
                    [caption.split(' ') for caption in ground_truths])

    images = {}
    for i in range(1, 4):
        plot_image = image_tensor[-i].permute(1, 2, 0).numpy()
        image_captions = [' '.join(cap)
                          for cap in all_ground_truths_captions[-i]]
        image_predicted_captions = ' '.join(decoded_sentences[-i])
        bleu_scores = calculate_bleu_scores(decoded_sentences[-i:][:1],
                                            all_ground_truths_captions[-i:][:1])
        images[f'test_image_{i}'] = wandb.Image(plot_image, caption=f'prediction: {image_predicted_captions}\n'
                                                f'ground truth: {image_captions}\n'
                                                f'bleu scores: {bleu_scores}')

    bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu_scores(
        decoded_sentences, all_ground_truths_captions)

    return bleu_1, bleu_2, bleu_3, bleu_4, images


def plot_attention(input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Save the figure to a wandb artifact
    wandb.log({"attention_matrix": wandb.Image(fig)})

    # Close the figure to prevent it from being displayed in the notebook
    plt.close(fig)


def plot_attention_self(attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.clone().detach().cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Save the figure to a wandb artifact
    wandb.log({"attention_matrix": wandb.Image(fig)})

    # Close the figure to prevent it from being displayed in the notebook
    plt.close(fig)


def plot_and_show_attention(encoder, decoder, input_sentence, input_tensor, output_lang_voc):
    output_words, attentions = evaluate(
        encoder, decoder, input_tensor, output_lang_voc)
    plot_attention(input_sentence, output_words,
                   attentions[0, :len(output_words), :])


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion, teacher_forcing_ratio):
    encoder.train()
    decoder.train()

    total_loss = 0
    for image_tensor, tokenized_captions, caption_texts in tqdm(dataloader):
        for tokenized_caption in tokenized_captions:
            image_tensor = image_tensor.to(device)
            tokenized_caption = tokenized_caption.to(device)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoded_images = encoder(image_tensor)

            if np.random.random_sample() < teacher_forcing_ratio:
                decoder_outputs, decoder_hidden, _ = decoder(
                    encoded_images, tokenized_caption)
            else:
                decoder_outputs, decoder_hidden, _ = decoder(
                    encoded_images)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                tokenized_caption.view(-1)
            )
            loss.backward()

            total_loss += loss.item()
            decoder_optimizer.step()
            encoder_optimizer.step()
    return total_loss / len(dataloader)


def train(train_dataloader, test_dataloader, encoder, decoder, logger, n_epochs, teacher_forcing_ratio=1.0,
          encoder_learning_rate=0.001, decoder_learning_rate=0.001, early_stopping=np.inf, print_every=100, plot_every=100, checkpoint=None):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoder_learning_rate)
    decoder_optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_learning_rate)
    criterion = nn.NLLLoss(ignore_index=0)

    if checkpoint:
        load_checkpoint(encoder, decoder, encoder_optimizer,
                        decoder_optimizer, checkpoint)

    max_bleu_4 = 0
    epochs_since_improvement = 0
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.perf_counter()
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer,
                           decoder_optimizer, criterion, teacher_forcing_ratio)
        epoch_train_end = time.perf_counter()
        bleu_1, bleu_2, bleu_3, bleu_4, images = evaluate(encoder, decoder,
                                                          test_dataloader)
        epoch_evaluate_end = time.perf_counter()
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                f"Epoch: {epoch}/{n_epochs}, Loss {print_loss_avg}, Bleu scores: {bleu_1, bleu_2, bleu_3, bleu_4}")

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            logger.log({'bleu_1_score': bleu_1,
                        'bleu_2_score': bleu_2,
                        'bleu_3_score': bleu_3,
                        'bleu_4_score': bleu_4,
                        'loss_avg': plot_loss_avg,
                        'epoch_train_time': epoch_train_end - epoch_start,
                        'epoch_evaluate_time': epoch_evaluate_end - epoch_train_end,
                        **images})
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if bleu_4 > max_bleu_4:
            epochs_since_improvement = 0
            print(f'New best bleu_4 score: {bleu_4}')
            max_bleu_4 = bleu_4
            save_checkpoint(
                epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, logger)
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= early_stopping:
            print(
                f'No improvement in {early_stopping} epochs. Stopping training.')
            break


def main():
    args = parse_arguments()

    # Read settings from the YAML file
    settings = read_settings(args.config)

    # Access and use the settings as needed
    model_settings = settings.get('model', {})
    encoder_settings = settings.get('encoder', {})
    decoder_settings = settings.get('decoder', {})
    train_settings = settings.get('train', {})
    dataset_settings = settings.get('dataset', {})
    dataloader_settings = settings.get('dataloader', {})

    print(f'{model_settings = }\n{train_settings = }\n{dataset_settings = }\n'
          f'{dataloader_settings = }\n{encoder_settings = }\n{decoder_settings = }')

    train_dataset = ImageCaptioningDataset(
        **dataset_settings, stage='train', transforms=ResNet101_Weights.DEFAULT.transforms())
    train_dataloader = DataLoader(
        train_dataset, **dataloader_settings)

    test_dataset = ImageCaptioningDataset(
        **dataset_settings, stage='test', transforms=ResNet101_Weights.DEFAULT.transforms())
    test_dataloader = DataLoader(
        test_dataset, **dataloader_settings)

    if 'attention_type' in decoder_settings:
        encoder = ImageEncoderAttention(
            **model_settings, **encoder_settings).to(device)
        decoder = DecoderWithAttention(**model_settings, **decoder_settings,
                                       output_size=train_dataset.lang.n_words, device=device, encoder_dim=encoder.out_features).to(device)
    elif 'num_layers' in decoder_settings:
        encoder = ViTImageEncoder(**encoder_settings).to(device)
        decoder = ImageCaptioningModel(vocab_size=train_dataset.lang.n_words,
                                       max_length=dataset_settings['max_length'], **decoder_settings, hidden_size=encoder.hidden_size, device=device).to(device)
    else:
        encoder = ImageEncoderFC(
            **model_settings, **encoder_settings).to(device)
        decoder = VanillaDecoderRNN(**model_settings, **decoder_settings,
                                    output_size=train_dataset.lang.n_words, device=device).to(device)

    logger = Logger(
        settings, f'{str(encoder)}_{str(decoder)}', 'INM706_Image_Captioning')
    logger.watch(encoder)
    logger.watch(decoder)

    # evaluate(encoder, decoder, test_dataloader)
    train(train_dataloader, test_dataloader,
          encoder, decoder, logger, **train_settings)


if __name__ == '__main__':
    main()
