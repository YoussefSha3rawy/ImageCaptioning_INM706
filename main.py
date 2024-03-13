import torch
import wandb
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dataset import ImageCaptioningDataset
from logger import Logger
from models import DecoderRNN, ImageEncoderRNN, ImageEncoderSelfAttentionRNN, AttnDecoderGRU
from utils import parse_arguments, read_settings, save_checkpoint
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchtext.data.metrics import bleu_score
import numpy as np

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
if torch.backends.mps.is_available():
    device = torch.device('mps')

print(f'{device = }')


def evaluate(encoder, decoder, dataloader):
    output_lang = dataloader.dataset.lang
    decoded_sentences = []
    all_captions = []
    for i, (index, image_tensor, tokenized_captions, captions) in enumerate(dataloader):
        all_captions.extend(np.transpose(np.array(captions), [1, 0]))
        with torch.no_grad():
            encoder_outputs = encoder(image_tensor.to(device))
            decoder_outputs, decoder_hidden, _ = decoder(
                encoder_outputs)

            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()
            for decoded_sentence in decoded_ids:
                decoded_words = []
                for idx in decoded_sentence:
                    if idx.item() == output_lang.EOS_TOKEN:
                        break
                    decoded_words.append(
                        output_lang.index2word[idx.item()])
                decoded_sentences.append(decoded_words)
    wandb.log({'test_image': wandb.Image(image_tensor[-1].cpu().permute(1, 2, 0).numpy(
    ), caption=f'prediction: {" ".join(decoded_words)}\nground truth: {np.array(captions)[:, -1]}')})
    bleu = bleu_score(decoded_sentences, list(
        map(lambda x: [y.split(' ') for y in x], all_captions)))

    return bleu


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
                decoder_optimizer, criterion):

    total_loss = 0
    for i, (index, image_tensor, tokenized_captions, captions) in enumerate(dataloader):
        image_tensor = image_tensor.to(device)
        tokenized_captions = [tokenized_caption.to(
            device) for tokenized_caption in tokenized_captions]
        for tokenized_caption in tokenized_captions:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(image_tensor)
            decoder_outputs, decoder_hidden, _ = decoder(
                encoder_outputs, encoder_hidden, tokenized_caption)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                tokenized_caption.view(-1)
            )
            loss.backward()

            total_loss += loss.item()
            decoder_optimizer.step()
            encoder_optimizer.step()
        print(f'Step {i}/{len(dataloader)}', end='\r')
    print()
    return total_loss / len(dataloader)


def train(
        train_dataloader, test_dataloader, encoder, decoder, logger, n_epochs, learning_rate=0.001,
        print_every=100, plot_every=100):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    max_bleu = 0
    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer,
                           decoder_optimizer, criterion)
        bleu = evaluate(encoder, decoder,
                        test_dataloader)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                f"Epoch: {epoch}/{n_epochs}, Loss {print_loss_avg}, Bleu score: {bleu}")

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            logger.log({'loss_avg': plot_loss_avg})
            logger.log({'bleu_score': bleu})
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if bleu > max_bleu:
            print(f'New best bleu score: {bleu}')
            max_bleu = bleu
            save_checkpoint(
                epoch, encoder, f'encoder_{encoder.cnn.__class__.__name__}', encoder_optimizer)
            save_checkpoint(
                epoch, decoder, f'decoder_{decoder.__class__.__name__}', decoder_optimizer)


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

    wandb_logger = Logger('ImageCaptioning', 'INM706_Image_Captioning')
    logger = wandb_logger.get_logger()

    train_dataset = ImageCaptioningDataset(
        **dataset_settings, stage='train', transforms=ResNet50_Weights.DEFAULT.transforms())
    train_dataloader = DataLoader(
        train_dataset, **dataloader_settings)

    test_dataset = ImageCaptioningDataset(
        **dataset_settings, stage='test', transforms=ResNet50_Weights.DEFAULT.transforms())
    test_dataloader = DataLoader(
        test_dataset, **dataloader_settings)

    encoder = ImageEncoderRNN(**model_settings, **encoder_settings).to(device)
    decoder = DecoderRNN(**model_settings, **decoder_settings,
                         output_size=train_dataset.lang.n_words, device=device).to(device)
    encoder = ImageEncoderSelfAttentionRNN(
        **model_settings, **encoder_settings).to(device)
    decoder = AttnDecoderGRU(**model_settings, **decoder_settings,
                             output_size=train_dataset.lang.n_words, device=device).to(device)
    logger.watch(encoder)
    logger.watch(decoder)

    train(train_dataloader, test_dataloader,
          encoder, decoder, logger, **train_settings)
    # evaluate(encoder, decoder, test_dataloader)


if __name__ == '__main__':
    main()
