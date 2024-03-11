import torch
import wandb
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dataset import ImageCaptioningDataset
from logger import Logger
from models import DecoderRNN, ImageEncoderRNN
from utils import parse_arguments, read_settings
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
if torch.backends.mps.is_available():
    device = torch.device('mps')

print(f'{device = }')


def evaluate(encoder, decoder, input_tensor, output_lang):
    EOS_token = 1
    with torch.no_grad():
        encoder_outputs = encoder(input_tensor)
        decoder_outputs, decoder_hidden, _ = decoder(
            encoder_outputs)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words


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

            encoder_outputs = encoder(image_tensor)
            decoder_outputs, decoder_hidden, _ = decoder(
                encoder_outputs, tokenized_caption)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                tokenized_caption.view(-1)
            )
            loss.backward()

            total_loss += loss.item()
            decoder_optimizer.step()
            encoder_optimizer.step()
        print(f'Step {i}/{len(dataloader)}', end='\r')

    return total_loss / len(dataloader)


def train(
        train_dataloader, encoder, decoder, logger, n_epochs, learning_rate=0.001,
        print_every=100, plot_every=100):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer,
                           decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f"Epoch: {epoch}/{n_epochs}, Loss {print_loss_avg}")
            print(evaluate(encoder, decoder,
                           train_dataloader.dataset[0], train_dataloader.dataset.lang))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            logger.log({'loss_avg': plot_loss_avg})
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


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

    print(f'{model_settings = }\n{train_settings = }\n{dataset_settings = }\n{encoder_settings = }\n{decoder_settings = }')

    wandb_logger = Logger('ImageCaptioning', 'INM706_Image_Captioning')
    logger = wandb_logger.get_logger()

    dataset = ImageCaptioningDataset(
        **dataset_settings, stage='train', transforms=ResNet50_Weights.DEFAULT.transforms())
    train_dataloader = DataLoader(
        dataset, **dataloader_settings)

    encoder = ImageEncoderRNN(**model_settings, **encoder_settings).to(device)
    decoder = DecoderRNN(**model_settings, **decoder_settings,
                         output_size=dataset.lang.n_words, device=device).to(device)

    train(train_dataloader, encoder, decoder, logger, **train_settings)


if __name__ == '__main__':
    main()
