import torch
import wandb
import time
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dataset import ImageCaptioningDataset
from logger import Logger
from models import SelfAttnDecoderRNN, ImageEncoderFC, ImageEncoderAttention, BahdanauAttnDecoderGRU,  DecoderRNN
from utils import parse_arguments, read_settings, save_checkpoint
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, EfficientNet_V2_S_Weights
from torchtext.data.metrics import bleu_score
import numpy as np

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
if torch.backends.mps.is_available():
    device = torch.device('mps')

print(f'{device = }')


def evaluate(encoder, decoder, dataloader):
    all_ground_truths_captions = []
    decoded_sentences = []
    with torch.no_grad():
        for i, (image_name, image_tensor, tokenized_caption, caption_texts) in enumerate(dataloader):
            encoder_outputs = encoder(image_tensor.to(device))
            decoder_outputs, decoder_hidden, _ = decoder(
                encoder_outputs)

            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()
            for index, decoded_sentence in enumerate(decoded_ids):
                sentence = dataloader.dataset.tokens_to_sentence(
                    decoded_sentence)
                decoded_sentences.append(sentence)
                ground_truths = np.array(caption_texts)[:, index]
                all_ground_truths_captions.append(
                    [caption.split(' ') for caption in ground_truths])

            print(f'Step {i}/{len(dataloader)}', end='\r')

    for i in range(1, 4):
        plot_image = image_tensor[-i].permute(1, 2, 0).numpy()
        image_captions = [' '.join(cap)
                          for cap in all_ground_truths_captions[-i]]
        image_predicted_captions = ' '.join(decoded_sentences[-i])
        bleu_scores = calculate_bleu_scores(decoded_sentences[-i:][:1],
                                            all_ground_truths_captions[-i:][:1])
        wandb.log({f'test_image_{i}': wandb.Image(
            plot_image, caption=f'prediction: {image_predicted_captions}\n'
                                f'ground truth: {image_captions}\n'
                                f'bleu scores: {bleu_scores}')})

    bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu_scores(
        decoded_sentences, all_ground_truths_captions)

    return bleu_1, bleu_2, bleu_3, bleu_4


def calculate_bleu_scores(candidate_corpus, reference_corpus, max_n=4):
    weights = [1]
    bleu_scores = []
    for n in range(1, max_n+1):
        bleu = bleu_score(candidate_corpus,
                          reference_corpus, max_n=n, weights=weights)
        bleu_scores.append(bleu)
        weights.insert(0, 0)

    return bleu_scores


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

    total_loss = 0
    for i, (image_name, image_tensor, tokenized_captions, caption_texts) in enumerate(dataloader):
        for tokenized_caption in tokenized_captions:
            image_tensor = image_tensor.to(device)
            tokenized_caption = tokenized_caption.to(device)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs = encoder(image_tensor)

            if np.random.random_sample() < teacher_forcing_ratio:
                decoder_outputs, decoder_hidden, _ = decoder(
                    encoder_outputs, tokenized_caption)
            else:
                decoder_outputs, decoder_hidden, _ = decoder(
                    encoder_outputs)

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
        train_dataloader, test_dataloader, encoder, decoder, logger, n_epochs, teacher_forcing_ratio=1.0, learning_rate=0.001, early_stopping=np.inf,
        print_every=100, plot_every=100):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    max_bleu = 0
    epochs_since_improvement = 0
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.perf_counter()
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer,
                           decoder_optimizer, criterion, teacher_forcing_ratio)
        epoch_train_end = time.perf_counter()
        bleu_1, bleu_2, bleu_3, bleu_4 = evaluate(encoder, decoder,
                                                  test_dataloader)
        epoch_evaluate_end = time.perf_counter()
        logger.log({'epoch_train_time': epoch_train_end - epoch_start,
                   'epoch_evaluate_time': epoch_evaluate_end - epoch_train_end})
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                f"Epoch: {epoch}/{n_epochs}, Loss {print_loss_avg}, Bleu scores: {bleu_1, bleu_2, bleu_3, bleu_4}")

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            logger.log({'bleu_1_score': bleu_1, 'bleu_2_score': bleu_2,
                       'bleu_3_score': bleu_3, 'bleu_4_score': bleu_4, 'loss_avg': plot_loss_avg})
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if avg_bleu := np.mean([bleu_1, bleu_2, bleu_3, bleu_4]) > max_bleu:
            epochs_since_improvement = 0
            print(f'New best bleu score: {avg_bleu}')
            max_bleu = avg_bleu
            save_checkpoint(
                epoch, encoder, f'encoder_{encoder.__class__.__name__}', encoder_optimizer)
            save_checkpoint(
                epoch, decoder, f'decoder_{decoder.__class__.__name__}', decoder_optimizer)
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
        **dataset_settings, stage='train', transforms=ResNet50_Weights.DEFAULT.transforms())
    train_dataloader = DataLoader(
        train_dataset, **dataloader_settings)

    test_dataset = ImageCaptioningDataset(
        **dataset_settings, stage='test', transforms=ResNet50_Weights.DEFAULT.transforms())
    test_dataloader = DataLoader(
        test_dataset, **dataloader_settings)

    encoder = ImageEncoderAttention(
        **model_settings, **encoder_settings).to(device)
    decoder = BahdanauAttnDecoderGRU(**model_settings, **decoder_settings,
                                     output_size=train_dataset.lang.n_words, device=device).to(device)

    # encoder = ImageEncoderFC(
    #     **model_settings, **encoder_settings).to(device)
    # decoder = DecoderRNN(**model_settings, **decoder_settings,
    #                      output_size=train_dataset.lang.n_words, device=device).to(device)

    logger = Logger(
        settings, f'{encoder.__class__.__name__}_{decoder.__class__.__name__}', 'INM706_Image_Captioning')
    logger.watch(encoder)
    logger.watch(decoder)

    train(train_dataloader, test_dataloader,
          encoder, decoder, logger, **train_settings)
    # evaluate(encoder, decoder, test_dataloader)


if __name__ == '__main__':
    main()
