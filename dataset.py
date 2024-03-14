from __future__ import unicode_literals, print_function, division
import os
import pandas as pd
import torch
import unicodedata
import re
from torch.utils.data import Dataset
from PIL import Image
import random
import shutil
import pickle

random.seed(43)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
if torch.backends.mps.is_available():
    device = torch.device('mps')


class Lang:
    EOS_TOKEN = 1

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class ImageCaptioningDataset(Dataset):
    def __init__(self, root_dir: str, stage: str, transforms=None, caption_file='results.csv', image_folder='flickr30k_images', max_length=20):
        self.root_dir = root_dir
        self.stage = stage
        self.transforms = transforms
        self.caption_file = caption_file
        self.image_folder = image_folder
        self.max_length = max_length

        self.load_captions(caption_file)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(
            self.root_dir, self.image_folder, self.stage, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        captions = self.captions[image_name]

        tokenized_captions = [self.tokenize_caption(
            caption) for caption in captions]

        return index, image, tokenized_captions, captions

    def load_captions(self, caption_file: str):
        df = pd.read_csv(os.path.join(
            self.root_dir, caption_file), sep='|', names=['image_name', 'comment_number', 'comment'])
        self.image_names = [image for image in os.listdir(
            os.path.join(self.root_dir, self.image_folder, self.stage)) if image.endswith('.jpg')]
        print(
            f'Loading {len(self.image_names)} {self.stage} images')
        image_captions = df.groupby('image_name')[
            'comment'].apply(list).to_dict()
        self.prepare_language_model(image_captions)

    def prepare_language_model(self, image_captions: dict[str, list[str]]):
        self.normalize_captions(image_captions)

        if os.path.exists('lang.pickle'):
            self.load_language_model()
            return

        self.lang = Lang('eng')

        for value in self.captions.values():
            for caption in value:
                self.lang.addSentence(caption)

        with open("lang.pickle", "wb") as output_file:
            pickle.dump(self.lang, output_file)

    def load_language_model(self):
        with open("lang.pickle", "rb") as input_file:
            self.lang = pickle.load(input_file)

    def normalize_captions(self, image_captions: dict[str, list[str]]):
        self.captions = {}
        for key, value in image_captions.items():
            normalized_captions = [self.normalize_string(
                caption) for caption in value]
            self.captions[key] = normalized_captions

    def tokenize_caption(self, sentence):
        tokenized_sentence = [self.lang.word2index[word]
                              for word in sentence.split(' ')][:self.max_length - 1]
        tokenized_sentence.append(self.lang.EOS_TOKEN)
        input_ids = torch.zeros(self.max_length, dtype=torch.long)
        input_ids[:len(tokenized_sentence)] = torch.tensor(tokenized_sentence)
        return input_ids

    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
        return s.strip()


def prepare_dataset(split=0.8):
    image_dir = os.path.join('flickr30k_images', 'flickr30k_images')
    image_names = [x for x in os.listdir(image_dir) if x.endswith('.jpg')]
    print(f'Number of images is {len(image_names)}')

    random.shuffle(image_names)

    split_index = int(len(image_names) * split)

    train_names = image_names[:split_index]
    test_names = image_names[split_index:]

    print(f'Number of training images is {len(train_names)}',
          f'Number of test images is {len(test_names)}', sep='\n')

    train_directory = os.path.join(image_dir, 'train')
    test_directory = os.path.join(image_dir, 'test')

    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    for name in train_names:
        shutil.move(os.path.join(image_dir, name),
                    os.path.join(train_directory, name))

    for name in test_names:
        shutil.move(os.path.join(image_dir, name),
                    os.path.join(test_directory, name))


if __name__ == '__main__':
    prepare_dataset()
