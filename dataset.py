from __future__ import unicode_literals, print_function, division
from io import open

import torch
import numpy as np
import unicodedata
import re
from torch.utils.data import Dataset, DataLoader

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
if torch.backends.mps.is_available():
    device = torch.device('mps')


class Lang:
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


class TranslationDataset(Dataset):
    MAX_LENGTH = 10

    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )
    SOS_token = 0
    EOS_token = 1

    def __init__(self, lang1='eng', lang2='fra', reverse=False):
        self.lang1 = lang1
        self.lang2 = lang2
        self.input_lang, self.output_lang, self.pairs = self.prepare_data(reverse)

    def __len__(self):
        return len(self.pairs)

    def load_data(self, reverse=False):
        # Read the file and split into lines
        lines = open('data/%s-%s.txt' % (self.lang1, self.lang2), encoding='utf-8'). \
            read().strip().split('\n')
        pairs = [[self.normalize_string(s) for s in l.split('\t')] for l in lines]

        # Reverse pairs, make Lang instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(self.lang2)
            output_lang = Lang(self.lang1)
        else:
            input_lang = Lang(self.lang1)
            output_lang = Lang(self.lang2)

        return input_lang, output_lang, pairs

    def prepare_data(self, reverse=False):
        input_lang, output_lang, pairs = self.load_data(reverse)
        pairs = self.filter_pairs(pairs)
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        self.input_lang_voc = input_lang.word2index
        self.output_lang_voc = output_lang.word2index
        return input_lang, output_lang, pairs

    def filter_pair(self, p):
        return len(p[0].split(' ')) < self.MAX_LENGTH and \
            len(p[1].split(' ')) < self.MAX_LENGTH and \
            p[1].startswith(self.eng_prefixes)

    def filter_pairs(self, pairs):
        return [pair for pair in pairs if self.filter_pair(pair)]

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

    def tokenize_sentence(self, lang, sentence):
        tokenized_sentence = [lang[word] for word in sentence.split(' ')]
        tokenized_sentence.append(self.EOS_token)
        return tokenized_sentence

    def tokenize_pair(self, pair):
        input_tensor = self.tokenize_sentence(self.input_lang_voc, pair[0])
        target_tensor = self.tokenize_sentence(self.output_lang_voc, pair[1])
        return (input_tensor, target_tensor)

    def __getitem__(self, index):
        input_sentence = self.pairs[index][0]
        output_sentence = self.pairs[index][1]
        in_sentence, out_sentence = self.tokenize_pair((input_sentence, output_sentence))
        input_ids = np.zeros(self.MAX_LENGTH, dtype=np.int32)
        target_ids = np.zeros(self.MAX_LENGTH, dtype=np.int32)
        input_ids[:len(in_sentence)] = in_sentence
        target_ids[:len(out_sentence)] = out_sentence
        return input_sentence, torch.tensor(input_ids, dtype=torch.long, device=device), torch.tensor(target_ids,
                                                                                                      dtype=torch.long,
                                                                                                      device=device)
