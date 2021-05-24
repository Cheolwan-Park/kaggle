import pandas as pd
from typing import *
import torch
import numpy as np
from torch.utils import data
import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from itertools import chain
from tqdm import tqdm


def review_data_to_texts(filename: str):
    tsv = pd.read_csv(filename, sep='\t', error_bad_lines=False)
    reviews = tsv['review']
    texts = []
    for review in tqdm(reviews):
        sentences = review.split('.')
        for sentence in sentences:
            texts.append(re.sub(pattern='(?:\\|<[^>]*>|")', repl='', string=sentence))
    return texts


def get_uniques_from_nested_lists(nested_lists: List[List]) -> List:
    uniques = {}
    for one_line in nested_lists:
        for item in one_line:
            if not uniques.get(item):
                uniques[item] = 1
    return list(uniques.keys())


def get_item2idx_idx2item(items: List, unique: bool = False) -> Tuple[Dict, Dict]:
    item2idx, idx2item = dict(), dict()
    items_unique = items if unique else list(set(items))
    items_unique = ['<pad>'] + items_unique
    for idx, item in enumerate(items_unique):
        item2idx[item] = idx
        idx2item[idx] = item
    return item2idx, idx2item


class WordsStorage:
    def __init__(self):
        lemm = WordNetLemmatizer()
        self.lemmatizer = lemm.lemmatize
        self.tokenized_matrix = None
        self.pairs_matrix = None
        self.pairs_flat = None
        self.unique_tokens = None
        self.token2idx, self.idx2token = None, None
        self.vocab_size = 0

    def make_tokenized_matrix_eng(self, texts: List[str]):
        self.tokenized_matrix = []
        print('making tokenized matrix...')
        for text in tqdm(texts):
            self.tokenized_matrix.append([self.lemmatizer(word) for word in word_tokenize(text)])

    def make_token_indices(self):
        assert self.tokenized_matrix
        self.unique_tokens = get_uniques_from_nested_lists(self.tokenized_matrix)
        self.token2idx, self.idx2token = get_item2idx_idx2item(self.unique_tokens, unique=True)
        self.vocab_size = len(self.token2idx)

    def get_window_pairs(self, tokens: List[str], win_size: int = 2, as_index: bool = True) -> List[tuple]:
        assert self.token2idx
        window_pairs = []
        for idx, token in enumerate(tokens):
            start = max(0, idx-win_size)
            end = min(len(tokens), idx + win_size + 1)
            for win_idx in range(start, end):
                if not idx == win_idx:
                    pair = (token, tokens[win_idx])
                    pair = pair if not as_index else tuple(self.token2idx[t] for t in pair)
                    window_pairs.append(pair)
        return window_pairs

    def make_pairs_matrix(self, win_size: int, as_index: bool = True):
        assert self.tokenized_matrix
        self.pairs_matrix = []
        print('making window pairs...')
        for sentence in tqdm(self.tokenized_matrix):
            self.pairs_matrix.append(self.get_window_pairs(sentence, win_size, as_index))
        self.pairs_flat = list(chain.from_iterable(self.pairs_matrix))

    def save_data(self, filename: str):
        torch.save({
            'vocab_size': self.vocab_size,
            'token2idx': self.token2idx
        }, filename)


class W2VTrainDataset(data.Dataset):
    def __init__(self, review_filename: str, word_storage_path: str):
        super(W2VTrainDataset, self).__init__()

        self.words = WordsStorage()
        self.words.make_tokenized_matrix_eng(review_data_to_texts(review_filename))
        self.words.make_token_indices()
        self.words.make_pairs_matrix(win_size=2, as_index=True)
        self.words.save_data(word_storage_path)

        self.dataset_size = len(self.words.pairs_flat)

    def vocab_size(self):
        return self.words.vocab_size

    def get_input_layer(self, word_idx: int):    # one hot encoding
        layer = torch.zeros(self.vocab_size(), 1)
        layer[word_idx][0] = 1.0
        return layer

    def __getitem__(self, idx):
        center_i, context_i = self.words.pairs_flat[idx]
        x = self.get_input_layer(center_i).float()
        y = torch.from_numpy(np.array(context_i)).long()

        return x, y

    def __len__(self):
        return self.dataset_size


class ReviewClassifierDataset(data.Dataset):
    def __init__(self, review_filename: str, word_storage_filename: str, max_seq_len: int):
        super(ReviewClassifierDataset, self).__init__()

        self.max_seq_len = max_seq_len

        word_storage_data = torch.load(word_storage_filename)
        self.t2i = word_storage_data['token2idx']
        self.vocab_size = word_storage_data['vocab_size']

        lemm = WordNetLemmatizer
        self.lemmatizer = lemm.lemmatize

        tsv = pd.read_csv(review_filename, sep='\t', error_bad_lines=False)
        self.reviews = tsv['review']
        self.sentiments = tsv['sentiment']
        self.dataset_size = len(tsv)

    def get_word_idx(self, word):
        if word not in self.t2i:
            return None
        else:
            return self.t2i[word]

    def get_input_layer(self, word_idx: int):    # one hot encoding
        layer = torch.zeros(self.vocab_size, 1)
        layer[word_idx][0] = 1.0
        return layer

    def __getitem__(self, idx):
        words = [self.get_word_idx(word) for word in word_tokenize(self.reviews[idx])]
        words = np.array([word for word in words if word][:self.max_seq_len])
        if len(words) < self.max_seq_len:
            words = np.pad(words, (0, self.max_seq_len-len(words)), 'constant', constant_values=0)
        words = [self.get_input_layer(word) for word in words]
        words = torch.stack(words)
        sentiment = np.array([int(self.sentiments[idx])])

        return words.float(), torch.from_numpy(sentiment).float()

    def __len__(self):
        return self.dataset_size
