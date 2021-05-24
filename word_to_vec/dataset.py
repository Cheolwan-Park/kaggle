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
    items_unique = items if unique else set(items)
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


class W2VTrainDataset(data.Dataset):
    def __init__(self, review_filename):
        super(W2VTrainDataset, self).__init__()

        self.words = WordsStorage()
        self.words.make_tokenized_matrix_eng(review_data_to_texts(review_filename))
        self.words.make_token_indices()
        self.words.make_pairs_matrix(win_size=2, as_index=True)

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
