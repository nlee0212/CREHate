# Standard
import os
import sys
import re
import random

# PIP
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, T5Tokenizer

# Custom
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import utils

class CSV2Dataset(Dataset):
    def __init__(self, cfg, filename, option):
        self.cfg = cfg
        self.filename = filename
        self.option = option

        self.df = self.get_df()
        if self.option == 'train':
            self.df = self.df.sample(frac=1) 

        if 't5' in self.cfg.model:
            self.tokenizer = T5Tokenizer.from_pretrained(self.cfg.model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model)
            if len(self.cfg.additional_tokens) > 0:
                num_added_toks = self.tokenizer.add_tokens(self.cfg.additional_tokens)

        self.xs = self.preprocess()

        if self.option != 'test':
            self.ys = self.get_ys()

    def get_df(self):
        return pd.read_csv(self.filename,encoding='utf-8')

    def get_tokenizer(self):
        return self.tokenizer

    def preprocess(self):
        if self.option == 'test':
            xs = self.df[self.cfg.test_col] 
        else:
            xs = self.df[self.cfg.sent_col]


        if self.cfg.remove_special_tokens:
            xs = utils.remove_special_tokens(xs)

        if self.cfg.if_arabic:
            arabert_prep = ArabertPreprocessor(model_name=self.cfg.model)
            xs = [arabert_prep.preprocess(x) for x in xs]


        encoding = self.tokenizer(
            list(xs),
            padding='max_length',
            max_length=self.cfg.max_length,
            truncation=True,
            return_tensors='pt'
        )

        xs = [(input_ids,attention) for input_ids,attention in zip(encoding.input_ids,encoding.attention_mask)]
        return xs

    def get_ys(self):
        if 't5' in self.cfg.model:
            encoding = self.tokenizer(
                [str(label) for label in self.df[self.cfg.label_col]],
                padding='max_length',
                max_length=self.cfg.max_length,
                truncation=True,
                return_tensors='pt' 
            )
            lm_labels = encoding.input_ids 
            lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

            ys = [(input_ids,attention) for input_ids,attention in zip(lm_labels,encoding.attention_mask)]

            return ys
        return [int(y) for y in self.df[self.cfg.label_col].tolist()]

    def __getitem__(self, idx):
        if self.option == 'test':
            return self.xs[idx]
        else:
            return self.xs[idx], self.ys[idx]

    def __len__(self):
        return len(self.xs)

def batch_sampling(batch_size,data_len,is_test=False):
    seq_lens = range(data_len)
    sample_indices = [ seq_lens[i:i+batch_size] for i in range(0,data_len, batch_size)]

    if not is_test:
        random.shuffle(sample_indices) 

    return sample_indices
