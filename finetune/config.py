# Standard
import random
import json

# PIP
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

# Custom




class Config:
    # User Setting
    SEED = 94
    if_arabic = False 


    def __init__(self, filename, SEED=None):
        if SEED:
            self.SEED = SEED
        self.set_random_seed()
        self.read_json(filename)

        self.model = self.config_df['model'] 
        self.max_length = self.config_df['max_length'] 
        self.checkpoint_filename = self.config_df['checkpoint_filename'] 
        self.best_filename = self.config_df['best_filename'] 
        self.additional_tokens = self.config_df['additional_tokens'] 
        self.remove_special_tokens = self.config_df['remove_special_tokens'] 
        self.get_special_tokens()

        self.train_data = self.config_df['train_data'] 
        self.val_data = self.config_df['val_data'] 
        self.test_data = self.config_df['test_data'] 
        self.test_res = self.config_df['test_res'] 

        self.sent_col = self.config_df['sent_col'] 
        self.label_col = self.config_df['label_col'] 
        self.num_labels = self.config_df['num_labels'] 

        self.test_res_col = self.config_df['test_res_col'] 
        self.test_col = self.config_df['test_col'] 

        self.batch_size = self.config_df['batch_size'] 
        self.num_workers = self.config_df['num_workers'] 
        self.distributed=self.config_df['distributed'] 

        self.train = self.config_df['train'] 
        self.evaluate = self.config_df['evaluate'] 
        self.test = self.config_df['test'] 

        self.resume = self.config_df['resume'] 
        self.resume_model = self.config_df['resume_model'] 

        self.start_epoch = 0
        self.epochs = 6


    def get_special_tokens(self,filename='{ANY_SPECIAL_TOKENS}.json'):

        with open(filename,'r') as f:
            self.additional_tokens += list(json.load(f).values())

    def read_json(self,filename):
        with open(filename,'r') as f:
            self.config_df = json.load(f)

    def set_random_seed(self):
        print(f'=> SEED : {self.SEED}')

        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(self.SEED)  

        pl.seed_everything(self.SEED)
# 
