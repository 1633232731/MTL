from torch.utils.data import Dataset
from typing import List
import pandas as pd
import numpy as np
from pretrain_utils import split
import torch
from pretrain_trfm import TrfmSeq2seq
from build_vocab import WordVocab




class MyData(Dataset):

    def __init__(self,dataset_name,dataset_detail,trfm,train,label):
        self.dataset_name = dataset_name
        self.dataset_detail = dataset_detail
        # self.dataset = get_all_dataset()
        self.trfm = trfm
        self.train = train
        self.label = label

    def __getitem__(self, index):

        return self.train[index],self.label[index]

    def __len__(self):
        return len(self.train)





