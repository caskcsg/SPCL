import copy
import functools
import json
import logging
import multiprocessing
import operator
import os
import pickle
import random
import time
import timeit
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import OrderedDict as odict
from typing import Optional
import tempfile
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import vocab
# from kmeans_pytorch import kmeans
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
warnings.simplefilter(action='ignore', category=FutureWarning)

CONFIG = {
    'bert_path': 'princeton-nlp/sup-simcse-roberta-large',
    'epochs' : 10,
    'lr' : 1e-3,
    'ptmlr' : 1e-5,
    'batch_size' : 32,
    'max_len' : 256,
    'bert_dim' : 1024,
    'pad_value' : 1,
    'mask_value' : 2,
    'dropout' : 0.1,
    'pool_size': 512,
    'support_set_size': 64,
    'num_classes' : 7,
    'warm_up' : 128,
    'dist_func': 'cosine',
    'data_path' : './MELD',
    'accumulation_steps' : 1,
    'avg_cluster_size' : 4096,
    'max_step' : 1024,
    'num_positive': 1,
    'ratio':1,
    'mu':0.5,
    'cl':True,
    'temperature': 0.08,
    'fgm': False,
    'train_obj': 'psup',
    'speaker_vocab' : '',
    'emotion_vocab' : '',
    'temp_path': '',
    'ngpus' : torch.cuda.device_count(),
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

tokenizer = AutoTokenizer.from_pretrained(CONFIG['bert_path'], local_files_only=False)
_special_tokens_ids = tokenizer('<mask>')['input_ids']
CLS = _special_tokens_ids[0]
MASK = _special_tokens_ids[1]
SEP = _special_tokens_ids[2]
CONFIG['CLS'] = CLS
CONFIG['SEP'] = SEP
CONFIG['mask_value'] = MASK


def dist(x, y):

    return (1-F.cosine_similarity(x, y, dim=-1))/2 + 1e-8

def score_func(x, y):

    return (1+F.cosine_similarity(x, y, dim=-1))/2 + 1e-8

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)