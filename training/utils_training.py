import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch
from transformers import BertModel, BertTokenizer,BertPreTrainedModel,AdamW
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import DataLoader, SequentialSampler,RandomSampler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

# fix config

def accuracy(out, labels):
    n_correct = (out==labels).sum().item()
    return n_correct

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def num_steps(df):
    return  int(len(df) / args['train_batch_size'] / args['gradient_accumulation_steps'] * args['num_train_epochs'])

def loss_function(output, label_ids):
  return torch.nn.BCEWithLogitsLoss()(output,label_ids)

def convert_to_all_labels(index):
  arr = [[0]* 4 for i in range(index.size)]
  for x in range(len(arr)):
    arr[x][index[x]] = 1
  return np.array(arr)

def load_config():
    with open('./config.json', 'r') as f:
        config = json.load(f)
