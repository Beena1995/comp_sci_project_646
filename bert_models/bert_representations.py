import torch
from torch.utils.data import TensorDataset
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np
import pandas as pd

def create_bert_representations(query,product,labels,tokenizer,batch_size=16,max_length=512,pad_to_max_length=True,return_attention_mask=True,return_tensors='pt',path="train"):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    for idx in range(0,len(query),batch_size):
      tokens_query = tokenizer.batch_encode_plus(
          query[idx:idx+batch_size], 
          product[idx:idx+batch_size],
          pad_to_max_length=pad_to_max_length,
          #add_special_tokens=True,
          max_length=max_length,
          return_attention_mask=return_attention_mask,
          return_tensors=return_tensors,
      )
      input_ids.append(tokens_query['input_ids'])
      attention_masks.append(tokens_query['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels,dtype=torch.int64)
    dataset = TensorDataset(
        input_ids, 
        attention_masks, 
        labels

    )
    return dataset