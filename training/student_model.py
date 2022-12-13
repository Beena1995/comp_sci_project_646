import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from training.utils_training import *
from training.train_eval import *
import bert_representations
from bert_representations import create_bert_representations
import DistillBERTClass
import train_eval
from train_eval import fit
def create_data_frame(df,esci_label2gain):
    df['gain'] = df['esci_label'].apply(lambda esci_label: esci_label2gain[esci_label])

def split_data(df):
    train_data_df, val_data_df = train_test_split(df_excat,test_size=0.3,random_state=42)
    return train_data_df,val_data_df

def training_representations(train_df):
    train_data = create_bert_representations(train_data_df['expanded_query'].to_list(),train_data_df['product_title'].to_list(),train_data_df['gain'].to_list(),tokenizer,batch_size=16,max_length=512,pad_to_max_length=True,return_attention_mask=True,return_tensors='pt',path="train")
    return train_data

def validation_representations(val_df):
    val_data = create_bert_representations(train_data_df['expanded_query'].to_list(),train_data_df['product_title'].to_list(),train_data_df['gain'].to_list(),tokenizer,batch_size=16,max_length=512,pad_to_max_length=True,return_attention_mask=True,return_tensors='pt',path="train")
    return val_data

def train(df,labels):
    create_data_frame(df,labels)
    train_data_df,val_data_df = split_data(df)
    train_data = training_representations(train_data_df)
    val_data = validation_representations(val_data_df)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    student_model = DistillBERTClass()
    teacher_model = DistillBERTClass()
    binary_excat,binary_sub,binary_com,binary_ir = DistillBERTClassBinary(),DistillBERTClassBinary(),DistillBERTClassBinary(),DistillBERTClassBinary()
    student_model.to(device)
    load_model(teacher_model,'model_distil_expanded.pt')
    load_model(binary_excat,'model_distil_excat.pt'),load_model(binary_sub,'model_distil_sub.pt'),load_model(binary_com,'model_distil_com.pt'),load_model(binary_ir ,'model_distil_ir.pt')
    optimizer = AdamW(student_model.parameters(),lr=5e-5)
    eval_sampler = SequentialSampler(val_data)
    train_dataloader = DataLoader(train_data,shuffle=True, batch_size=args['train_batch_size'])
    eval_dataloader = DataLoader(val_data, sampler=eval_sampler, batch_size=args['eval_batch_size'])
    num_train_steps = num_steps(df)
    model.freeze_bert_encoder()
    fit_student(1)
    model.unfreeze_bert_encoder()
    fit_student()
    eval_multiclass()

def main():
    df = pd.read_csv("./sampled_query_expansion.csv")
    labels =  {'E' : 3,'S' : 2,'C' : 1,'I' : 0}

if __name__ == '__main__':
    main()















