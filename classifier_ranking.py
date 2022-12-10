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
          return_attention_mask=return_attention_mask, # 0: padded tokens, 1: not padded tokens; taking into account the sequence length
          return_tensors=return_tensors,
      )
      input_ids.append(tokens_query['input_ids'])
      attention_masks.append(tokens_query['attention_mask'])
      if idx%10000==0: print(idx)
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels,dtype=torch.int64)
    train_data_save = {'input_ids' : input_ids , 'attention_mask' :attention_masks ,'labels': labels}
    torch.save(train_data_save, f'{location}parquet_data/bert_representations/' + path +'_tensor_dict_distil.pt')
    dataset = TensorDataset(
        input_ids, 
        attention_masks, 
        labels

    )
    # 0: query_inputs_ids, 1 : query_attention_mask, 2 : query_token_type_ids, 3
    return dataset

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def accuracy(out, labels):
    n_correct = (out==labels).sum().item()
    return n_correct

def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    print(y_pred.sigmoid())
    if sigmoid: y_pred = y_pred.sigmoid()
#     return ((y_pred>thresh)==y_true.byte()).float().mean().item()
    return np.mean(((y_pred>thresh)==y_true.byte()).float().cpu().numpy(), axis=1).sum()


def fbeta(y_pred, y_true, thresh=0.2, beta=2, eps=1e-9, sigmoid=True):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean().item()

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

num_train_steps = int(
        len(df) / args['train_batch_size'] / args['gradient_accumulation_steps'] * args['num_train_epochs'])

def loss_function(output, label_ids):
    return torch.nn.CrossEntropyLoss()(output,label_ids)

def fit(epoch = 0 , num_epochs = args['num_train_epochs']):
    global_step = 0
    model.train()
    epoch = epoch
    for i_ in tqdm(range(epoch,int(num_epochs)), desc="Epoch"):
        epoch +=1
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids = batch
           
            output = model(input_ids, input_mask)
            loss = loss_function(output, label_ids)
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            big_val, big_idx = torch.max(output.data, dim=1)
            n_correct = accuracy(big_idx, label_ids)
            if step%500==0:
              loss_step = tr_loss/len(train_dataloader)
              accu_step = (n_correct)/len(train_dataloader)
              print(f"Training Loss per 1000 steps: {loss_step}")
              print(f"Training Accuracy per 1000 steps: {accu_step}")
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
              
    #             scheduler.batch_step()
                # modify learning rate with special warm up BERT uses
                lr_this_step = args['learning_rate'] * warmup_linear(global_step/num_train_steps, args['warmup_proportion'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            },os.path.join(args['output_dir'], "model_distil.pt"))

        logger.info('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
        logger.info('Eval after epoc {}'.format(i_+1))
        eval()

def convert_to_all_labels(index):
  arr = [[0]* 4 for i in range(index.size)]
  for x in range(len(arr)):
    arr[x][index[x]] = 1
  return np.array(arr)

def eval():
    # Run prediction for full data  
    all_logits = None
    all_labels = None
    
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    loop = tqdm(eval_dataloader, leave=True)
    for input_ids,input_mask,label_ids in loop:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            #tmp_eval_loss = model(input_ids, input_mask)
            outputs= model(input_ids, input_mask)
            loss = loss_function(outputs,label_ids)

        #logits = logits.detach().cpu().numpy()
        #label_ids = label_ids.to('cpu').numpy()
        #tmp_eval_accuracy = accuracy(loss, label_ids)
        big_val, big_idx = torch.max(outputs.data, dim=1)
        tmp_eval_accuracy = accuracy(big_idx, label_ids)
        labels = label_ids.detach().cpu().numpy()
        label_arr = convert_to_all_labels(labels)

        if all_logits is None:
            all_logits = outputs.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, outputs.detach().cpu().numpy()), axis=0)
            
        if all_labels is None:
            all_labels = label_arr
        else:    
            all_labels = np.concatenate((all_labels, label_arr), axis=0)
        
        
        big_idx = big_idx.detach().cpu().numpy()
        label_ids = label_ids.detach().cpu().numpy()
        eval_loss += loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    
#     ROC-AUC calcualation
    # Compute ROC curve and ROC area for each class


        
    # Compute micro-average ROC curve and ROC area

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    f1 = f1_score(all_labels,all_logits)

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'roc_auc' : roc_auc,
              'f1': f1
             }

    output_eval_file_numpy = os.path.join(args['output_dir'], "eval_results_np.txt")
    output_eval_file = os.path.join(args['output_dir'], "eval_results_distil.txt")
    np.save(output_eval_file_numpy, result)
    print(result)
    return result

train_data_df, val_data_df = train_test_split(df,test_size=0.2,random_state=42)

train_data = create_bert_representations(train_data_df['query'].to_list(),train_data_df['product_title'].to_list(),train_data_df['gain'].to_list(),tokenizer,batch_size=16,max_length=512,pad_to_max_length=True,return_attention_mask=True,return_tensors='pt',path="train")

val_data = create_bert_representations(val_data_df['query'].to_list(),val_data_df['product_title'].to_list(),val_data_df['gain'].to_list(),tokenizer,batch_size=16,max_length=512,pad_to_max_length=True,return_attention_mask=True,return_tensors='pt',path="val")


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = DistillBERTClass()
state = torch.load(os.path.join(args['output_dir'], "model_distil.pt"))
epoch = state['epoch']
model.load_state_dict(state['model_state_dict'])
model.to(device)
optimizer = AdamW(model.parameters(),lr=5e-5)
optimizer.load_state_dict(state['optimizer_state_dict'])
#optimizer.to(device)
eval_sampler = SequentialSampler(val_data)
train_dataloader = DataLoader(train_data,shuffle=True, batch_size=args['train_batch_size'])
eval_dataloader = DataLoader(val_data, sampler=eval_sampler, batch_size=args['eval_batch_size'])

fit(num_epochs=args['num_train_epochs'])

eval()

# read data
df = pd.read_csv('/sampled_data_v2.csv')

esci_label2gain = {
        'E' : 3,
        'S' : 2,
        'C' : 1,
        'I' : 0,
    }
df['gain'] = df['esci_label'].apply(lambda esci_label: esci_label2gain[esci_label])


