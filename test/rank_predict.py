import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch
from transformers import BertModel, BertTokenizer,BertPreTrainedModel,AdamW
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import DataLoader, SequentialSampler,RandomSampler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score, accuracy_score, ndcg_score
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from pathlib import Path
from BertClassifier import DistillBERTClass
from training.utils_training import *

args = {
    "no_cuda": False,
    "bert_model": 'distilbert-base-uncased',
    "max_seq_length": 512,
    "do_train": True,
    "do_eval": True,
    "do_lower_case": True,
    "batch_size": 32,
}

def predict(model,data):
    
    scores = np.zeros(len(data))
    all_logits = None
    all_labels = None
    big_index = None
    labels_f1 = None
    tot_loss, tot_accuracy = 0, 0
    nb_steps, nb_examples = 0, 0
    n_examples = len(data)
        
    model.eval()   
    with torch.no_grad():
        for i in tqdm(range(0, n_examples, args["batch_size"])):        
            j = min(i + args["batch_size"], n_examples)
            input_ids = data[i:j][0]
            attention_mask = data[i:j][1]
            label_ids = data[i:j][2]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label_ids = label_ids.to(device)
            outputs= model(input_ids, attention_mask)
            loss = loss_function(outputs,label_ids)
            prob = F.softmax(outputs)
            rank_score = prob.detach().cpu().numpy()
            j = i+rank_score.shape[0]
            # print('rank score 1 ', rank_score)
            scores[i:j]  = rank_score[:,3] + 0.1 * rank_score[:,2] + 0.01 * rank_score[:,1]
            i = j

            big_val, big_idx = torch.max(outputs.data, dim=1)
            tmp_accuracy = accuracy(big_idx, label_ids)
            labels = label_ids.detach().cpu().numpy()
            label_arr = convert_to_all_labels(labels)
            if all_logits is None:
                all_logits = prob.detach().cpu().numpy()
                # print(all_logits.shape)
            else:
                all_logits = np.concatenate((all_logits, prob.detach().cpu().numpy()), axis=0)
                # print(all_logits.shape)

            if all_labels is None:
                  all_labels = label_arr
            else:    
                  all_labels = np.concatenate((all_labels, label_arr), axis=0)

            if big_index is None:
                big_index = big_idx.detach().cpu().numpy()
            else:    
                big_index = np.concatenate((big_index, big_idx.detach().cpu().numpy()), axis=0)

            if labels_f1 is None:
                labels_f1 = labels
            else:    
                labels_f1 = np.concatenate((labels_f1, labels), axis=0)

            label_ids = label_ids.detach().cpu().numpy()
            tot_loss += loss.mean().item()
            tot_accuracy += tmp_accuracy

            nb_examples += input_ids.size(0)
            nb_steps += 1

    tot_loss = tot_loss / nb_steps
    tot_accuracy = tot_accuracy / nb_examples

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):
        print('logits ', all_logits[:, i])
        print('labels ', all_labels[:, i])
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    print(np.unique(big_index, return_counts=True))   
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    f1_micro = f1_score(labels_f1,big_index,average='micro')
    f1_macro = f1_score(labels_f1,big_index,average='macro')

    result = {'eval_loss': tot_loss,
              'eval_accuracy': tot_accuracy,
              'roc_auc' : roc_auc,
              'f1_micro': f1_micro,
              'f1_macro' : f1_macro
             }

    output_metric_file_numpy = "results_np_expanded.txt"
    np.save(output_metric_file_numpy, result)
    print(result)
    return result, scores


def cal_ndcg(df_test)
    tot_ndcg = 0
    for idq in df_test['query_id'].unique():
        df_temp = df_test[df_test['query_id']==idq]
        tot_ndcg += ndcg_score(np.asarray([df_temp['target'].tolist()]), np.asarray([df_temp['scores'].tolist()]))
    return tot_ndcg

def main():
    test_data_df = pd.read_csv('data/test/test_data_expandedquery.csv')
    test_data_df['query'] = test_data_df['expanded_query']
    esci_label2gain = {
        'E' : 3,
        'S' : 2,
        'C' : 1,
        'I' : 0,
    }
    test_data_df['gain'] = test_data_df['esci_label'].apply(lambda esci_label: esci_label2gain[esci_label])
    test_data = create_bert_representations(test_data_df['query'].to_list(),test_data_df['product_title'].to_list(),test_data_df['gain'].to_list(),tokenizer,batch_size=16,max_length=512,pad_to_max_length=True,return_attention_mask=True,return_tensors='pt',path="test")
    checkpoint = torch.load("/model1/model_distil_expanded.pt", map_location=torch.device('cpu'))
    model = DistillBERTClass()
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    res,scores = predict(model, test_data)
    test_data_df['scores'] = scores
    test_data_df = test_data_df.sort_values(by=['query_id', 'scores'], ascending=False)
    test_data_df['target'] = test_data_df['esci_label'].apply(lambda esci_label: esci_label2gain[esci_label])
    ndcg = cal_ndcg(df_test)

if if __name__ == '__main__':
    main()