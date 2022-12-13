import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import utils_training
from utils_training import *

args = load_config()
#add config file
def eval_multiclass():
    # Run prediction for full data  
    all_logits = None
    all_labels = None
    big_index = None
    labels_f1 = None
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    loop = tqdm(eval_dataloader, leave=True)
    for input_ids,attention_mask,label_ids in loop:
        input_ids = input_ids.to(device)
        attentioin_mask = attention_mask.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            outputs= model(input_ids, attention_mask)
            loss = loss_function(outputs,label_ids)
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

        if big_index is None:
            big_index = big_idx.detach().cpu().numpy()
        else:    
            big_index = np.concatenate((big_index, big_idx.detach().cpu().numpy()), axis=0)

        if labels_f1 is None:
            labels_f1 = labels
        else:    
            labels_f1 = np.concatenate((labels_f1, labels), axis=0)
        
        label_ids = label_ids.detach().cpu().numpy()
        eval_loss += loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    f1 = f1_score(labels_f1,big_index,average='micro')

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'roc_auc' : roc_auc,
              'f1': f1
             }
    return result

def eval_binary():
    model_excat.eval()
    fin_targets = []
    fin_outputs = []
    loop = tqdm(eval_dataloader, leave=True)
    for input_ids,attention_mask,label_ids in loop:
      input_ids = input_ids.to(device,dtype=torch.long)
      attentioin_mask = attention_mask.to(device,dtype=torch.long)
      label_ids = label_ids.to(device,dtype=torch.float)

      with torch.no_grad():
        outputs = model_excat(input_ids, attentioin_mask)
        fin_targets.extend(label_ids.cpu().detach().numpy().tolist())
        fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    fin_outputs = np.array(fin_outputs) >= 0.5
    roc_auc = roc_auc_score(fin_targets, fin_outputs, average='micro')
    f1 = f1_score(fin_targets, fin_outputs)

    result = {'roc_auc' : roc_auc,
              'f1': f1
             }

    output_eval_file_numpy = os.path.join(args['output_dir'], "eval_results_np_excat.txt")
    np.save(output_eval_file_numpy, result)
    return result


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
            if (step + 1) % args['gradient_accumulation_steps'] == 0:          
                lr_this_step = args['learning_rate'] * warmup_linear(global_step/num_train_steps, args['warmup_proportion'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
        #eval()

def fit_student(epoch,max_epoch = args['num_train_epochs']):
    student_model.train()
    teacher_model.eval(),binary_excat.eval(),binary_sub.eval(),binary_com.eval(),binary_ir.eval()
    global_step = 0
    epoch = epoch
    for i_ in tqdm(range(epoch,int(max_epoch)), desc="Epoch"):
        epoch +=1
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids = batch
        student_logits = student_model(input_ids,input_mask)
        student_loss_1 = loss_function_student(student_logits,label_ids)
      
        with torch.no_grad():
            teacher_logits = teacher_model(input_ids,input_mask)
            excat_logits = binary_excat(input_ids,input_mask)
            loss_excat  = loss_function(excat_logits.squeeze(), label_ids.float())
=            sub_logits = binary_sub(input_ids,input_mask)
            loss_subt = loss_function(sub_logits.squeeze(), label_ids.float())
            com_logits = binary_com(input_ids,input_mask)
            loss_com = loss_function(com_logits.squeeze(), label_ids.float())
            ir_logits = binary_ir(input_ids,input_mask)
            loss_ir = loss_function(ir_logits.squeeze(), label_ids.float())
            #print(ir_logits)
            binary_logits = torch.cat((torch.sigmoid(ir_logits),torch.sigmoid(com_logits),torch.sigmoid(sub_logits),torch.sigmoid(excat_logits)),dim=1)
            soft_targets = 0.65 * teacher_logits + 0.35 * binary_logits
            #added_distil_logits = added_distil_logits/torch.norm(added_distil_logits)
        #distil_loss = loss_function_distil(F.log_softmax(input=(student_logits/2),dim=1),F.softmax(input=(added_distil_logits/2),dim=1))
        student_loss_2 = loss_function_student(student_logits,soft_targets)
        print("distil_loss",student_loss_2)
        loss = 0.3 * student_loss_1 + 0.7 * student_loss_2


        loss.backward()
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        big_val, big_idx = torch.max(student_logits.data, dim=1)
        n_correct = accuracy(big_idx, label_ids)
       
        if (step + 1) % args['gradient_accumulation_steps'] == 0:
              
    #             scheduler.batch_step()
                # modify learning rate with special warm up BERT uses
            lr_this_step = args['learning_rate'] * warmup_linear(global_step/num_train_steps, args['warmup_proportion'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

