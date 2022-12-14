import argparse
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import evaluation
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def main():
    

    """ 0. Init variables """
    col_query = "query"
    col_query_id = "query_id"
    col_product_id = "product_id" 
    col_product_title = "product_title"
    col_product_locale = "product_locale"
    col_esci_label = "esci_label" 
    col_small_version = "small_version"
    col_split = "split"
    col_gain = 'gain'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esci_label2gain = {
        'E' : 1.0,
        'S' : 0.1,
        'C' : 0.01,
        'I' : 0.0,
    }
    
    """ 1. Load data """    
  
    df_examples_products = pd.read_csv('./sampled_data_expandedquery_v2.csv')
    df_examples_products[col_gain] = df_examples_products[col_esci_label].apply(lambda esci_label: esci_label2gain[esci_label])
    
    df_examples_products = df_examples_products[[col_query_id, col_query, col_product_title, col_gain]]
    df_train = df_examples_products[df_examples_products[col_query_id].isin(list_query_id_train)]
    df_dev = df_examples_products[df_examples_products[col_query_id].isin(list_query_id_dev)]
    
    """ 2. Prepare data loaders """
    train_samples = []
    for (_, row) in df_train.iterrows():
        train_samples.append(InputExample(texts=[row[col_query], row[col_product_title]], label=float(row[col_gain])))
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size, drop_last=True)
    dev_samples = {}
    query2id = {}
    for (_, row) in df_dev.iterrows():
        try:
            qid = query2id[row[col_query]]
        except KeyError:
            qid = len(query2id)
            query2id[row[col_query]] = qid
        if qid not in dev_samples:
            dev_samples[qid] = {'query': row[col_query], 'positive': set(), 'negative': set()}
        if row[col_gain] > 0:
            dev_samples[qid]['positive'].add(row[col_product_title])
        else:
            dev_samples[qid]['negative'].add(row[col_product_title])
    evaluator = CERerankingEvaluator(dev_samples, name='train-eval')
    
    """ 3. Prepare Cross-enconder model:
        https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_cross-encoder_kd.py
    """
    model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
    num_epochs = 1
    num_labels = 1
    max_length = 512
    default_activation_function = torch.nn.Identity()
    model = CrossEncoder(
        model_name, 
        num_labels=num_labels, 
        max_length=max_length, 
        default_activation_function=default_activation_function, 
        device=device
    )
    loss_fct=torch.nn.MSELoss()
    evaluation_steps = 5000
    warmup_steps = 5000
    lr = 7e-6
    """ 4. Train Cross-encoder model """
    model.fit(
        train_dataloader=train_dataloader,
        loss_fct=loss_fct,
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=f"{args.model_save_path}_tmp",
        optimizer_params={'lr': lr},
    )
    model.save(args.model_save_path)


if __name__ == "__main__": 
    main()