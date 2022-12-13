import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import random
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import accuracy_score, f1_score
import os
import pathlib
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

class QueryProductClassifier(nn.Module):

    def __init__(self, size_petrained=768, dense_hidden_dim=126, num_dense_layers=1, num_labels=1, dropout_rate=0.1):
        super(QueryProductClassifier, self).__init__()
        self.num_labels = 1 if num_labels <= 2 else num_labels
        self.size_petrained = size_petrained * 2
        fc_layers = []
        prev_dim = self.size_petrained
        self.dropout_embedding = nn.Dropout(dropout_rate)
        for _ in range(num_dense_layers):
            fc_layers.append(nn.Linear(prev_dim, dense_hidden_dim, bias=True))
            prev_dim = dense_hidden_dim
        fc_layers.append(nn.Linear(prev_dim, self.num_labels))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, query_embedding, Product_embedding):
        # query_embedding: [batch_size, num_features]
        # product_embedding: [batch_size, num_features]
        embedding = torch.cat((query_embedding, Product_embedding), 1) # [batch_size, num_features * 2]
        embedding = self.dropout_embedding(embedding) # [batch_size, num_features * 2]
        logits = self.fc(embedding).squeeze(-1) # [batch_size, num_labels]
        return logits


def generate_dataset(query_embedding, product_embedding, Y):
    query_embedding = torch.tensor(query_embedding).type(torch.FloatTensor)
    product_embedding = torch.tensor(product_embedding).type(torch.FloatTensor)
    Y = torch.tensor(Y)
    dataset = TensorDataset(
         query_embedding, 
         product_embedding, 
         Y,
    )
    return dataset

def set_seed(random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

ef train(model, train_inputs, validation_inputs, path_model, device='cpu', batch_size=128, weight_decay=0.01, num_train_epochs=4, 
    lr=5e-5, eps=1e-8, num_warmup_steps=0, max_grad_norm=1, validation_steps=250, random_seed=42):
    
    set_seed(random_seed=random_seed)

    """ Step 0: prapare data loaders and model """
    train_sampler = RandomSampler(train_inputs)
    train_dataloader = DataLoader(train_inputs, sampler=train_sampler, batch_size=batch_size)
    validation_sampler = SequentialSampler(validation_inputs)
    validation_dataloader = DataLoader(validation_inputs, sampler=validation_sampler, batch_size=batch_size)
    model.to(device)
    
    """ Step 1: preparere optimizer """
    num_training_batches = len(train_dataloader)
    total_training_steps = num_training_batches * num_train_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)  
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps)

    """ Step 2: preparere variables """
    validation_metric = np.empty(len(validation_dataloader))
    validation_loss = np.empty_like(validation_metric)
    
    best_metric_value = 0.0
    best_model = None
    input_metric = {
        'y_true' : None, 
        'y_pred' : None,
    }

    if model.num_labels > 2:
        criterion = nn.CrossEntropyLoss()
        metric = accuracy_score
    else:
        criterion = nn.BCELoss()
        metric = f1_score
        input_metric['average'] = 'macro'
    
    """ Step 3: experiments """
    
    for idx_epoch in range(0, num_train_epochs):
        
        """ Step 3.1: Training """
        for (idx_train_batch, train_batch) in enumerate(train_dataloader):
            model.train()
            # 0: query_embedding, 1: product_embedding, 2: labels 
            labels = train_batch[2].to(device)
            optimizer.zero_grad()
            logits = model(train_batch[0].to(device), train_batch[1].to(device))

            if model.num_labels > 2:
                loss = criterion(logits.view(-1, model.num_labels), labels.view(-1))
                logits = logits.detach().cpu().numpy()
                hypothesis = np.argmax(logits, axis=1)
            else:
                output = torch.sigmoid(logits)
                output, labels = output.type(torch.FloatTensor), labels.type(torch.FloatTensor)
                loss = criterion(output, labels)
                output = output.detach().cpu().numpy()
                hypothesis = np.digitize(output, [0.5])
            
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm) # clipping gradient for avoiding exploding gradients
            optimizer.step()
            scheduler.step()
            expected_predictions = labels.detach().cpu().numpy()

            input_metric['y_true'] = expected_predictions
            input_metric['y_pred'] = hypothesis

            training_metric = metric(**input_metric)

            if idx_train_batch % validation_steps == 0:
                model.eval()
                print(f"Training - Epoch {idx_epoch+1}/{num_train_epochs}, Batch: {idx_train_batch+1}/{num_training_batches}, Loss: {loss:.3f} Metric:{training_metric:.3f}")
                """ Step 3.2: evaluating """
                for (idx_validation_batch, validation_batch) in enumerate(validation_dataloader):
                    # 0: query_embedding, 1: product_embedding, 2: labels
                    labels = validation_batch[2].to(device)
                    with torch.no_grad():
                        logits = model(validation_batch[0].to(device), validation_batch[1].to(device))
                    if model.num_labels > 2:
                        loss = criterion(logits.view(-1, model.num_labels), labels.view(-1))
                        logits = logits.detach().cpu().numpy()
                        hypothesis = np.argmax(logits, axis=1)
                    else:
                        output = torch.sigmoid(logits)
                        output, labels = output.type(torch.FloatTensor), labels.type(torch.FloatTensor)
                        loss = criterion(output, labels)
                        output = output.detach().cpu().numpy()
                        hypothesis = np.digitize(output, [0.5])
                    expected_predictions = labels.detach().cpu().numpy()
                    input_metric['y_true'] = expected_predictions
                    input_metric['y_pred'] = hypothesis
                    validation_metric[idx_validation_batch] = metric(**input_metric)
                    validation_loss[idx_validation_batch] = loss
                current_validation_metric = np.mean(validation_metric)

                print(f"Validation - Epoch {idx_epoch+1}/{num_train_epochs}, Batch: {idx_train_batch+1}/{num_training_batches}, Loss: {np.mean(validation_loss):.3f}, Metric:{np.mean(validation_metric):.3f}")
                
                if current_validation_metric > best_metric_value:
                    best_metric_value = current_validation_metric
                    best_model = model
                    """ Step 4: store model """
                    pathlib.Path(path_model).mkdir(parents=True, exist_ok=True)
                    torch.save(best_model.state_dict(), os.path.join(path_model, "pytorch_model.bin"))

if __name__ == '__main__':
    dict_labels_type = dict()
    dict_labels_type['esci_labels'] = {
        'E' : 0,
        'S' : 1,
        'C' : 2,
        'I' : 3,
    }

    df = pd.read_csv('sampled_data_v2.csv')
    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]

    df_train = df_train[['query_id','product_id','esci_label']]
    df_test = df_test[['query_id','product_id','esci_label']]


    num_examples = len(df_train)
    array_queries = np.zeros((num_examples, 768))
    array_products = np.zeros((num_examples, 768))
    dict_products = np.load('bert_representations/dict_products_train.npy', allow_pickle=True)
    dict_queries = np.load('bert_representations/dict_query_train.npy', allow_pickle=True)
    for i in tqdm(range(num_examples)):
            array_queries[i] = dict_queries[()][df_train.iloc[i]['query_id']]
            array_products[i] = dict_products[()][df_train.iloc[i]['product_id']]
            
    np.save('array_queries_train.npy', array_queries)
    np.save('array_products_train.npy', array_products)

    num_examples = len(df_test)
    array_queries = np.zeros((num_examples, 768))
    array_products = np.zeros((num_examples, 768))
    dict_products = np.load('bert_representations/dict_products_test.npy', allow_pickle=True)
    dict_queries = np.load('bert_representations/dict_query_test.npy', allow_pickle=True)
    for i in tqdm(range(num_examples)):
            array_queries[i] = dict_queries[()][df_test.iloc[i]['query_id']]
            array_products[i] = dict_products[()][df_test.iloc[i]['product_id']]
            
    np.save('array_queries_test.npy', array_queries)
    np.save('array_products_test.npy', array_products)

    df_train['class_id'] = df_train['esci_label'].apply(lambda label: dict_labels_type['esci_labels'][label])
    np.save('array_labels_train.npy', df_train['class_id'].to_numpy())

    df_test['class_id'] = df_test['esci_label'].apply(lambda label: dict_labels_type['esci_labels'][label])
    np.save('array_labels_test.npy', df_test['class_id'].to_numpy())



    num_labels = 4
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ 1. Load data """
    query_array = np.load('array_queries_train.npy')
    products_array = np.load('array_products_train.npy')
    labels_array = np.load('array_labels_train.npy')

    n_examples = labels_array.shape[0]
    dev_size = 1000 / n_examples
    ids_train, ids_dev = train_test_split(
        range(0, n_examples), 
        test_size=dev_size, 
        random_state=42,
    )

    query_array_train, products_array_train, labels_array_train = query_array[ids_train], products_array[ids_train], labels_array[ids_train]
    query_array_dev, products_array_dev, labels_array_dev = query_array[ids_dev], products_array[ids_dev], labels_array[ids_dev]

    train_dataset = generate_dataset(
        query_array_train,
        products_array_train,
        labels_array_train,
    )

    dev_dataset = generate_dataset(
        query_array_dev,
        products_array_dev,
        labels_array_dev,
    )

    """ 2. Prepare model """
    model = QueryProductClassifier(
        num_labels=4,
    )
    train(
        model, 
        train_dataset, 
        dev_dataset, 
        './models', 
        device=train_device, 
        batch_size=16, 
        weight_decay=0.01, 
        num_train_epochs=50, 
        lr=5e-5, 
        eps=1e-8, 
        num_warmup_steps=0, 
        max_grad_norm=1,
        validation_steps=100,
        random_seed=42
    )



    query_array_test = np.load('array_queries_test.npy')
    products_array_test = np.load('array_products_test.npy')
    labels_array_test = np.load('array_labels_test.npy')

    test_dataset = generate_dataset(
        query_array_test,
        products_array_test,
        labels_array_test,
    )

    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    """ 2. Prepare model """
    model = QueryProductClassifier(num_labels=4)
    model.load_state_dict(torch.load("models/pytorch_model.bin"),)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    """ 3. Generate hypothesis"""
    a_hypothesis = np.array([])
    model.eval()
    with torch.no_grad():
        for test_batch in tqdm(data_loader):
            # 0: query_encoding, 1: product_encoding
            with torch.no_grad():
                logits = model(test_batch[0].to(device), test_batch[1].to(device))
                logits = logits.detach().cpu().numpy()
                hypothesis = np.argmax(logits, axis=1)
                a_hypothesis = np.concatenate([
                    a_hypothesis,
                    hypothesis,
                ])
    a_hypothesis = a_hypothesis.astype(int)

    """ 4. Prepare hypothesis file """
    df_test = df[df['split']=='test']
    class_id2esci_label = {
            0 : 'E',
            1 : 'S',
            2 : 'C',
            3 : 'I',
        }
    labels = [class_id2esci_label[int(hyp)] for hyp in a_hypothesis ]

    print(len(df_test['example_id'].to_list()),len(labels),len(df_test['esci_label'].to_list()))

    df_hypothesis = pd.DataFrame({
        'example_id' : df_test['example_id'].to_list(),
        'esci_label' : labels,
        'gold_label' : df_test['esci_label'].to_list()
    })
    df_hypothesis[['example_id', 'esci_label']].to_csv(
        './hypothesis/task_2_esci_classifier_model.csv',
        index=False,
        sep=',',
    )
    macro_f1 = f1_score(
        df_hypothesis['gold_label'], 
        df_hypothesis['esci_label'], 
        average='macro',
    )
    micro_f1 = f1_score(
        df_hypothesis['gold_label'], 
        df_hypothesis['esci_label'], 
        average='micro',
        )
    print("macro\tmicro")
    print(f"{macro_f1:.3f}\t{micro_f1:.3f}")







