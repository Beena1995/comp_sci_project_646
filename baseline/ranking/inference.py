# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
#  
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import argparse
import numpy as np
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from tqdm import tqdm


def main():


    """ 0. Init variables """
    col_query = "query"
    col_query_id = "query_id"
    col_product_id = "product_id" 
    col_product_title = "product_title"
    col_product_locale = "product_locale"
    col_small_version = "small_version"
    col_split = "split"
    col_scores = "scores"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ 1. Load data """    
   

    df_examples_products = pd.read_csv('./sampled_data_expandedquery_v2.csv')
    features_query = df_examples_products[col_query].to_list()
    features_product = df_examples_products[col_product_title].to_list()
    n_examples = len(features_query)
    scores = np.zeros(n_examples)

    """ 2. Prepare Cross-encoder model """
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    """ 3. Generate hypothesis """
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, n_examples, args.batch_size)):
            j = min(i + args.batch_size, n_examples)
            features_query_ = features_query[i:j]
            features_product_ = features_product[i:j]
            features = tokenizer(features_query_, features_product_,  padding=True, truncation=True, return_tensors="pt").to(device)
            scores[i:j] = np.squeeze(model(**features).logits.cpu().detach().numpy())
            i = j
""" 4. Prepare hypothesis file """   
df_hypothesis = pd.DataFrame({
    col_query_id : df_examples_products[col_query_id].to_list(),
    col_product_id : df_examples_products[col_product_id].to_list(),
    col_scores : scores,
})
df_hypothesis = df_hypothesis.sort_values(by=[col_query_id, col_scores], ascending=False)
df_hypothesis[[col_query_id, col_product_id]].to_csv(
    args.hypothesis_path_file,
    index=False,
    sep=',',
)


if __name__ == "__main__": 
    main()

