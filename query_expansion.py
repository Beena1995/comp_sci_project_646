import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from rank_bm25 import *
import numpy as np
import re
from math import log, sqrt
from collections import Counter
import swifter
from rocchio import *

nltk.download('stopwords')

from nltk.corpus import stopwords

def bm25_order(df_products):
    tokenized_products = [product.split(" ") for product in df_products['product_title'].to_list()]
    bm25 = BM25Okapi(tokenized_products) 

def get_top_n(x, bm25):
    tokenized_query = x['query'].split(" ")
    # print(tokenized_query)
    rank_lst = bm25.get_top_n(tokenized_query, df_products['product_title'].to_list(), n=50)
    rank_ids = [df_products.loc[df_products['product_title']==product,'product_id'].values[0] for product in rank_lst]
    return rank_ids

def query_expansion(x, prod_dict):
  print(x['query_id'])
  expanded_query = findNewQuery(x['query'], 10, x['rank_list'], invertedIndex, prod_dict)
  # print(expanded_query)
  return expanded_query

def main():
    df = pd.read_csv('/test/test_data.csv')
    df_products = df[['product_id','product_title']].drop_duplicates()
    df_products['product_title'] = df_products['product_title'].map(lambda x:x.lower())
    df_products['product_title'] = df_products['product_title'].map(lambda x:re.sub(r'[^\w\s]',' ',x))
    bm25 =  bm25_order(df_products)
    df_query = df[['query', 'query_id']].drop_duplicates()   
    df_query['rank_list'] = df_query.swifter.apply(lambda x: get_top_n(x, bm25), axis = 1)
    prod_dict = df_products.set_index('product_id').T.to_dict('list')
    df_query['expanded_query'] = df_query.swifter.apply(lambda x: query_expansion(x), axis =1)



if __name__ == '__main__':
    main()