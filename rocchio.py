import numpy as np
import math

ALPHA = 1
BETA = 0.75
GAMMA = 0.15

def generateInvertedIndex(prod_dict):
    invertedIndex = {}
    tokenDict = {}
    # doc_dict = open(f'{location}/parquet_data/products.txt')
    doc_dict = prod_dict
    for key, val in doc_dict.items():
        doc_id = key
        text = val[0]
        # print(val[0])
        doc_text = val[0].split()
        length = len(doc_text)
        tokenDict[doc_id] = length
        for word in text.split():
            if word not in invertedIndex.keys():
                docIDCount = {doc_id : 1}
                invertedIndex[word] = docIDCount
            elif doc_id in invertedIndex[word].keys():
                invertedIndex[word][doc_id] += 1
            else:
                docIDCount = {doc_id : 1}
                invertedIndex[word].update(docIDCount)
    return invertedIndex


def queryFrequency(query, invertedIndex):
    queryFreq = dict.fromkeys(invertedIndex.keys(), 0)
    # print(query)
    # print(query.split())
    for term in query.split():
      if term in queryFreq.keys():
            queryFreq[term] += 1
      else:
            queryFreq[term] = 1
    #print(queryFreq)
    return queryFreq

def calculateDocsCount(doc, docIndex, prod_dic)t:
    doc_dict = prod_dict
    # print("doc", doc)
    # print(doc, doc_dict[doc])
    text = doc_dict[doc][0]
    for term in text.split():
        if term in docIndex.keys():
            docIndex[term] += 1
        else:
            docIndex[term] = 1
    return docIndex

def findDocs(k, sortedBM25Score, invertedIndex, relevancy):
    relIndex = {}
    nonRelIndex = {}
    if relevancy == "Relevant":
        for i in range(0, k):
            # print(i, sortedBM25Score)
            doc = sortedBM25Score[i]
            relIndex = calculateDocsCount(doc, relIndex)
        for term in invertedIndex:
            if term not in relIndex.keys():
                relIndex[term] = 0
        return relIndex
    elif relevancy == "Non-Relevant":
        for i in range(k+1,len(sortedBM25Score)):
            doc = sortedBM25Score[i]
            nonRelIndex = calculateDocsCount(doc, nonRelIndex)
        for term in invertedIndex:
            if term not in nonRelIndex.keys():
                nonRelIndex[term] = 0   
        return nonRelIndex

def findRelDocMagnitude(docIndex):
    mag = 0
    for term in docIndex:
        mag += float(docIndex[term]**2)
        mag = float(sqrt(mag))
    return mag

def findNonRelDocMagnitude(docIndex):
    mag = 0
    for term in docIndex:
        mag += float(docIndex[term]**2)
    mag = float(sqrt(mag))
    return mag


def findRocchioScore(term, queryFreq, relDocMag, relIndex, nonRelMag, nonRelIndex):
    Q1 = ALPHA * queryFreq[term] 
    Q2 = (BETA/relDocMag) * relIndex[term]
    Q3 = (GAMMA/nonRelMag) * nonRelIndex[term]
    rocchioScore = ALPHA * queryFreq[term] + (BETA/relDocMag) * relIndex[term] - (GAMMA/nonRelMag) * nonRelIndex[term]
    return rocchioScore

def findNewQuery(query, k, sortedBM25Score, invertedIndex, prod_dict):
    queryFreq = queryFrequency(query, invertedIndex)
    relIndex = findDocs(k, sortedBM25Score, invertedIndex, "Relevant")
    relDocMag = findRelDocMagnitude(relIndex)
    nonRelIndex = findDocs(k, sortedBM25Score, invertedIndex, "Non-Relevant")
    nonRelMag = findNonRelDocMagnitude(nonRelIndex)
    updatedQuery = {}
    newQuery = query
    for term in invertedIndex:
        updatedQuery[term] = findRocchioScore(term, queryFreq, relDocMag, relIndex, nonRelMag, nonRelIndex)
    sortedUpdatedQuery = sorted(updatedQuery.items(), key=lambda x:x[1], reverse=True)
    if len(sortedUpdatedQuery)<5:
        loopRange = len(sortedUpdatedQuery)
    else:
        loopRange = 5
    for i in range(loopRange):
        term,frequency = sortedUpdatedQuery[i]
        #print("term, frequency", term, frequency)
        if term not in query:
            newQuery +=  " "
            newQuery +=  term
    return newQuery

def getReduceIndex(query, invertedIndex):
    query_term_freq = {}
    query_term_list = query.split()
    reduced_inverted_index = {}
    for term in query.split():
        if term in query_term_freq.keys():
            query_term_freq[term] += 1
        else:
            query_term_freq[term] = 1

    for term in query_term_freq:
        if term in  invertedIndex:
            reduced_inverted_index.update({term:invertedIndex[term]})
        else:
            reduced_inverted_index.update({term:{}})
    return reduced_inverted_index

