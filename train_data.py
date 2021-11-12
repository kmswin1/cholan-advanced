import os
import torch
import pandas as pd
import pickle
from collections import defaultdict
import numpy as np
import time
import sys
import string
import requests, json
import collections

def get_wikidata_id(wikipedia_title):
    wikidata_Qids_List = []

    for i, title in enumerate(wikipedia_title):
        try:
            url = 'https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&ppprop=wikibase_item&titles=%s&format=json' % str(title)
            response = requests.get(url).json()['query']['pages']
            df = pd.io.json.json_normalize(response)
            df.columns = df.columns.map(lambda x: x.split(".")[-1])
            wikidata_id = df.get(key='wikibase_item').values

            wikidata_Qids_List.append(wikidata_id[0])
        except:
            print("Invalid Title - ", title)
            wikidata_Qids_List.append("NA")

    return wikidata_Qids_List


def readfile1(filename):
    '''
    read file
    '''
    f = open(filename)
    sentence_data = []
    entity_data = []
    wiki_title_data = []
    sentence = []
    entity = []
    wiki_title = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                entity_dict = collections.OrderedDict.fromkeys(entity)
                entity = list(entity_dict.keys())
                wiki_title_dict = collections.OrderedDict.fromkeys(wiki_title)
                wiki_title = list(wiki_title_dict.keys())
                #entity = list(set(entity))
                #wiki_title = list(set(wiki_title))

                sentence_data.append(sentence)
                entity_data.append(entity)
                wiki_title_data.append(wiki_title)
                sentence = []
                entity = []
                wiki_title = []
            continue
        splits = line.split('\t')
        sentence.append(splits[0].strip('\n'))
        if len(splits) == 7 and (splits[1] == 'B'):
            entity.append(splits[2])
            wiki_title.append(splits[4][len("http://en.wikipedia.org/wiki/"):].replace('_', ' '))

    if len(sentence) > 0:
        #entity_dict = collections.OrderedDict.fromkeys(entity)
        #entity = list(entity_dict.keys())
        #wiki_title_dict = collections.OrderedDict.fromkeys(wiki_title)
        #wiki_title = list(wiki_title_dict.keys())
        sentence_data.append(sentence)
        entity_data.append(entity)
        wiki_title_data.append(wiki_title)
        sentence = []
        entity = []
        wiki_title = []
    return create_df(sentence_data, entity_data, wiki_title_data)


def create_df(sentence_list, entity_list, wiki_title_list):
    df_file = pd.DataFrame()
    total_entity_count = 0
    total_wiki_entity_count = 0
    for i in range(0, len(sentence_list)):
        #for j in range(0, len(sentence_list)-1):
        sentence = ' '.join(token for token in sentence_list[i])
        wikidata_id = get_wikidata_id(wiki_title_list[i])
        total_entity_count += len(wiki_title_list[i])
        if len(wikidata_id) == len(wiki_title_list[i]):
            total_wiki_entity_count += len(wikidata_id)
            # Print the count of entity aligned with the qids
            print("Sentence - ", i + 1, "\tActual_Entities - ", len(wiki_title_list[i]), "\tAligned_Entities - ", len(wikidata_id), "\tTotal_Entities - ", total_entity_count, "\tTotal_Aligned_Entities - ", total_wiki_entity_count)
            entity = ' '.join(entity + ' EntityMentionSEP' for entity in entity_list[i])
            wiki_title = ' '.join(wiki_title + ' WikiLabelSEP' for wiki_title in wiki_title_list[i])
            uri = ' '.join(qid for qid in wikidata_id)
            d = {'Entity': entity, 'Sentence': sentence.replace(","," "), 'Uri': uri, 'WikiTitle': wiki_title}
            df_file = df_file.append(d, ignore_index=True)

    df_file = df_file.fillna('NIL_ENT')
    return df_file

if __name__ == '__main__':
    in_file = "/home/juyeon/github/cholan-advanced/data/conll/"
    out_file = "/home/juyeon/github/cholan-advanced/data/conll/generated"



    aida_train = readfile1(in_file + '/aida_train.txt')
    aida_train.to_csv(out_file + 'aida_train.csv' , sep='\t', encoding='utf-8', index=False)
    
    aida_test = readfile1(in_file + '/testa_testb_aggregate_original')
    aida_test.to_csv(out_file + 'aida_test.csv' , sep='\t', encoding='utf-8', index=False)
    
    msnbc = readfile1(in_file + '/msnbc.conll')
    msnbc.to_csv(out_file + 'msnbc.csv' , sep='\t', encoding='utf-8', index=False)
    
    ace2004 = readfile1(in_file + '/ace2004.conll')
    ace2004.to_csv(out_file + 'ace2004.csv' , sep='\t', encoding='utf-8', index=False)
    
    aquaint = readfile1(in_file + '/aquaint.conll')
    aquaint.to_csv(out_file + 'aquaint.csv' , sep='\t', encoding='utf-8', index=False)
    
    wikipedia = readfile1(in_file + '/wikipedia.conll')
    wikipedia.to_csv(out_file + 'wikipedia.csv' , sep='\t', encoding='utf-8', index=False)
    
    #wikipedia_title = ["Uzbekistan national football team", "Germany"]
    #wikidata_id = get_wikidata_id(wikipedia_title)
    #print(wikipedia_title, " - ", wikidata_id)
    
    