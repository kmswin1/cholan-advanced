import dataset as D
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
s = D.CoNLLDataset('/data/project/research/cholan-advanced/data/dca/generated','/data/project/research/cholan-advanced/data/conll/')

def evaluate():
    model = SentenceTransformer('mention_sent')
    candidates = {}
    i = 0
    for line in s.train:
        for meta in s.train[line]:
            for candidate in meta['candidates']:
                candidates[candidate[0]] = i
                i += 1
    for line in s.testB:
        for meta in s.testB[line]:
            for candidate in meta['candidates']:
                candidates[candidate[0]] = i
                i += 1
    for line in s.aquaint:
        for meta in s.aquaint[line]:
            for candidate in meta['candidates']:
                candidates[candidate[0]] = i
                i += 1
    for line in s.ace2004:
        for meta in s.ace2004[line]:
            for candidate in meta['candidates']:
                candidates[candidate[0]] = i
                i += 1
    for line in s.wikipedia:
        for meta in s.wikipedia[line]:
            for candidate in meta['candidates']:
                candidates[candidate[0]] = i
                i += 1
    tot = 0
    cnt = 0
    for line in s.testB:
        for meta in s.testB[line]:
            mention_embedding = model.encode(meta['mention'])
            sentence_embedding = model.encode(meta['context'][1])
            k = mention_embedding * sentence_embedding
            temp = {}
            for candidate in candidates.keys():
                temp[candidate] = np.dot(k, model.encode(candidate))
            q = sorted(temp.items(), key=lambda x: x[1], reverse=True)[:30]
            tot += 1
            t = [ent[0] for ent in q]
            if meta['gold'][0] in t:
                cnt += 1
    print (cnt/tot)

    tot = 0
    cnt = 0
    for line in s.aquaint:
        for meta in s.aquaint[line]:
            mention_embedding = model.encode(meta['mention'])
            sentence_embedding = model.encode(meta['context'][1])
            k = mention_embedding * sentence_embedding
            temp = {}
            for candidate in candidates.keys():
                temp[candidate] = np.dot(k, model.encode(candidate))
            q = sorted(temp.items(), key=lambda x: x[1], reverse=True)[:30]
            tot += 1
            t = [ent[0] for ent in q]
            if meta['gold'][0] in t:
                cnt += 1
    print(cnt / tot)

    tot = 0
    cnt = 0
    for line in s.msnbc:
        for meta in s.msnbc[line]:
            mention_embedding = model.encode(meta['mention'])
            sentence_embedding = model.encode(meta['context'][1])
            k = mention_embedding * sentence_embedding
            temp = {}
            for candidate in candidates.keys():
                temp[candidate] = np.dot(k, model.encode(candidate))
            q = sorted(temp.items(), key=lambda x: x[1], reverse=True)[:30]
            tot += 1
            t = [ent[0] for ent in q]
            if meta['gold'][0] in t:
                cnt += 1
    print(cnt / tot)

    tot = 0
    cnt = 0
    for line in s.ace2004:
        for meta in s.ace2004[line]:
            mention_embedding = model.encode(meta['mention'])
            sentence_embedding = model.encode(meta['context'][1])
            k = mention_embedding * sentence_embedding
            temp = {}
            for candidate in candidates.keys():
                temp[candidate] = np.dot(k, model.encode(candidate))
            q = sorted(temp.items(), key=lambda x: x[1], reverse=True)[:30]
            tot += 1
            t = [ent[0] for ent in q]
            if meta['gold'][0] in t:
                cnt += 1
    print(cnt / tot)