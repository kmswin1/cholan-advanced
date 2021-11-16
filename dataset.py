from os import path
import re
import random
from collections import OrderedDict
from pprint import pprint
import pickle as pkl
import json

# pwd: '/home/juyeon/github/cholan-advanced'

doc2type = pkl.load(open('./data/dca/doc2type.pkl', 'rb'))
entity2type = pkl.load(open('./data/dca/entity2type.pkl', 'rb'))
mtype2id = {'PER':0, 'ORG':1, 'GPE':2, 'UNK':3}

def judge(s1, s2):
    if s1==s2:
        return True
    if s2.replace('. ', ' ').replace('.', ' ') == s1:
        return True
    if s2.replace('-', ' ') == s1:
        return True
    return False


def read_csv_file(path):
    data = {}
    flag = 0

    if path.find('aida')>=0:
        flag = 1
    else:
        types = json.load(open('./data/dca/type/'+path.split('/')[-1].split('.')[0]+'.json', 'rb'))
    docid = '0'
    with open(path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            comps = line.strip().split('\t')
            # doc_name = comps[0]
            doc_name = comps[0] + ' ' + comps[1]
            mention = comps[2]
            mtype = [0,0,0,0]
            if flag == 1:
                doc = ''
                for c in doc_name:
                    try:
                        doc += str(int(c))
                    except:
                        break
                if not doc==docid:
                    docid = doc
                    p = 0
                    tt = doc2type[docid]
                try:

                    while not judge(mention.lower(), tt[p][0].lower()):
                        p += 1
                    mtype[mtype2id[tt[p][1]]] = 1

                except:
                    print(docid+mention)

                    mtype[mtype2id['UNK']] = 1
            else:
                if path.find('wikipedia')<0:
                    tt = types['sample_%d'%i]['pred'] + types['sample_%d'%i]['overlap']
                    for t in tt:
                        if t == 'MISC':
                            t = 'UNK'
                        if t == 'LOC':
                            t = 'GPE'
                        mtype[mtype2id[t]] = 1
                else:
                    mtype[mtype2id['UNK']] = 1
            
            lctx = comps[3]
            rctx = comps[4]
            
            if comps[6] != 'EMPTYCAND':
                cands = [c.split(',') for c in comps[6:-2]]
                cands = [[','.join(c[2:]).replace('"', '%22').replace(' ', '_'), float(c[1])] for c in cands]
            else:
                cands = []
            
            gold = comps[-1].split(',')
            if gold[0] == '-1':
                gold = (','.join(gold[2:]).replace('"', '%22').replace(' ', '_'), 1e-5, -1)
            else: 
                gold = (','.join(gold[3:]).replace('"', '%22').replace(' ', '_'), 1e-5, -1)
            if doc_name not in data:
                    data[doc_name] = []
            data[doc_name].append({'mention': mention,
                                    'mtype': mtype,
                                    'context': (lctx, rctx),
                                    'candidates': cands,
                                    'gold': gold})
    return data

def read_conll_file(data, path):
    
    conll = {}
    with open(path, 'r', encoding='utf8') as f:
        if path.find('aida')>=0:
            flag = 1
        else:
            flag = 0
        cur_sent = None
        cur_doc = None
        for line in f:
            line = line.strip()
            if line.startswith('-DOCSTART-'):
                docname = line.split()[1][1:]
                conll[docname] = {'sentences': [], 'mentions': []}
                cur_doc = conll[docname]
                cur_sent = []
            else:
                if line == '':
                    cur_doc['sentences'].append(cur_sent)
                    cur_sent = []
                else:
                    comps = line.split('\t')
                    tok = comps[0]
                    cur_sent.append(tok)

                    if len(comps) >= 6:
                        mention = comps[2]
                        bi = comps[1]
                        wikilink = comps[4]
                        
                        if flag == 1:
                            wiki_title = comps[4][len("http://en.wikipedia.org/wiki/"):].replace('_', ' ')

                        else:
                            wiki_title = comps[4][len("en.wikipedia.org/wiki/"):].replace('_', ' ')
                            
                        if bi == 'I':
                            cur_doc['mentions'][-1]['end'] += 1
                            
                        else:
                            new_ment = {'sent_id': len(cur_doc['sentences']),
                                        'mention': mention,
                                        'start': len(cur_sent) - 1,
                                        'end': len(cur_sent),
                                        'wikilink': wikilink,
                                        'wiki_title': wiki_title}
                            cur_doc['mentions'].append(new_ment)
                        
                                
        # merge with data
        rmpunc = re.compile('[\W]+')
        for doc_name, content in data.items():
            conll_doc = conll[doc_name.split()[0]]
            content[0]['conll_doc'] = conll_doc

            cur_conll_m_id = 0
            for m in content:
                mention = m['mention']
                # flag = 0

                while True:
                    cur_conll_m = conll_doc['mentions'][cur_conll_m_id]
                    cur_conll_mention = ' '.join(conll_doc['sentences'][cur_conll_m['sent_id']][cur_conll_m['start']:cur_conll_m['end']])
                    if rmpunc.sub('', cur_conll_mention.lower()) == rmpunc.sub('', mention.lower()):
                        m['conll_m'] = cur_conll_m

                        # if flag == 1:
                        #     print(cur_conll_m_id, cur_conll_mention, mention)
                        # flag = 0

                        cur_conll_m_id += 1
                        break
                    else:
                        # print(cur_conll_m_id, cur_conll_mention, mention)
                        # flag = 1
                        cur_conll_m_id += 1


class CoNLLDataset:
    """
    reading dataset from CoNLL dataset, extracted by https://github.com/dalab/deep-ed/
    """

    def __init__(self, path, conll_path):
        # path = '/home/juyeon/github/cholan-advanced/data/dca/generated'
        print('load csv')
        self.train = read_csv_file(path + '/aida_train.csv')
        self.testB = read_csv_file(path + '/aida_testB.csv')
        self.msnbc = read_csv_file(path + '/wned-msnbc.csv')
        self.ace2004 = read_csv_file(path + '/wned-ace2004.csv')
        self.aquaint = read_csv_file(path + '/wned-aquaint.csv')
        self.wikipedia = read_csv_file(path + '/wned-wikipedia.csv')
        self.wikipedia.pop('Jiří_Třanovský Jiří_Třanovský', None)

        # path = '/home/juyeon/github/cholan-advanced/data/conll'
        print('load conll')
        read_conll_file(self.train, conll_path + '/aida_train.txt')
        read_conll_file(self.testB, conll_path + '/aida_testb.txt')
        read_conll_file(self.msnbc, conll_path + '/msnbc.conll')
        read_conll_file(self.ace2004, conll_path + '/ace2004.conll')
        read_conll_file(self.aquaint, conll_path + '/aquaint.conll')
        read_conll_file(self.wikipedia, conll_path + '/wikipedia.conll')