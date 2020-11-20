from __future__ import print_function
import os
import sys
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_ym import Dictionary
import gensim.models.word2vec as word2vec

def create_dictionary(dataroot):
    dictionary = Dictionary()
    questions = []
    files = [
        'v2_OpenEnded_mscoco_train2014_questions.json',
        'v2_OpenEnded_mscoco_val2014_questions.json',
        'v2_OpenEnded_mscoco_test2015_questions.json',
        'v2_OpenEnded_mscoco_test-dev2015_questions.json'
    ]
    for path in files:
        question_path = os.path.join(dataroot, path)
        qs = json.load(open(question_path))['questions']
        for q in qs:
            dictionary.tokenize(q['question'], True)
    return dictionary


def create_dictionary_chest(dataroot):
    dictionary = Dictionary()
    questions = []
    files = [
        'train_questions.json',
        'val_questions.json',
        'test_questions.json'
    ]
    for path in files:
        question_path = os.path.join(dataroot, path)
        qs = json.load(open(question_path))['questions']
        for q in qs:
            dictionary.tokenize(q['question'], True)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=str, default="./data/cache_1014")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # d = create_dictionary('data')
    # d.dump_to_file('data/dictionary.pkl')
    # d = Dictionary.load_from_file('data/dictionary.pkl')
    # d = create_dictionary_chest('/home/ymeng/store_ym/nlp_422/unilm/unilm-v1/data/med/')
    # d.dump_to_file('data/dictionary_chest_0807.pkl')
    cache = args.cache
    d = create_dictionary_chest(cache)

    d.dump_to_file('{}/dictionary_chest.pkl'.format(cache))

    # d = Dictionary.load_from_file('data/dictionary_chest_0807.pkl')
    emb_dim = 50 
    glove_file = 'data/glove/glove.6B.%dd.txt'% emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    model = word2vec.Word2Vec.load("/home/ymeng/store_ym/medical_report_vqa/data/emb_model/gensim%d.model"%emb_dim)
    weights_gensim = np.array([model[w] for w in d.idx2word])
    np.save('%s/gensim_init_%dd_chest.npy' % (cache, emb_dim), weights_gensim)
    np.save('%s/glove6b_init_%dd_chest.npy' % (cache, emb_dim), weights)