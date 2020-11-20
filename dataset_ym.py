from __future__ import print_function
import os
import json
import pickle as cPickle
import numpy as np
import utils

import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms
from PIL import Image

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property 
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry

def _load_dataset(cache, name, img_id2val, img_name_list):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    # question_path = os.path.join(
    #     dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    # questions = sorted(json.load(open(question_path))['questions'],
    #                    key=lambda x: x['question_id'])
 
    question_path = os.path.join(cache, "%s_questions.json"%name)
    questions_list = json.load(open(question_path))['questions']
    questions = sorted(questions_list,
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(cache, '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])
    # print(answers)

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        # print("img_id",img_id2val[img_id])
        if img_id in img_id2val.keys():
            # print(img_id2val[img_id], " id: ", img_id)
            if img_id in img_name_list.keys():
                entries.append(_create_entry(img_id2val[img_id], question, answer))

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, train_target, dataroot='data', cache='data/cache'):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', "test"]
        ans2label_path = os.path.join(cache, 'trainval_ans2label.pkl')
        # label2ans_path = os.path.join(cache, 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        # print("total answers:",len(self.ans2label ))/
        # self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        # print(self.num_ans_candidates)
        self.dictionary = dictionary
        img_name_list = {}
        with open("./data/new_name.txt","r") as f:
            for line in f:
                # label = line.split('_')[0]
                # img_name_list[label] = line.strip()
                img_name_list[line.strip()] = line.strip()
        # self.img_id2idx = cPickle.load(
        #     open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
        # print('loading features from h5 file')
        # h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        # with h5py.File(h5_path, 'r') as hf:
            # self.featuress = np.array(hf.get('image_features'))
        #     self.spatials = np.array(hf.get('spatial_features'))
        self.img_name_list = img_name_list
        img_id2idx = {}
        

        sp = np.concatenate(([[0.]],[[0.]],[[1.]], [[1.]],[[1.]],[[1.]]), axis=1)
        
        # features_list = []
        # spatials_list = []      
        ft = "data/cache2/features.json"

        with open(ft,"r") as f:
            features_tmp = json.load(f)
            for f_id, name_feature in enumerate(features_tmp.keys()):
                # img_id2idx[name_feature[0:-4]] = f_id
                # img_id2idx[name_feature.split("_")[0]] = f_id
                img_id2idx[name_feature] = f_id
                # print(name_feature.split("_")[0])
                # spatials_list.append(sp) 
                # features_list.append(features_tmp[name_feature])
        '''
        feature_np = np.array(features_list)
        spatials_np = np.array(spatials_list)
        # print(spatials_np.shape)
        
        self.features = feature_np.reshape([feature_np.shape[0],1,feature_np.shape[1]])
        self.features = np.concatenate([self.features,self.features],axis = 1).astype("float32")
     
        self.spatials = spatials_np
        self.spatials = np.concatenate([self.spatials,self.spatials],axis= 1).astype("float32")
        self.v_dim = self.features.size(2)
        self.s_dim = self.spatials.size(2)
        print(self.features.shape)
        '''
        self.img_id2idx = img_id2idx
        self.entries = _load_dataset(cache, name, self.img_id2idx, self.img_name_list)
        self.tokenize()
        self.tensorize()
        
        # self.transform = transforms.Compose([transforms.Resize((512, 512)),
        #                                          transforms.ToTensor()])
        # AttributeError: 'module' object has no attribute 'Resize' 
        self.v_dim = 512
        self.s_dim = self.v_dim 
        

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            try:
                tokens = self.dictionary.tokenize(entry['question'], False)
                if entry['question'] not in ['cardiac','silhouette' ]:
                    import pdb; pdb.set_trace()
            except Exception as e:
                print(entry['question'])
                # import pdb; pdb.set_trace()

            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        # self.features = torch.from_numpy(self.features)
        # self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        image_path =  os.path.join("./data/images/",str([entry['image_id']]))
        image = cv2.imread(image_path)
        try:
            img = np.array(cv2.resize(image, (512, 512), interpolation = cv2.INTER_AREA),dtype=np.float32)
            img = np.transpose(img,(2,0,1))
            img = img[:,:,0]
        except:
            # print("loading fail:",self.img_name_list[entry['image_id']])
            img = np.zeros((512,512))
        # image = Image.open(image_path).convert('RGB')
        # img = self.transform(image).float()
        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        # question_txt = entry["question"]
        spatials = target
        question_id = entry['question_id']
        if labels is not None:
            target.scatter_(0, labels, scores) #len = len(labels). scores are the score for each answer. it's a list of scores of words
        # i = entry['image_id']
        # print(question.size(), img.shape)
        # return img.astype(np.float32), spatials, question, target #(117,14,117)
        return img.astype(np.float32), question_id, question, target#(117,14,117)

 
    def __len__(self):
        return len(self.entries)


class dicti:
    def __init__(self, ntoken):
        self.ntoken = ntoken


class VQAFeatureDataset_debug(Dataset):
    def __init__(self, name,  dataroot='data', cache='cache'):
        super(VQAFeatureDataset_debug, self).__init__()
        # ans2label_path = os.path.join(dataroot, cache, 'trainval_ans2label.pkl')
        # # label2ans_path = os.path.join(dataroot,  cache, 'trainval_label2ans.pkl')
        # self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.dictionary = dicti(100)
        self.num_hid = 1000
        self.v_dim  = 512
        self.num_ans_candidates = 100



    def __getitem__(self, index):
        img = torch.zeros(512,512)
        question_id = (torch.rand(1)*100).int().long() 
        question = torch.rand(14)
        target = torch.zeros(self.num_ans_candidates).float()

        return img #, question_id, question, target

    def __len__(self):
        return 10000







