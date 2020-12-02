
import json
import pickle 


import os
import json
import pickle
import numpy as np
# import utils
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

import argparse
import collections
from base_model import *
# import en_core_web_sm
# nlp = en_core_web_sm.load()

transform1 = transforms.Compose([
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(), 
    ]
)

import functools

# @functools.lru_cache(maxsize = 130000)
# def my_image_read(path):
#     return Image.open(path).convert('RGB')


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
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
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
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset_test(questions, img_id2val):
    entries = []
    answer = {'labels':"", 'scores':0}
    for question in questions:
        img_id = question['image_id']
        if img_id not in img_id2val:
            length = len(img_id2val)
            img_id2val[img_id] = length

        entries.append(_create_entry(img_id2val[img_id], question, answer))
    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, questions, dictionary, dataroot='data', cache='data/cache'):
        super(VQAFeatureDataset, self).__init__()
        ans2label_path = os.path.join(cache, 'trainval_ans2label.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.dictionary = dictionary
        img_name_list = {}
        with open("%s/new_name.txt" % dataroot,"r") as f:
            for line in f:
                img_name_list[line.strip()] = line.strip()
        self.img_name_list = img_name_list
        img_id2idx = {}
        # sp = np.concatenate(([[0.]],[[0.]],[[1.]], [[1.]],[[1.]],[[1.]]), axis=1)
        ft = "%s/cache2/features.json" % dataroot
        with open(ft,"r") as f:
            features_tmp = json.load(f)
            for f_id, name_feature in enumerate(features_tmp.keys()):
                img_id2idx[name_feature] = f_id
        self.img_id2idx = img_id2idx
        self.entries = _load_dataset_test(questions, self.img_id2idx)
        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=14):
        for entry in self.entries:
            try:
                tokens = self.dictionary.tokenize(entry['question'], False)
            except:
                import pdb; pdb.set_trace()
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            # utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            entry['answer']['labels'] = None
            entry['answer']['scores'] = None


    def __getitem__(self, index):
        entry = self.entries[index]
        path = '/home/ymeng/store_ym/data/IU_X_ray/NLMCXR_png'
        image_path =  os.path.join(path, str(entry['image_id']))
        image = cv2.imread(image_path)
        img = np.array(cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA),dtype=np.float32)
        img = np.transpose(img,(2,0,1))
        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        question_id = entry['question_id']
        return img.astype(np.float32), question_id, question, target

    def __len__(self):
        return len(self.entries)



class VQA():
    def __init__(self, lemmatized_sent_list, pairs_qa_list):
        self.lemmatized_sent_list = lemmatized_sent_list
        self.pairs_qa_list = pairs_qa_list
        self.question_str2idx = {}
        self.questions_lemma = []
        self.questions = []
        self.answer_str2idx = {}
        self.answers_lemma = []
        self.answers = []
        self.entries = []

def evaluate(model, test_dset, dataloader, cache):
    num_data = 0
    ans2label_path = os.path.join(cache, 'trainval_ans2label.pkl')
    label2ans_path = os.path.join(cache, 'trainval_label2ans.pkl')
    ans2label = pickle.load(open(ans2label_path, 'rb'))
    label2ans = pickle.load(open(label2ans_path, 'rb'))
    entry = test_dset.entries

    logits_total = [] 
    for _, (v, b, q, a) in enumerate(iter(dataloader)):
        with torch.no_grad():
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            pred = model(v, b, q, None)
            logits = (pred > 0).float()    
            if _ == 0:
                questions_list = b.cpu().numpy()
                answers_list = logits.cpu().numpy()
            else:
                questions_list = np.concatenate([questions_list, b.cpu().numpy()])
                answers_list = np.concatenate([answers_list, logits.cpu().numpy()])

    return questions_list, answers_list

def get_keys_split():
    keys = json.load(open('data_ori/patients.json',"r"))
    Y = np.load(open('data_ori/Y.npy', 'rb')).astype(int)
    train_id =int(len(keys)* 0.7)
    val_id = int(len(keys)* 0.8)
    return keys[0:train_id], keys[train_id:val_id], Y[0:train_id,:], Y[train_id:val_id,:]

def get_question_id(threshold = 50):        
    Y = np.load(open('./data_ori/Y.npy', 'rb')).astype(int) # Y.shape = [len_samples, num_questions] 
    num_classes = Y.shape[-1]
    indices = np.sum(Y, axis=0) > threshold
    index_tag = np.array([i for i in range(Y.shape[-1])])
    index_picked = index_tag[indices].tolist()
    print("got {} questions by weights".format(len(index_picked)))
    return index_picked


def trans_sent_to_word_set(sent):
    doc = nlp(sent)
    word_set = set(list(w.lemma_ for w in doc))
    return word_set
def sent_lemmatization(sent):
    word_set = list(trans_sent_to_word_set(sent))
    return " ".join(word_set)

def generate_vqa_eval_data(vqa):
    keys_train, keys_val, Y_train,Y_val= get_keys_split()
    question_ind_list = get_question_id()
    questions_index_with_y = json.load(open('data_ori/questions_y_index.json',"r"))
    ques_new = [questions_index_with_y[ind] for ind in question_ind_list] 
    
    questions_json = []
    for ind in question_ind_list:
        Y_i = Y_train[:,ind]
        ind = int(ind) 
        question = questions_index_with_y[ind]
        if question in vqa.question_str2idx:
            question_index = vqa.question_str2idx[question]
        else:
            ques_lemma = sent_lemmatization(question)
            question_index = vqa.questions_lemma.index(ques_lemma)
        question_fresh = vqa.questions[question_index]
        for i,y in enumerate(Y_i):
            if y == 1:
                ques0 = {'image_id':keys_train[i], 'question': question_fresh, 'question_id': question_index}
                questions_json.append(ques0)
    return questions_json


def generate_vqa_eval_data_all_weights(vqa):
    keys_train, keys_val, Y_train,Y_val= get_keys_split()
    question_ind_list = get_question_id(threshold=80)
    questions_index_with_y = json.load(open('data_ori/questions_y_index.json',"r"))
    ques_new = [questions_index_with_y[ind] for ind in question_ind_list] 
    questions_json = []
    for ind in question_ind_list:
        Y_i = Y_train[:,ind]
        ind = int(ind) 
        question = questions_index_with_y[ind]
        if question in vqa.question_str2idx:
            question_index = vqa.question_str2idx[question]
        else:
            ques_lemma = sent_lemmatization(question)
            question_index = vqa.questions_lemma.index(ques_lemma)
        question_fresh = vqa.questions[question_index]
        for i,y in enumerate(Y_i):
            ques0 = {'image_id':keys_train[i], 'question': question_fresh, 'question_id': question_index}
            questions_json.append(ques0)
    return questions_json


def generate_ques_ans_list(model, train_dset, cache, question2answer_dict, questions_json):
    batch_size = 64
    question_str_list =  list(question2answer_dict.keys())
    train_loader = DataLoader(train_dset, batch_size, shuffle=False, num_workers=1)
    questions_list, answers_list = evaluate(model, train_dset, train_loader,cache)
    pick_idx = np.load("%s/pick_idx.npy"%cache) 
    
    generate_qa_pair_dict = collections.defaultdict(list)
    for i, qj in enumerate(questions_json):       
        
        ques_id = questions_list[i]
        ques = question_str_list[ques_id]
        answers_pool = question2answer_dict[ques]
        answer_id = np.where(answers_list[i]>0)[0]
        answer_str = [answers_pool[k] for k in answer_id]

        image_id = qj['image_id']
        generate_qa_pair_dict[image_id].append([ques, " ".join(answer_str)])

    # ============ SORT VQA output BY IMAGE_ID AND INTO STR ===========================
    print("store generate_qa_pair_dict!")
    return generate_qa_pair_dict




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_tag', type=str, default="1121_30_smallid")
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args


def generate_unilm_pairs(args, model_path):
    file_name = args.file_tag
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    saved_path = "/home/ymeng/store_ym/medical_report_vqa/data_saved/%s/vqa_eval"%file_name
    if os.path.isdir(saved_path) == False:
        os.makedirs(saved_path)
    dataroot = "/home/ymeng/store_ym/nlp_422/py3_bottom_up/data"
    cache = "%s/cache_%s"%(dataroot, file_name)
    dictionary = Dictionary.load_from_file('{}/dictionary_chest.pkl'.format(cache)) 
    questions_json = json.load(open("%s/questions.json"%cache,"r"))
    train_dset = VQAFeatureDataset(questions_json, dictionary, dataroot=dataroot, cache=cache)
    model = torch.load(model_path)      
    question2answer_dict = json.load(open("%s/question2answer_dict.json"%cache,"r"))
    generate_qa_pair_dict = generate_ques_ans_list(model, train_dset, cache, question2answer_dict, questions_json)
    pickle.dump(generate_qa_pair_dict,open("%s/generate_qa_pairs_by_image_id.pkl"%saved_path,"wb"))

    link_keys_index = {}
    count = 0
    line_list = []
    for k, qa_list in generate_qa_pair_dict.items():
        count += len(qa_list) 
        link_keys_index[k] = count
        for (q,a) in qa_list:
            line = q + " [SEP] " + a 
            line_list.append(line)
    json.dump(link_keys_index, open("%s/keys_index.json"%saved_path,"w"))
    json.dump(line_list, open("%s/line_list.json"%saved_path,"w"))

    index_list = []
    line_set = []
    for i,line in enumerate(line_list):
        if line not in line_set:
            index_list.append(len(line_set))
            line_set.append(line)
        else:
            index_list.append(line_set.index(line))
    json.dump(index_list, open("%s/index_list.json"%saved_path,"w"))
    print("total line:", len(line_list))  
    print("unrepeat line:", len(line_set))   
    unilm_path = "/home/ymeng/store_ym/nlp_422/unilm/unilm-v1/data_generate_answer/medical_report/qa2s_%s/eval"%file_name
    print("saved in: ",unilm_path)
    
    if os.path.isdir(unilm_path) == False:
        os.makedirs(unilm_path)
    qa_path = os.path.join(unilm_path, "qa.txt")
    f = open(qa_path,"w")   
    for line in line_set:
        f.write(line + "\n")
    f.close()




def generate_report_from_unilm(idx, file_name="1121_30_smallid"):
    def load_unilm_data(ind, file_tag="1121_30_smallid"):
        generate_file = "/home/ymeng/store_ym/nlp_422/unilm/unilm-v1/data_generate_answer/medical_report/qa2s_%s/eval"%file_tag
        sent_path = os.path.join(generate_file, "sent_generate%d.txt"%ind)
        unilm_sentences = open(sent_path,"r").readlines()
        return unilm_sentences
    saved_path = "/home/ymeng/store_ym/medical_report_vqa/data_saved/%s/vqa_eval"%file_name
    link_keys_index = json.load(open("%s/keys_index.json"%saved_path,"r"))
    line_list = json.load(open("%s/line_list.json"%saved_path,"r"))
    index_list = json.load(open("%s/index_list.json"%saved_path,"r"))
    sentences = load_unilm_data(idx, file_tag = file_name)
    # generate_qa_p air_dict = pkl.load(open("%s/generate_qa_pairs_by_image_id.pkl"%saved_path,"rb"))
    report_generated_list = {}
    count = 0
    for k in link_keys_index:
        count_new = link_keys_index[k]
        if count_new < count:
            print("False")
        index_tmp = index_list[count:count_new]
        report_generated_list[k] = [sentences[i].strip() for i in index_tmp]
        # if count == 0:
        #     print("{}\n\n{}\n\n{}\n\n".format(line_tmp,report_generated_in_list[k],data[k]))
        #     import pdb; pdb.set_trace()
        count = count_new
    return report_generated_list

if __name__ == "__main__":
    args = parse_args()
    step = 2
    if step == 1:        
        name = "small_mask_newatt_emb_gensim_dim50"
        model_name = "./saved_lungs/%s/%smodel.pth"%(args.file_tag, name)
        generate_unilm_pairs(args, model_name)
    else:
        file_name = args.file_tag
        idx = 1
        report_generated_list = generate_report_from_unilm(idx, file_name=file_name)
        saved_path = "/home/ymeng/store_ym/medical_report_vqa/data_saved/%s/vqa_eval"%file_name
        json.dump(report_generated_list, open("%s/generated_report_%d.json"%(saved_path, idx),"w"))



    


