import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset_mask import Dictionary
from dataset_mask import VQAFeatureDataset
# from dataset_mask import VQAFeatureDataset_fast as VQAFeatureDataset
# import base_model_mask as base_model
import base_model
# import model_ym as base_model
from train_mask import train
import utils
from torch.autograd import Variable
import pickle as cPickle
import os
import json


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
   
    return scores

def getzero(a):
    for i in range(a.shape[0]):
        if a[i] > 0:
            return i

def show_answer(a,label2ans):
    a_np = a.cpu().numpy()
    ans = []
    if len(a_np.shape) > 1:
        for i in range(a_np.shape[0]):
            n = getzero(a_np[i])
            if n != None:
                ans.append(label2ans[n])
            else:
                ans.append("no answer")
            
    else:
        for i in range(a_np.shape[0]):
            ans.append(label2ans[a_np[i]])
    return ans


def evaluate(model, test_dset, dataloader, data_root, cache):
    score = 0
    upper_bound = 0
    num_data = 0
    ans2label_path = os.path.join(data_root, cache, 'trainval_ans2label.pkl')
    label2ans_path = os.path.join(data_root, cache, 'trainval_label2ans.pkl')
    ans2label = cPickle.load(open(ans2label_path, 'rb'))
    label2ans = cPickle.load(open(label2ans_path, 'rb'))
    entry = test_dset.entries
    # for img,v, b, q, a, qt in iter(dataloader):
    for _, (v, b, q, a) in enumerate(iter(dataloader)):
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        pred = model(v, b, q, None)
        logits = torch.max(pred, 1)[1].data
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)
        print(batch_score)
        ans = show_answer(a,label2ans)
        ans2 = show_answer(logits,label2ans)
        f = open("./records/result_lung.json","w+")
        results = []
        for i0 in range(len(ans)):
            i = _ * 50 + i0
            if ans[i0]!= ans2[i0]:
                print(entry[i]['answer']['labels'] , '\t',  ans[i0], ans2[i0])
            res = {"entry":entry[i]['answer']['labels'],"true":ans[i],"pred":ans2[i]}
            results.append(res)
        json.dump(results,f)
        f.close()
    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    args = parse_args()
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    cache = "./data/cache_%s"%args.cache
    dictionary = Dictionary.load_from_file('{}/dictionary_chest.pkl'.format(cache))#  
    
    print("build dataset")

    train_target = "{}/train_target.pkl".format(cache)
    val_target = "{}/val_target.pkl".format(cache)
    #test_target = "{}/test_target.pkl".format(cache) 
    features = "data/cache2/features.json"
    train_dset = VQAFeatureDataset('train', dictionary, train_target, cache=cache)
    eval_dset = VQAFeatureDataset('val', dictionary, val_target, cache=cache)
    # train_dset = VQAFeatureDataset_debug('train')
    # eval_dset = VQAFeatureDataset_debug('val')
    #test_dset = VQAFeatureDataset('test', dictionary, test_target, cache=cache)
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    emb_dim = 50
    num_hid = emb_dim


    answer_np = np.load("%s/answer.npy"%cache)
    mask = np.load("%s/mask.npy"%cache) 
    pick_idx = np.load("%s/pick_idx.npy"%cache) 

    tag = "small_mask_"+ args.tag
    print("pick_idx.shape:",pick_idx.shape, "model_type:", tag)
    model = getattr(base_model, constructor)(train_dset, num_hid, emb_dim, answer_np, mask, pick_idx, tag=tag).cuda()
    if "gensim" in tag: 
        model.w_emb.init_embedding('%s/gensim_init_%dd_chest.npy'%(cache, emb_dim))
    elif "glove" in tag:
        model.w_emb.init_embedding('%s/glove6b_init_%dd_chest.npy'%(cache, emb_dim))
    # model.w_emb.init_embedding('data/glove6b_init_300d_chest_0807.npy')
    
    # model_path = "./saved_lungs/lung_0807/model.pth"
    # model.load_state_dict(torch.load(model_path))
    # model = nn.DataParallel(model).cuda()
    model = model.cuda()
    print("-----start to train----")
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
    output_file = "saved_lungs/%s"%args.cache
    train(model, train_loader, eval_loader, cache, args.epochs, output_file, tag)


    # pthfile = output_file +"/model_pretrain.pth"    
    # pthfile = "saved_models_lung/exp32/model.pth" 
    # model.load_state_dict(torch.load(pthfile))
    

    # test_loader =  DataLoader(test_dset, 50, shuffle=False, num_workers=1)
    
    # eval_score, bound = evaluate(model, test_dset, test_loader,data_root, cache)
   # print('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
