import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset_mask import Dictionary
from dataset_mask import VQAFeatureDataset
# from dataset_mask import VQAFeatureDataset_fast as VQAFeatureDataset
import base_model_mask as base_model
# import base_model
# import model_ym as base_model
from train_mask import train
import utils
from torch.autograd import Variable
import pickle as cPickle
import os
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--batch_size', type=int, default= 32)#512
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cache', type=str, default="1121_30_smallid")
    parser.add_argument('--tag', type=str, default="")
    args = parser.parse_args()
    return args


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

    train_dset = VQAFeatureDataset('train', dictionary, train_target, cache=cache)
    eval_dset = VQAFeatureDataset('val', dictionary, val_target, cache=cache)

    constructor = 'build_%s' % args.model
    emb_dim = args.emb_dim
    num_hid = emb_dim

    answer_np = np.load("%s/answer.npy"%cache)
    mask = np.load("%s/mask.npy"%cache) 
    pick_idx = np.load("%s/pick_idx.npy"%cache) 

    tag = "small_mask_"+ args.tag
    print("pick_idx.shape:", pick_idx.shape, "model_type:", tag)
    model = getattr(base_model, constructor)(train_dset, num_hid, emb_dim, answer_np, mask, pick_idx, tag=tag).cuda()
    if "gensim" in tag: 
        model.w_emb.init_embedding('%s/gensim_init_%dd_chest.npy'%(cache, emb_dim))
        print("init w_emb with gensim ", emb_dim)
    elif "glove" in tag:
        model.w_emb.init_embedding('%s/glove6b_init_%dd_chest.npy'%(cache, emb_dim))
        print("init w_emb with glove ", emb_dim)
    else:
        print("not init w_emb")

    model = model.cuda()
    print("-----start to train----")
    train_loader = DataLoader(train_dset, args.batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, args.batch_size, shuffle=True, num_workers=1)
    output_file = "saved_lungs/%s"%args.cache
    train(model, train_loader, eval_loader, cache, args.epochs, output_file, tag)

