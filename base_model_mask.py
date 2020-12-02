import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn.utils.weight_norm import weight_norm
import torchvision.models as models
from build_model import Resnet18Templet
from base_model import VisualFeatureExtractor, WordEmbedding, QuestionEmbedding

class FCNet(nn.Module):
    def __init__(self, dims):
        super(FCNet, self).__init__()
        layers = []
        for i in range(len(dims)-2): 
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())
        self.main = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.main(x)

class FCNet_(nn.Module):
    def __init__(self, dims):
        super(FCNet, self).__init__()
        layers = []
        for i in range(len(dims)-2): 
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.ReLU())
        self.main = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.main(x)

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

class SimpleClassifier2(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier2, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(hid_dim, out_dim)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

class reluclassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(reluclassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, out_dim), dim=None),
            nn.ReLU()
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        logits = self.logits(v,q)
        w = nn.functional.softmax(logits, 1)
        return w
    def logits(self, v, q):
        vq = torch.cat((v, q), 1)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits

    def logits_ori(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention,self).__init__()
        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)
    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits


class AttModel_v1(nn.Module):
    def __init__(self, embedding, v_net, q_net, classifier, n):
        super(AttModel_v1, self).__init__()
        self.embedding = embedding
        self.v_net = v_net
        self.q_net= q_net
        self.classifier = classifier
        self.n = n
    def forward(self, img, emb_id):
        emb = self.embedding(emb_id)        
        v_proj = self.v_net(img)
        w_proj = self.q_net(emb.float()).repeat(1, self.n).view(img.shape[0],-1)
        joint_repr = v_proj * w_proj
        # logits = self.classifier(joint_repr)
        return joint_repr



class BaseModel(nn.Module):
    def __init__(self, visual_net, w_emb, q_emb, a_emb, answer_np, a_net, v_att, q_net, v_net, classifier, classifier2, mask, pick_idx, tag=""):
        super(BaseModel,self).__init__()
        self.visual_net = visual_net
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.a_emb = a_emb
        self.a_net = a_net
        self.answer_id = answer_np
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.mask = torch.tensor(mask).cuda()
        self.tag = tag
        self.bn1 = nn.BatchNorm1d(num_features=mask.shape[1])
        self.classifier2 = classifier2
        self.pick_idx = torch.tensor(pick_idx).cuda()
        self.thresh = torch.tensor(0.5)
        print(self.mask.shape, self.pick_idx.shape)

    def forward(self, v, q_id, q, label):
        v = self.visual_net(v)
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)
        att = self.v_att(v, q_emb)
        if "qnet" in self.tag:
            q_repr = self.q_net(q_emb)
        else:
            q_repr = q_emb
        v_emb = att * v
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr

        if "emb" in self.tag:
            aw_emb = self.w_emb(torch.tensor(self.answer_id).cuda())
            ans_emb = self.a_emb(aw_emb)
            if "ansnet" in self.tag:              
                a_repr = self.a_net(ans_emb).transpose(1,0)
            else:
                a_repr = ans_emb.transpose(1,0)
            ans = a_repr[np.newaxis,:,:].repeat(q_id.shape[0],1,1)  
            q_mask = self.pick_idx[q_id].unsqueeze(dim=1).repeat(1,ans_emb.shape[1],1)
            ans_mask = torch.gather(ans, 2, q_mask.long())

            joint_repr = joint_repr[:,np.newaxis,:]
            logits = torch.matmul(joint_repr, ans_mask).squeeze()
            if "finalfc" in self.tag:
                logits = self.classifier2(logits.float())
            return logits
        else:
            q_mask = self.pick_idx[q_id]
            logits = self.classifier(joint_repr)
            
            logits = logits * q_mask
            return logits



def build_baseline0_newatt(dataset, num_hid, emb_dim, answer_np, mask, pick_idx, tag):
    visual_net = Resnet18Templet(3, pretrained=False)
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim, 0.0)
    q_emb = QuestionEmbedding(emb_dim, num_hid, 1, False, 0.0)
    a_emb = QuestionEmbedding(emb_dim, num_hid, 1, False, 0.0)
    v_dim = 2048
    # if "attv1" in tag:
    #     v_att = 

    if "newatt" in tag:
        v_att = NewAttention(v_dim, num_hid, num_hid)

    else:
        v_att = Attention(v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    a_net_fc = FCNet([num_hid, num_hid])
    a_net_rl = reluclassifier(num_hid, num_hid)
    a_net_lin = nn.Linear(num_hid, num_hid)
    # v_net = FCNet([v_dim, num_hid])
    # v_net = nn.Linear(v_dim, num_hid)
    # v_net = weight_norm(nn.Linear(v_dim, num_hid), dim=None)
    v_net = reluclassifier(v_dim, num_hid)
    num_candidates = pick_idx.shape[1]
    classifier = SimpleClassifier(num_hid, 2 * num_candidates, num_candidates, 0.5)    
    classifier2 = SimpleClassifier2(num_candidates, 2 * num_candidates, num_candidates, 0.5)
    return BaseModel(visual_net, w_emb, q_emb, a_emb, answer_np,a_net_fc, v_att, q_net, v_net, classifier, classifier2, mask, pick_idx,tag=tag)


def build_attention_attv1(model_type="v1", word_in_dim=100, word_out_dim=128,  pretrained=False, output_nums=2):
    visual_net = Resnet18Templet(3, pretrained=False)
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim, 0.0)
    q_emb = QuestionEmbedding(emb_dim, num_hid, 1, False, 0.0)
    a_emb = QuestionEmbedding(emb_dim, num_hid, 1, False, 0.0)
    visual_out_dim = 2048
    q_net = FCNet([num_hid, num_hid])
    a_net_fc = FCNet([num_hid, num_hid])
    # word_net =nn.Linear(word_in_dim, word_out_dim)        
    if model_type == 'v1': #build_attention_suc
        print( "v * w.reapeat + classifier from 2048, word_out_dim=",word_out_dim )
        classifier = SimpleClassifier(visual_out_dim , word_out_dim, output_nums, 0.5)
        return AttModel_v1(embedding, visual_net, q_net, classifier, visual_out_dim//word_out_dim)

