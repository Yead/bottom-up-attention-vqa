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
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
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


class BaseModel(nn.Module):
    def __init__(self, visual_net, w_emb, q_emb, a_emb, answer_np, a_net, v_att, q_net, v_net, classifier, classifier2, mask, tag=""):
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

        if "ansemb" in self.tag:
            aw_emb = self.w_emb(torch.tensor(self.answer_id).cuda())
            ans_emb = self.a_emb(aw_emb)
            # import pdb; pdb.set_trace()
            if "ansnet" in self.tag:              
                a_repr = self.a_net(ans_emb).transpose(1,0)
            else:
                a_repr = ans_emb.transpose(1,0)
            ans = a_repr[np.newaxis,:,:].repeat(q_id.shape[0],1,1)  
            joint_repr = joint_repr[:,np.newaxis,:]
            logits = torch.matmul(joint_repr, ans).squeeze()
            if "finalfc" in self.tag:
                logits = self.classifier2(logits.float())
            if "mask" in self.tag:
                q_mask = self.mask[q_id]
                logits = logits * q_mask 
                # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            return logits
            
        elif "mask_ori" in self.tag :
            return self.mask_ori(q_id, joint_repr)
        else:
            logits = self.classifier(joint_repr)
            return logits

    def emb_mask(self, q_id, joint_repr):
        ans_emb = self.a_emb(self.w_emb(torch.tensor(self.answer_id).cuda())).transpose(1,0)
        ans = ans_emb[np.newaxis,:,:].repeat(q_id.shape[0],1,1)  
        joint_repr = joint_repr[:,np.newaxis,:]
        logits = torch.matmul(joint_repr, ans).squeeze()
        logits_norm = self.bn1(logits)
        q_mask = self.mask[q_id]
        logits_norm = logits_norm * q_mask                
        return logits_norm
    
    def emb(self, joint_repr):
        ans_emb = self.a_emb(self.w_emb(torch.tensor(self.answer_id).cuda())).transpose(1,0)
        logits = torch.matmul(joint_repr, ans_emb) 
        return logits
    
    def mask_ori(self, q_id, joint_repr):
        q_mask = self.mask[q_id]
        logits = self.classifier(joint_repr)
        logits = logits * q_mask
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

def build_baseline0_newatt(dataset, num_hid, emb_dim, answer_np, mask, pick_idx, tag):
    visual_net = Resnet18Templet(3, pretrained=False)
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim, 0.0)
    q_emb = QuestionEmbedding(emb_dim, num_hid, 1, False, 0.0)
    a_emb = QuestionEmbedding(emb_dim, num_hid, 1, False, 0.0)
    v_dim = 2048
    v_att = Attention(v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    a_net = FCNet([num_hid, num_hid])
    # v_net = FCNet([v_dim, num_hid])
    # v_net = nn.Linear(v_dim, num_hid)
    # v_net = weight_norm(nn.Linear(v_dim, num_hid), dim=None)
    v_net = reluclassifier(v_dim, num_hid)
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    classifier2 = SimpleClassifier2(
        dataset.num_ans_candidates, 2 * dataset.num_ans_candidates, dataset.num_ans_candidates, 0.5)
    return BaseModel(visual_net, w_emb, q_emb, a_emb, answer_np,a_net, v_att, q_net, v_net, classifier, classifier2, mask, tag=tag)






class CoAttenModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_emb, fc,embed_dim=300,k=30):
        super(CoAttenModel,self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_emb = v_emb
        self.tanh = nn.Tanh()
        self.conv = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        # self.W_b = nn.Parameter(torch.randn(embed_dim, 63))
        # self.W_v = nn.Parameter(torch.randn(k, 63))
        self.W_b = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_v = nn.Parameter(torch.randn(k, embed_dim))
        self.W_q = nn.Parameter(torch.randn(k, embed_dim))
        self.w_hv = nn.Parameter(torch.randn(k, 1))
        self.w_hq = nn.Parameter(torch.randn(k, 1))

        self.W_w = nn.Linear(embed_dim, embed_dim)
        # self.W_p = nn.Linear(embed_dim*2, embed_dim)
        self.W_s = nn.Linear(embed_dim*2, embed_dim)

        self.fc = fc

    def forward(self, v, b, q, label):
        # v = v[:,0,:].unsqueeze(dim=1).expand(-1,300,-1)
        v_emb, avg_v = self.v_emb(v)
        v_conv = self.conv(v_emb)
        v_test = fn.interpolate(v_conv, size=[300,300],mode='bilinear').squeeze(dim=1)
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)
        q_emb = q_emb.unsqueeze(dim=1)
        # import pdb;pdb.set_trace()
        v_word, q_word = self.parallel_co_attention_word(v_test, w_emb)  # 
        # v_phrase, q_phrase = self.parallel_co_attention(v, phrase)
        v_sent, q_sent = self.parallel_co_attention_ques(v_test, q_emb)

        h_w = self.tanh(self.W_w(q_word + v_word))
        # h_p = self.tanh(self.W_p(torch.cat(((q_phrase + v_phrase), h_w), dim=1)))
        h_s = self.tanh(self.W_s(torch.cat(((q_sent + v_sent), h_w), dim=1)))

        logits = self.fc(h_s)

        return logits

    def parallel_co_attention_word(self, V, Q):  # V : B x 512 x 196, Q : B x L x 512 || v:B*2*2048   q:B*14*300 
        C = torch.matmul(Q, torch.matmul(self.W_b, V)) # B x L x 196              || B * 14 * 2048

        H_v = self.tanh(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))                            # B x k x 196     ||B* k*14
        H_q = self.tanh(torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))           # B x k x L       ||B* k*14
        
        a_v = fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2) # B x 1 x 196
        a_q = fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2) # B x 1 x L

        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1))) # B x 512
        q = torch.squeeze(torch.matmul(a_q, Q))                  # B x 512

        return v, q



    def parallel_co_attention_ques(self, V, Q):  # V : B x 512 x 196, Q : B x L x 512 || v:B*2*2048   q:B*14*300 
        C = torch.matmul(Q, torch.matmul(self.W_b, V)) # B x L x 196              || B * 14 * 2048

        H_v = self.tanh(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))                            # B x k x 196     ||B* k*14
        H_q = self.tanh(torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))           # B x k x L       ||B* k*14
        
        a_v = fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2) # B x 1 x 196
        a_q = fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2) # B x 1 x L

        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1))) # B x 512
        q = torch.squeeze(torch.matmul(a_q, Q))                  # B x 512

        return v, q




def build_co_attention(dataset, num_hid, emb_dim):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim, 0.0)
    q_emb = QuestionEmbedding_word(emb_dim, num_hid, 1, False, 0.0)
    v_emb = VisualFeatureExtractor()
    fc = nn.Linear(embed_dim, dataset.num_ans_candidates)
    return CoAttenModel(w_emb, q_emb, v_emb,fc)



