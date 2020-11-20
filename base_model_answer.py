import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn.utils.weight_norm import weight_norm
import torchvision.models as models


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



class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout):
        super(WordEmbedding,self).__init__()
        self.emb = nn.Embedding(ntoken+1, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim
    
    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        print(self.ntoken, self.emb_dim, weight_init.shape)
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        output = self.dropout(emb)
        return output


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)
    
    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == "LSTM":
            return(Variable(weight.new(*hid_shape).zero_()), Variable(weight.new(*hid_shape).zero_()))
        else:
            return(Variable(weight.new(*hid_shape).zero_()))

    def forward(self,x):
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x,hidden)        
        if self.ndirections == 1:
            return output[:,-1]
        
        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]          
        return torch.cat(forward_,backward,dim=1)



    def forwad_all(x,batch):
        batch = x.size(0)
        hidden = init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x,hidden)
        return output


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
    def __init__(self, w_emb, q_emb, a_emb, answer_np, v_att, q_net, v_net, classifier, mask, tag=""):
        super(BaseModel,self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.a_emb = a_emb
        self.answer_id = answer_np
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.mask = torch.tensor(mask).cuda()
        self.tag = tag
    
    def forward(self, v, b, q, label):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)
        att = self.v_att(v,q_emb)
        q_repr = self.q_net(q_emb)
        v_emb = (att * v).sum(1)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        if self.tag == "emb_mask":
            return self.emb_mask(q_id, joint_repr)
        elif self.tag == "mask":
            return self.mask_ori(b, joint_repr)
        else:
            logits = self.classifier(joint_repr)
            return logits
    def emb_mask(self, q_id, joint_repr):
        ans_emb = self.a_emb(self.w_emb(torch.tensor(self.answer_id).cuda())).transpose(1,0)
        ans = ans_emb[np.newaxis,:,:].repeat(q_id.shape[0],1,1)  
        q_mask = self.mask[q_id][:,np.newaxis,:].repeat(1,ans_emb.shape[0],1)  
        ans_mask = (ans * q_mask).float()        
        joint_repr = joint_repr[:,np.newaxis,:]
        logits = torch.matmul(joint_repr, ans_mask) 
        return logits.squeeze()
    
    def emb(self, joint_repr):
        ans_emb = self.a_emb(self.w_emb(torch.tensor(self.answer_id).cuda())).transpose(1,0)
        logits = torch.matmul(joint_repr, ans_emb) 
        return logits
    
    def mask_ori(self, q_id, joint_repr):
        q_mask = self.mask[q_id]
        logits = self.classifier(joint_repr)
        logits = logits * q_mask
        return logits


def build_baseline0_newatt(dataset, num_hid, emb_dim, answer_np, mask):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim, 0.0)
    q_emb = QuestionEmbedding(emb_dim, num_hid, 1, False, 0.0)
    a_emb = QuestionEmbedding(emb_dim, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, a_emb, answer_np, v_att, q_net, v_net, classifier, mask, tag="")



class BaseModel_answer(nn.Module):
    def __init__(self, w_emb, q_emb, a_emb, answer_np, v_att, q_net, v_net, classifier):
        super(BaseModel_answer,self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.a_emb = a_emb
        self.answer_id = answer_np
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
    
    def forward(self, v, b, q, label):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)
        att = self.v_att(v,q_emb)
        q_repr = self.q_net(q_emb)
        v_emb = (att * v).sum(1)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        ans_emb = self.a_emb(self.w_emb(torch.tensor(self.answer_id).cuda())).transpose(1,0)
        logits = torch.matmul(joint_repr, ans_emb) 
        # logits = self.classifier(joint_repr)
        return logits

class BaseModel_mask(BaseModel_answer):
    def __init__(self, w_emb, q_emb, a_emb, answer_np, v_att, q_net, v_net, classifier, mask):
        super(BaseModel_mask, self).__init__(w_emb, q_emb, a_emb, answer_np, v_att, q_net, v_net, classifier)
        
        self.mask = torch.tensor(mask).cuda()
    
    def forward(self, v, q_id, q, label):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)
        att = self.v_att(v,q_emb)
        q_repr = self.q_net(q_emb)
        v_emb = (att * v).sum(1)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        
        ans_emb = self.a_emb(self.w_emb(torch.tensor(self.answer_id).cuda())).transpose(1,0)
        ans = ans_emb[np.newaxis,:,:].repeat(q.shape[0],1,1)  
        q_mask = self.mask[q_id][:,np.newaxis,:].repeat(1,ans_emb.shape[0],1)  
        ans_mask = (ans * q_mask).float()
        
        joint_repr = joint_repr[:,np.newaxis,:]
        logits = torch.matmul(joint_repr, ans_mask) 
        # logits = self.classifier(joint_repr)
        return logits.squeeze()






def build_baseline0_newatt_mask(dataset, num_hid, emb_dim, answer_np, mask):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim, 0.0)
    q_emb = QuestionEmbedding(emb_dim, num_hid, 1, False, 0.0)
    a_emb = QuestionEmbedding(emb_dim, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel_mask(w_emb, q_emb, a_emb, answer_np, v_att, q_net, v_net, classifier, mask)



def build_baseline0_newatt_nomask(dataset, num_hid, emb_dim, answer_np):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim, 0.0)
    q_emb = QuestionEmbedding(emb_dim, num_hid, 1, False, 0.0)
    a_emb = QuestionEmbedding(emb_dim, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel_answer(w_emb, q_emb, a_emb, answer_np, v_att, q_net, v_net, classifier)


def build_baseline0_newatt_ori(dataset, num_hid, emb_dim):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim, 0.0)
    q_emb = QuestionEmbedding(emb_dim, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    # import pdb; pdb.set_trace()
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


import torch.nn.functional as fn
class QuestionEmbedding_word(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        super(QuestionEmbedding_word, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)
    
    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == "LSTM":
            return(Variable(weight.new(*hid_shape).zero_()), Variable(weight.new(*hid_shape).zero_()))
        else:
            return(Variable(weight.new(*hid_shape).zero_()))

    def forward(self,x):
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x,hidden)        
        if self.ndirections == 1:
            return output[:,-1]
        
        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]    
    
        return torch.cat(forward_,backward,dim=1)

    def forwad_all(x,batch):
        batch = x.size(0)
        hidden = init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x,hidden)
  
        return output

class VisualFeatureExtractor(nn.Module):
    def __init__(self, model_name='densenet201', pretrained=False):
        super(VisualFeatureExtractor, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, self.out_features, self.avg_func, self.bn, self.linear = self.__get_model()
        self.activation = nn.ReLU()

    def __get_model(self):
        model = None
        out_features = None
        func = None
        if self.model_name == 'resnet152':
            resnet = models.resnet152(pretrained=self.pretrained)
            modules = list(resnet.children())[:3]
            model = nn.Sequential(*modules)
            out_features = resnet.fc.in_features
            func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        elif self.model_name == 'densenet201':
            densenet = models.densenet201(pretrained=self.pretrained)
            modules = list(densenet.features)[:3]
            model = nn.Sequential(*modules)
            func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            out_features = densenet.classifier.in_features
        linear = nn.Linear(in_features=out_features, out_features=out_features)
        bn = nn.BatchNorm1d(num_features=out_features, momentum=0.1)
        return model, out_features, func, bn, linear

    def forward(self, images):
        """
        :param images:
        :return:
        """
        visual_features = self.model(images)
        avg_features = self.avg_func(visual_features).squeeze()
        # avg_features = self.activation(self.bn(self.linear(avg_features)))
        return visual_features, avg_features



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




def build_co_attention(dataset, num_hid,  emb_dim):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim, 0.0)
    q_emb = QuestionEmbedding_word(emb_dim, num_hid, 1, False, 0.0)
    v_emb = VisualFeatureExtractor()
    fc = nn.Linear(embed_dim, dataset.num_ans_candidates)
    return CoAttenModel(w_emb, q_emb, v_emb,fc)



