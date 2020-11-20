import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torchvision.models as models
from torch.autograd import Variable
import numpy as np

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
        # print(self.ntoken, self.emb_dim, weight_init.shape)
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
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):  #(dataset.v_dim, q_emb.num_hid, num_hid)
        super(NewAttention,self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)
    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        v_proj = v.unsqueeze(dim=2)
        q_proj = q.unsqueeze(dim=1)
        att = torch.bmm(v_proj, q_proj) # v_dim * q_dim        
        v_att = torch.mean(att, dim=2) * v_proj.squeeze()
        q_att = torch.mean(att, dim=1) * q_proj.squeeze()
        vq = torch.cat((v_att, q_att), 1)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits



class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        # logits = self.att(v, q)
        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits




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
        # print("LSTM--layers:",self.nlayers," in_dim",self.in_dim, " num_hid:", self.num_hid)
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
    def __init__(self, model_name='resnet152', pretrained=False):
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
            resnet = models.resnet18(pretrained=self.pretrained)
            modules = list(resnet.children())[:6]
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




def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)



class dicti:
    def __init__(self, ntoken):
        self.ntoken = ntoken

class datasets:
    def __init__(self, ntoken ,num_hid,v_dim, num_ans_candidates):
        self.dictionary = dicti(ntoken)
        self.num_hid = num_hid
        self.v_dim  = v_dim 
        self.num_ans_candidates = num_ans_candidates

# from torchsummary import summary

if __name__ == "__main__":        
    num_hid = 30
    dt = datasets(100, num_hid, 100, 100)
    model = build_baseline0_newatt(dt, num_hid)
    v = torch.randn((1, 3, 512, 512)).cuda()
    b = torch.randn((1, 3, 512, 512)).cuda()
    q = torch.randn((1, 14)).cuda()

    # # print(model)
    # summary(model,[(3, 512, 512),(12, 12),(10,),(10,)])
    pred = model(v, b, q, None)

