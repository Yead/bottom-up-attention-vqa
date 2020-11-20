import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import pickle as cPickle
import numpy as np

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True):             
    """Function that measures Binary Cross Entropy between target and output logits.                    
    
    See :class:`~torch.nn.BCEWithLogitsLoss` for details.                                        
                           
    Args:                     
        input: Variable of arbitrary shape 
        target: Variable of the same shape as input 
        weight (Variable, optional): a manual rescaling weight if provided it's repeated to match input tensor shape 
        size_average (bool, optional): By default, the losses are averaged over observations for each minibatch. However, if the field  sizeAverage is set to False, the losses are instead summed                  
    for each minibatch. Default: ``True`` 
 
    Examples::            
        >>> input = autograd.Variable(torch.randn(3), requires_grad=True)                      
        >>> target = autograd.Variable(torch.FloatTensor(3).random_(2))                      
        >>> loss = F.binary_cross_entropy_with_logits(input, target)                      
        >>> loss.backward()   
    """                 
    if not (target.size() == input.size()): 
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size())) 
    max_val = (-input).clamp(min=0) 
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log() 

    if weight is not None: 
        loss = loss * weight  

    if size_average:        
        return loss.mean()    
    else:               
        return loss.sum()  



def instance_bce_with_logits(logits, labels, weights):
    assert logits.dim() == 2
 
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, weights)
    # loss = binary_cross_entropy_with_logits(logits, labels, weights)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores



####################################################
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



def evaluate(model, dataloader, cache):
    score = 0
    upper_bound = 0
    num_data = 0
    ans2label_path = os.path.join( cache, 'trainval_ans2label.pkl')
    label2ans_path = os.path.join( cache, 'trainval_label2ans.pkl')
    ans2label = cPickle.load(open(ans2label_path, 'rb'))
    label2ans = cPickle.load(open(label2ans_path, 'rb'))
    for v, b, q, a in iter(dataloader):
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        pred = model(v, b, q, None)
        logits = torch.max(pred, 1)[1].data
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)
        # print(batch_score)
        ans = show_answer(a,label2ans)
        ans2 = show_answer(logits,label2ans)
        # for i in range(len(ans)):
        #     print("ques:",qt[i],"0:",ans[i],"1: ",ans2[i],ans2[i]==ans[i])
    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound
##################################################u###########################


def get_id_np(tag_list, tag):
    ques_id_dict = {}
    id_max = 0
    for q in tag_list:
        ques_id = q[tag]
        if isinstance(ques_id, list):
            if len(ques_id) > 0:
                ques_id = ques_id[0]
            else:
                ques_id = 0
        if ques_id not in ques_id_dict.keys():
            ques_id_dict[ques_id] = 0
        ques_id_dict[ques_id] += 1
        if ques_id > id_max:
            id_max = ques_id
    ques_id_np = np.zeros([id_max+1])
    sum_ques_nums = 0
    for i in range(id_max+1):
        if i in ques_id_dict.keys():
            ques_id_np[i] = ques_id_dict[i]
    # ques_weights_np = np.array([idx for idx in ques_id_np],dtype=np.float)
    ques_weights_np = np.array([1 for idx in ques_id_np],dtype=np.float)
    # ques_id_np = ques_id_np/ques_id_np.sum()  
    return ques_weights_np



def train(model, train_loader,eval_loader,  cache, num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    dataroot = 'data'
    name = 'train'
    # question_path = os.path.join(cache, "%s_questions.json"%name)
    # questions_list = json.load(open(question_path))['questions']
    answer_path = os.path.join(cache, 'total_target.pkl')
    answers = cPickle.load(open(answer_path, 'rb'))
    ans_id_np = get_id_np(answers, 'labels')

    ans_id_torch = torch.from_numpy(ans_id_np)#.to(torch.int)
    
    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        # for i, (v, b, q, a,qt) in enumerate(train_loader):           
        for i, (v, b, q, a) in enumerate(train_loader):  

        # for i, (t) in enumerate(train_loader): 
        #     v = Variable(t).cuda()  #.permute(0, 3, 1, 2) #features
        #     b = Variable(t).cuda() 
        #     q = Variable(t).cuda() #quesiton
        #     a = Variable(t).cuda() #targets
            v = Variable(v).cuda()
            b = torch.squeeze(b)
            q = Variable(q).cuda() #quesiton
            a = Variable(a).cuda() #targets

            # weights = Variable(torch.FloatTensor([1]*117)).cuda()
            weights = Variable(ans_id_torch.float()).cuda()
            # weights = ans_id_torch
            pred = model(v, b, q, a)
            # loss1 = instance_bce_with_logits(pred, a, weights)
            loss = instance_bce_with_logits(pred, a, weights)
            # print("loss1:", loss1)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            # import pdb; pdb.set_trace()
            # total_loss += loss.data[0] * v.size(0)
            total_loss += loss.data * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader, cache)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            # dict_path = os.path.join(output, 'model_dict.pth')
            # torch.save(model.state_dict(), dict_path)

            model_path = os.path.join(output, 'model.pth')
            torch.save(model,model_path)
            best_eval_score = eval_score


def _evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    for v, b, q, a in iter(dataloader):
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        pred = model(v, b, q, None)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound


def check_ans_size(cache):
    dataroot = 'data'
    # cache = 'cache_0810'
    name = 'test'
    # question_path = os.path.join(dataroot, cache, "%s_questions.json"%name)
    # questions_list = json.load(open(question_path))['questions']
    answer_path = os.path.join(dataroot, cache, '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    ans_id_np = get_id_np(answers, 'labels')

    ans_id_torch = torch.from_numpy(ans_id_np)#.to(torch.int)
    print(ans_id_torch.size())
    return ans_id_torch, answers 
#{'image_id': u'CXR431_IM-2072-1002.png', 'labels': [], 'question_id': 17459, 'scores': []}] 
if  __name__ == "__main__":
    cache = 'cache_0814'
    torch, answers  = check_ans_size(cache)
    import pdb; pdb.set_trace()
