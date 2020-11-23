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
    target = (target > 0).float()                
    if not (target.size() == input.size()): 
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size())) 
    max_val = (-input).clamp(min=0) 
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log() 

    # P = ((-max_val).exp() + (-input - max_val).exp()).log() 

    # weight = (target * 7 + (1-target) * 1) * torch.square(target - P)
    weight = (target * 7 + (1-target) * 1) 
    if weight is not None: 
        loss = loss * weight  

    if size_average:        
        return loss.mean()    
    else:               
        return loss.sum()  



def instance_bce_with_logits(logits, labels, weights=None):
    assert logits.dim() == 2
    # loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, weights)
    weights = Variable(torch.FloatTensor([7])).cuda()
    loss = binary_cross_entropy_with_logits(logits, labels, weights)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logit, labels, q_id, pick_num):
    mask = pick_num[q_id,:] 
    score = labels.max(1)[0].unsqueeze(dim=1).repeat(1, logit.shape[1]) 
    predict = (logit > 0)
    targets = (labels > 0)
    scores = [((predict & targets).float()*score).sum(1), ((targets^predict).float() * score * mask).sum(1)]
    return scores

def f1_loss(y_true, y_pred): 
    tp = torch.sum((y_true*y_pred).float(),dim=0)
    tn = torch.sum(((1-y_true)*(1-y_pred)).float(),dim=0)
    fp = torch.sum(((1-y_true)*y_pred).float(), dim=0)
    fn = torch.sum((y_true*(1-y_pred)).float(), dim=0)    
    #percision与recall，这里的torch.epsilon代表一个小正数，用来避免分母为零
    eps = (torch.finfo(torch.float32).eps)
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    # import inputs; inputs.set_trace()
    f1 = 2*p*r / (p+r+eps)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)#其实就是把nan换成0
    return torch.mean(f1), torch.mean(p), torch.mean(r)
def compute_f1_score(logit, labels, q_id, pick_num):
    predict = (logit > 0).float()
    targets = (labels > 0).float() 
    pick_num = pick_num.int()
    f1_score = np.zeros([pick_num.shape[0], 3])
    for i in range(pick_num.shape[0]):
        ids = torch.where(q_id==i)[0]
        if len(ids) > 0:
            y_true = targets[ids, 0:pick_num[i].sum()]
            y_pred = predict[ids,0:pick_num[i].sum()]
            f1, p, r = f1_loss(y_true, y_pred)
            f1_score[i] = [f1.item(), p.item(), r.item()]
    return f1_score



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



def evaluate(model, dataloader, cache, pick_num):
    score = [0, 0]
    upper_bound = 0
    f1_score = torch.zeros([pick_num.shape[0], 3])
    ans2label_path = os.path.join( cache, 'trainval_ans2label.pkl')
    label2ans_path = os.path.join( cache, 'trainval_label2ans.pkl')
    ans2label = cPickle.load(open(ans2label_path, 'rb'))
    label2ans = cPickle.load(open(label2ans_path, 'rb'))
    data = None
    for v, b, q, a in iter(dataloader):
        with torch.no_grad():
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            pred = model(v, b, q, None)
            if data is None:
                data = [pred, a.cuda(), b]
            else:
                data = [torch.cat([p0, p1], dim=0) for p0, p1 in zip(data, [pred, a.cuda(), b])]
            batch_score = compute_score_with_logits(pred, a.cuda(), b, pick_num)
            score[0] += batch_score[0].sum()
            score[1] += batch_score[1].sum()
            upper_bound += a.sum()
            # # print(batch_score)
            # ans = show_answer(a,label2ans)
            # ans2 = show_answer(logits,label2ans)
        # for i in range(len(ans)):
        #     print("ques:",qt[i],"0:",ans[i],"1: ",ans2[i],ans2[i]==ans[i])
    f1_score = compute_f1_score(*data, pick_num)
    
    score = [s/len(dataloader.dataset) for s in score]
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound, f1_score
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



def train(model, train_loader,eval_loader,  cache, num_epochs, output, tag):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, '%s_log.txt'%tag))
    logger.write(tag)
    
    best_eval_score = 0

    dataroot = 'data'
    name = 'train'
    # question_path = os.path.join(cache, "%s_questions.json"%name)
    # questions_list = json.load(open(question_path))['questions']
    answer_path = os.path.join(cache, 'total_target.pkl')
    answers = cPickle.load(open(answer_path, 'rb'))
    # ans_id_np = get_id_np(answers, 'labels')

    # ans_id_torch = torch.from_numpy(ans_id_np)#.to(torch.int)
    pick_idx = model.pick_idx
    pick_num = (pick_idx>0).float()
    for epoch in range(num_epochs):
        total_loss = 0
        train_score = [0, 0]
        t = time.time()
        for i, (v, b, q, a) in enumerate(train_loader):  
            v = Variable(v).cuda()
            b = torch.squeeze(b)
            q = Variable(q).cuda() #quesiton
            a = Variable(a).cuda() #targets
            pred = model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data, b, pick_num)
            total_loss += loss.data * v.size(0)
            train_score[0] += batch_score[0].sum()
            train_score[1] += batch_score[1].sum()
            

        total_loss /= len(train_loader.dataset)
        train_score = [100 * s / len(train_loader.dataset) for s in train_score]
        
        model.train(False)
        eval_score, bound, f1_score = evaluate(model, eval_loader, cache, pick_num)
        model.train(True)

        logger.write('epoch %d, time: %.2f \ttrain_loss: %.2f, score: %.2f %.2f \teval score: %.2f %.2f (%.2f)' % (epoch, time.time()-t, total_loss, train_score[0],  train_score[1], 100 * eval_score[0], 100 * eval_score[1], 100 * bound))
        logger.write(str(f1_score))
        # logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        # logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        # logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score[0]-eval_score[1]/2 > best_eval_score:
            # dict_path = os.path.join(output, 'model_dict.pth')
            # torch.save(model.state_dict(), dict_path)
            model_path = os.path.join(output, '%smodel.pth'%tag)
            torch.save(model,model_path)
            best_eval_score = eval_score[0]-eval_score[1]/2





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
