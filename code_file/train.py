import os
import time
import torch
import torch.nn as nn
import utils
import numpy as np
from torch.autograd import Variable


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1]
    pred_y = logits.data.cpu().numpy().squeeze()
    target_y = labels
    scores = sum(pred_y==target_y)
    return scores

def load_lossfunc(model):
    if model.model_name=='FrameQA':
        loss_func = torch.nn.CrossEntropyLoss()
    elif model.model_name=='Count':
        loss_func = torch.nn.MSELoss()
    elif model.model_name=='Trans' or model.model_name=='Action':
        loss_func = torch.nn.MultiMarginLoss()
    else:
        raise ValueError('Unknown task.')
    return loss_func

def train(model, train_loader, eval_loader, num_epochs, output, max_len=35):
    utils.create_dir(output % (model.model_name))
    optim = torch.optim.Adamax(model.parameters())  #SGD(, lr=0.05, momentum=0.9)
    logger = utils.Logger(os.path.join(output % (model.model_name), 'log.txt'))
    best_test_score = 0
    if model.model_name=='Count':
        best_eval_score = 99999999

    else:
        best_eval_score = 0
    loss_func = load_lossfunc(model)

    for epoch in range(num_epochs):
        total_loss = 0
        t = time.time()
        import json
        print('======================Epoch %d \'s train========================='% (epoch))
        for i, (v, q_w, q_c, a) in enumerate(train_loader):
            # reshape multi-choice question: batch_size x num_choide x max_length ==> batch_size*num_choide x max_length
            q_w = np.array(q_w)
            q_w = torch.from_numpy(q_w.reshape(-1,q_w.shape[-1]))
            q_c = np.array(q_c)
            q_c = torch.from_numpy(q_c.reshape(-1, q_c.shape[-2], q_c.shape[-1]))
            q_c = Variable(q_c.cuda())
            v = np.array(v)
            v = np.tile(v, [1, model.num_choice]).reshape(-1,v.shape[-2], v.shape[-1])
            v = Variable(torch.from_numpy(v).cuda())
            q_w = Variable(q_w.cuda())
            a = a.type(torch.LongTensor)
            if model.model_name=='Count':
                a = Variable(a.cuda()).float()
            else:
                a = Variable(a.cuda())

            pred = model(v, q_w, q_c, a)
            loss = loss_func(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            total_loss += loss.item() * v.size(0)


        total_loss /= len(train_loader.dataset)
        logger.write('\ttrain_loss: %.2f' % (total_loss))
        model.train(False)
        print('===========Epoch %d \'s val==========' % (epoch))
        # val_score = model.evaluate(val_loader)
        # logger.write('\t val score: %.2f ' % (100 * val_score))
        print('===========Epoch %d \'s test==========' % (epoch))
        eval_score= model.evaluate(eval_loader)
        model.train(True)

        logger.write('\tcosting time: %.2f' % (time.time()-t))
        logger.write('\teval score: %.2f ' % (eval_score * 100))
        if model.model_name=='Count':
            if eval_score < best_eval_score:
                model_path = os.path.join(output % (model.model_name), 'model.pth')
                torch.save(model.state_dict(), model_path)
                best_eval_score = eval_score
                best_test_score = eval_score
            logger.write('\tcurrent best eval score: %.2f ' % (best_test_score))
        else:
            if eval_score > best_eval_score:
                model_path = os.path.join(output % (model.model_name), 'model.pth')
                torch.save(model.state_dict(), model_path)
                best_eval_score = eval_score
                best_test_score = eval_score
            logger.write('\tcurrent best eval score: %.2f ' % (100 * best_test_score))
    logger.write('\tfinal best eval score: %.2f ' % (100 * best_eval_score))


