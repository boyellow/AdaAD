'''Some helper functions for PyTorch, including:
    - progress_bar: progress bar mimic xlua.progress.
    - set_lr : set the learning rate
    - clip_gradient : clip gradient
'''

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function

import torch.nn.functional as F
from torch.autograd import Variable

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        #print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def model_predict(model, X, batch_size=200):
    # use_cuda = torch.cuda.is_available()
    use_cuda = True
    steps = math.ceil(X.shape[0]/batch_size)
    y_probs = []
    for i in range(steps):
        if i != steps - 1:
            batch_X = X[i*batch_size:(i+1)*batch_size]
        else:
            batch_X = X[i*batch_size:]
        if use_cuda:
            model = model.cuda()
            model.eval()
            batch_X = torch.tensor(batch_X).float()
            batch_X = batch_X.permute(0, 3, 1, 2)
            inputs = Variable(batch_X, volatile=True).cuda()
        batch_y_logits = model(inputs)
        batch_y_probs = F.softmax(batch_y_logits, dim=1).cpu().detach().numpy()
        y_probs.append(batch_y_probs)
    y_probs = np.concatenate(y_probs, axis=0)
    return y_probs


