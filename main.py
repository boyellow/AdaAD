#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2022/07/12 20:49:21
@Author      :Bo Huang
@version      :1.0
'''

import os
import utils
import torch
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from config.dataset_config import getData
from networks.resnet import ResNet18, ResNet34
from networks.wideresnet import WideResNet, Yao_WideResNet
from networks.mobilenetv2 import MobileNetV2
from networks.vgg import VGG
from AT_helper import Madry_PGD, adaad_inner_loss
from advertorch.attacks import LinfPGDAttack


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Adversarial Training')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--method', type=str, default='Plain_Madry')

    parser.add_argument('--teacher_model', type=str, default='wideresnet34_10')
    parser.add_argument('--temp', default=30.0, type=float, help='temperature for distillation')

    # AdaAD options
    parser.add_argument('--AdaAD_alpha', default=1.0, type=float, help='AdaAD_alpha')

    # IAD options
    parser.add_argument('--IAD_begin', default=60, type=int, help='IAD_begin')
    parser.add_argument('--IAD_alpha', default=1.0, type=float, help='IAD_alpha')
    parser.add_argument('--IAD_beta', default=0.1, type=float, help='IAD_beta')

    # Inner optimization options
    parser.add_argument('--epsilon', type=int, default=16, help='perturbation bound')
    parser.add_argument('--num_steps', type=int, default=10, help='maximum perturbation step K')
    parser.add_argument('--step_size', type=int, default=4, help='step size')
    parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")

    # Is mixture of clean and adversarial loss
    parser.add_argument('--mixture', action='store_true')

    # Training options
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--optim', type=str, default='SGD', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lr_max', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_schedule', type=str, default='piecewise', choices=['cosine', 'piecewise', 'constant'], help='learning schedule')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay')

    # whether warm_up lr from 0 to max_lr in the first n epochs
    parser.add_argument('--warmup_lr', action='store_true')
    parser.add_argument('--warmup_lr_epochs', default=15, type=int)

    parser.add_argument('--gpu_id', type=str, default='0', choices=['0', '1', '2', '3', '4', '5', '6', '7'])
    parser.add_argument('--root_path', type=str, default='tuned_models', help='root path')

    parser.add_argument('--is_desc', action='store_true')
    parser.add_argument('--desc_str', type=str, default='', help='desc_str')

    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--seed', default=42, type=int, help='seed')

    args = parser.parse_args()
    return args


def get_auto_fname(args):
    names = args.method

    if args.method in ['Plain_Clean', 'Plain_Madry']:
        names = names + '-%s' % args.model

    if args.method == 'AdaAD_with_IAD1':
        names = names + '-T(%s)-S(%s)' % (args.teacher_model, args.model) + \
            '-temp(%s)-begin(%s)-alpha(%s)-beta(%s)' % (args.temp,
                                                        args.IAD_begin, args.IAD_alpha, args.IAD_beta)

    if args.method in ['AdaAD']:
        names = names + '-T(%s)-S(%s)' % (args.teacher_model, args.model) + '-alpha(%.4f)' % args.AdaAD_alpha

    if args.method != 'Plain_Clean':
        names = names + '-'.join(['-eps(%d)' % args.epsilon, 's_eps(%d)' %
                                 args.step_size, 'n_steps(%d)' % args.num_steps])

    names = names + '-'.join(['-epochs(%s)' % str(args.epochs), 'bs(%s)' % str(args.bs), 'optim(%s)' %
                             args.optim, 'lr_max(%s)' % str(args.lr_max), 'lr_sche(%s)' % args.lr_schedule])

    if args.warmup_lr:
        names = names + '-warmup(%s)' % str(args.warmup_lr_epoch)

    if args.is_desc:
        names = names + '-(%s)' % args.desc_str

    return names


def lr_decay(epoch, total_epoch):
    if args.lr_schedule == 'piecewise':
        if total_epoch == 200:
            epoch_point = [100, 150]
        elif total_epoch == 110: 
            epoch_point = [100, 105] # Early stop for Madry adversarial training
        else:
            raise ValueError
        if epoch < epoch_point[0]:
            if args.warmup_lr and epoch < args.warmup_lr_epoch:
                return 0.001 + epoch / args.warmup_lr_epoch * (args.lr_max-0.001)
            return args.lr_max
        if epoch < epoch_point[1]:
            return args.lr_max / 10
        else:
            return args.lr_max / 100
    elif args.lr_schedule == 'cosine':
        if args.warmup_lr:
            if epoch < args.warmup_lr_epoch:
                return 0.001 + epoch / args.warmup_lr_epoch * (args.lr_max-0.001)
            else:
                return np.max([args.lr_max * 0.5 * (1 + np.cos((epoch-args.warmup_lr_epoch) / (total_epoch-args.warmup_lr_epoch) * np.pi)), 1e-4])
        return np.max([args.lr_max * 0.5 * (1 + np.cos(epoch / total_epoch * np.pi)), 1e-4])
    elif args.lr_schedule == 'constant':
        return args.lr_max
    else:
        raise NotImplementedError


args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
# print('gpu_id:', args.gpu_id)

file_name = get_auto_fname(args)

save_path = os.path.join(args.root_path, args.dataset, args.method, file_name)

print('Save path:%s' % save_path)
if not os.path.isdir(save_path):
    os.makedirs(save_path)
results_log_csv_name = os.path.join(save_path, 'results.csv')
print('==> Preparing data..')

# setup data loader
num_classes, train_data, test_data = getData(args.dataset)

trainloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.bs,
    shuffle=True,
    num_workers=4,
    pin_memory=True)
testloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=400,
    shuffle=False,
    num_workers=4,
    pin_memory=True)

# Model
if args.model == 'mobilenetV2':
    net = MobileNetV2(num_classes=num_classes)
elif args.model == 'resnet18':
    net = ResNet18(num_classes)
elif args.model == 'resnet34':
    net = ResNet34(num_classes)
elif args.model == 'vgg16':
    net = VGG('VGG16')
elif args.model == 'wideresnet34_10':
    net = WideResNet(num_classes=num_classes)
elif args.model == 'wideresnet28_10':
    net = WideResNet(num_classes=num_classes, depth=28, widen_factor=10)
elif args.model == 'wideresnet70_16':
    net = WideResNet(num_classes=num_classes, depth=70, widen_factor=16)
else:
    raise NotImplementedError


use_cuda = torch.cuda.is_available()
print('use_cuda:%s' % str(use_cuda))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True
net.to(device)


if args.dataset == 'CIFAR10':
    if args.method not in ['Plain_Clean', 'Plain_Madry']:
        if args.teacher_model == 'wideresnet34_10':
            teacher_net = WideResNet(num_classes=num_classes)
            teacher_path = 'teacher_models/CIFAR10/wideresnet34_10/Madry-eps(0.031)-step_eps(0.008)-num_steps(10)-bs(128)-lr_max(0.1)-lr_sche(piecewise)-total_epochs(110)/best_PGD10_acc_model.pth'
            teacher_state_dict = torch.load(teacher_path)['net']

        elif args.teacher_model == 'Chen2021LTD_WRD34_20':
            teacher_net = Yao_WideResNet(
                num_classes=num_classes, depth=34, widen_factor=20, sub_block1=False)
            teacher_path = 'teacher_models/CIFAR10/wideresnet34_20/Chen2021LTD_34_20.pt'
            teacher_state_dict = torch.load(teacher_path)
            state_dict = OrderedDict()
            for k in list(teacher_state_dict.keys()):
                state_dict[k[7:]] = teacher_state_dict.pop(k)
            teacher_state_dict = state_dict

        else:
            raise NotImplementedError
        
        print('==> Loading teacher..')
        teacher_net.load_state_dict(teacher_state_dict)
        teacher_net.to(device)
        teacher_net.eval()

elif args.dataset == 'CIFAR100':
    if args.method not in ['Plain_Clean', 'Plain_Madry']:
        if args.teacher_model == 'Chen2021WRN34_10':
            teacher_net = Yao_WideResNet(
                num_classes=num_classes, depth=34, widen_factor=10, sub_block1=True)
            teacher_path = 'teacher_models/CIFAR100/wideresnet34_10/Chen2021LTD.pt'
            teacher_state_dict = torch.load(teacher_path)

        else:
            raise NotImplementedError
        
        print('==> Loading teacher..')
        teacher_net.load_state_dict(teacher_state_dict)
        teacher_net.to(device)
        teacher_net.eval()

else:
    raise NotImplementedError




if args.optim == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr_max,
                          momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optim == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr_max,
                           weight_decay=args.weight_decay)
else:
    raise NotImplementedError


# setup checkpoint
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(save_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(
        save_path, 'best_PGD10_acc_model.pth'))

    net.load_state_dict(checkpoint['net'])
    best_Test_acc = checkpoint['clean_acc']
    best_Test_PGD10_acc = checkpoint['PGD10_acc']
    best_Test_acc_epoch = checkpoint['epoch']
    start_epoch = checkpoint['epoch'] + 1

else:
    start_epoch = 0
    best_Test_acc = 0
    best_Test_Clean_acc_epoch = 0
    best_Test_PGD10_acc = 0
    best_Test_PGD10_acc_epoch = 0

    print('==> Preparing %s %s %s' % (args.model, args.dataset, args.method))
    print('==> Building model..')


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    global train_loss

    train_loss = 0
    correct_ori = 0
    total = 0

    net.train()

    lr_current = lr_decay(epoch, args.epochs)
    optimizer.param_groups[0].update(lr=lr_current)
    print('learning_rate: %s' % str(lr_current))

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if args.method == 'Plain_Clean':
            ori_outputs = net(inputs)
            loss = nn.CrossEntropyLoss()(ori_outputs, targets)
            optimizer.zero_grad()

        elif args.method == 'Plain_Madry':
            adv_inputs = Madry_PGD(net, inputs, targets, step_size=args.step_size/255,
                                   steps=args.num_steps, epsilon=args.epsilon/255)

            net.train()
            optimizer.zero_grad()

            adv_outputs = net(adv_inputs)
            adv_loss = nn.CrossEntropyLoss()(adv_outputs, targets)

            ori_outputs = net(inputs)
            ori_loss = nn.CrossEntropyLoss()(ori_outputs, targets)

            if args.mixture:
                loss = args.mixture_alpha*ori_loss + \
                    (1-args.mixture_alpha)*adv_loss
            else:
                loss = adv_loss

        elif args.method == 'AdaAD':
            adv_inputs = adaad_inner_loss(net, teacher_net, inputs, step_size=args.step_size/255,
                                          steps=args.num_steps, epsilon=args.epsilon/255)

            net.train()
            optimizer.zero_grad()

            ori_outputs = net(inputs)
            adv_outputs = net(adv_inputs)

            with torch.no_grad():
                teacher_net.eval()
                t_ori_outputs = teacher_net(inputs)
                t_adv_outputs = teacher_net(adv_inputs)

            if args.dataset == 'CIFAR10':
                kl_loss1 = nn.KLDivLoss()(F.log_softmax(adv_outputs, dim=1),
                                          F.softmax(t_adv_outputs.detach(), dim=1))
                kl_loss2 = nn.KLDivLoss()(F.log_softmax(ori_outputs, dim=1),
                                          F.softmax(t_ori_outputs.detach(), dim=1))

            if args.dataset == 'CIFAR100':
                kl_loss1 = (1/len(adv_outputs))*torch.sum(nn.KLDivLoss(reduce=False)(
                    F.log_softmax(adv_outputs, dim=1), F.softmax(t_adv_outputs.detach(), dim=1)))
                kl_loss2 = (1/len(adv_outputs))*torch.sum(nn.KLDivLoss(reduce=False)(
                    F.log_softmax(ori_outputs, dim=1), F.softmax(t_ori_outputs.detach(), dim=1)))

            loss = args.AdaAD_alpha*kl_loss1 + (1-args.AdaAD_alpha)*kl_loss2

        elif args.method == 'AdaAD_with_IAD1':
            optimizer.zero_grad()
            adv_inputs = adaad_inner_loss(net, teacher_net, inputs, step_size=args.step_size/255,
                                          steps=args.num_steps, epsilon=args.epsilon/255)
            net.train()
            ori_outputs = net(inputs)
            adv_outputs = net(adv_inputs)
            Alpha = torch.ones(len(inputs)).cuda()

            # basicop = net(adv_inputs).detach()
            guide = teacher_net(adv_inputs)
            teacher_outputs = teacher_net(adv_inputs)

            KL_loss = nn.KLDivLoss(reduce=False)
            XENT_loss = nn.CrossEntropyLoss()

            if epoch >= args.IAD_begin:
                for pp in range(len(adv_outputs)):

                    L = F.softmax(guide, dim=1)[pp][targets[pp].item()]
                    L = L.pow(args.IAD_beta).item()
                    Alpha[pp] = L
                loss = args.IAD_alpha*args.temp*args.temp*(1/len(adv_outputs))*torch.sum(KL_loss(F.log_softmax(adv_outputs/args.temp, dim=1), F.softmax(teacher_outputs/args.temp, dim=1)).sum(dim=1)) + args.IAD_alpha*(
                    1/len(adv_outputs))*torch.sum(KL_loss(F.log_softmax(adv_outputs, dim=1), F.softmax(net(inputs), dim=1)).sum(dim=1).mul(1-Alpha))+(1.0-args.IAD_alpha)*XENT_loss(net(inputs), targets)
            else:
                loss = args.IAD_alpha*args.temp*args.temp*(1/len(adv_outputs))*torch.sum(KL_loss(F.log_softmax(adv_outputs/args.temp, dim=1), F.softmax(
                    teacher_outputs/args.temp, dim=1)).sum(dim=1))+(1.0-args.IAD_alpha)*XENT_loss(net(inputs), targets)

        else:
            raise NotImplementedError

        loss.backward()
        optimizer.step()

        train_loss += loss.data

        correct_ori += torch.max(ori_outputs, 1)[1].eq(targets.data).cpu().sum()
        total += targets.size(0)

        utils.progress_bar(
            batch_idx,
            len(trainloader),
            'Total_Loss: %.3f| Clean Acc: %.3f%%(%d/%d)'
            '' % (train_loss / (batch_idx + 1),
                  100. * float(correct_ori) / total,
                  correct_ori,
                  total))

    Train_acc = 100. * float(correct_ori) / total


def test(epoch):
    global Test_acc
    global best_Test_acc
    global best_Test_acc_epoch
    global Test_PGD10_acc
    global best_Test_PGD10_acc
    global best_Test_PGD10_acc_epoch
    global test_loss

    test_loss = 0
    correct_ori = 0
    correct_PGD10 = 0
    correct_total = 0
    total = 0

    net.eval()

    adversary = LinfPGDAttack(net, loss_fn=nn.CrossEntropyLoss(),
                              eps=8/255, nb_iter=10, eps_iter=2/255, rand_init=True, clip_min=0., clip_max=1., targeted=False)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        ori_outputs = net(inputs)
        loss = nn.CrossEntropyLoss()(ori_outputs, targets)

        test_loss += loss.data

        correct_ori += torch.max(ori_outputs,
                                 1)[1].eq(targets.data).cpu().sum()
        total += targets.size(0)

        adv_PGD10 = adversary.perturb(inputs, targets)

        adv_PGD10_outputs = net(adv_PGD10)

        correct_PGD10 += torch.max(adv_PGD10_outputs,
                                   1)[1].eq(targets.data).cpu().sum()
        correct_total = total

        utils.progress_bar(
            batch_idx,
            len(testloader),
            'Total_Loss: %.3f| Clean Acc: %.3f%%|(%d/%d)| PGD10 Acc: %.3f%%|(%d/%d)'
            '' % (test_loss / (batch_idx + 1),
                  100. * float(correct_ori) / total,
                  correct_ori,
                  total,
                  100. * float(correct_PGD10) / correct_total,
                  correct_PGD10,
                  correct_total))

    # Save checkpoint.
    Test_acc = 100. * float(correct_ori) / total
    Test_PGD10_acc = 100. * float(correct_PGD10) / correct_total

    if Test_acc > best_Test_acc:
        print('Saving..')
        print("best_Test_clean_acc: %0.3f, \tits Test_PGD10_acc: %0.3f" %
              (Test_acc, Test_PGD10_acc))
        state = {
            'net': net.state_dict() if use_cuda else net,
            'clean_acc': Test_acc,
            'PGD10_acc': Test_PGD10_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, os.path.join(save_path, 'best_clean_acc_model.pth'))
        best_Test_acc = Test_acc
        best_Test_acc_epoch = epoch

    if Test_PGD10_acc > best_Test_PGD10_acc:
        print('Saving..')
        print("best_Test_PGD10_acc: %0.3f, \tits Test_clean_acc: %0.3f" %
              (Test_PGD10_acc, Test_acc))
        state = {
            'net': net.state_dict() if use_cuda else net,
            'clean_acc': Test_acc,
            'PGD10_acc': Test_PGD10_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, os.path.join(save_path, 'best_PGD10_acc_model.pth'))
        best_Test_PGD10_acc = Test_PGD10_acc
        best_Test_PGD10_acc_epoch = epoch

    if epoch == args.epochs - 1:
        print('Saving..')
        state = {
            'net': net.state_dict() if use_cuda else net,
            'clean_acc': Test_acc,
            'PGD10_acc': Test_PGD10_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, os.path.join(
            save_path, 'last_epoch_model(%s).pth' % epoch))


def main():

    # record train log
    with open(results_log_csv_name, 'w') as f:
        f.write(
            'epoch, train_loss, test_loss, train_acc, test_clean_acc, test_PGD10_acc, time\n')

    # start train
    for epoch in range(start_epoch, args.epochs):
        print('current time:', datetime.now().strftime('%b%d-%H:%M:%S'))
        train(epoch)
        test(epoch)
        # Log results
        with open(results_log_csv_name, 'a') as f:
            f.write('%5d, %.5f, %.5f, %.5f, %.5f, %.5f, %s,\n'
                    '' % (epoch,
                          train_loss,
                          test_loss,
                          Train_acc,
                          Test_acc,
                          Test_PGD10_acc,
                          datetime.now().strftime('%b%d-%H:%M:%S')))

    print("best_Test_Clean_acc: %.3f" % best_Test_acc)
    print("best_Test_Clean_acc_epoch: %d" % best_Test_acc_epoch)
    print("best_Test_PGD10_acc: %.3f" % best_Test_PGD10_acc)
    print("best_Test_PGD10_acc_epoch: %d" % best_Test_PGD10_acc_epoch)

    # best ACC
    with open(results_log_csv_name, 'a') as f:
        f.write('%s,%03d,%0.3f,%s,%03d,%0.3f,\n' % ('best clean acc (test)',
                                                    best_Test_acc_epoch,
                                                    best_Test_acc,
                                                    'best PGD10 acc (test)',
                                                    best_Test_PGD10_acc_epoch,
                                                    best_Test_PGD10_acc))


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main()
