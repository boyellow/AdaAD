from config.dataset_config import getData
from eval_utils import test_autoattack, test_robust
from networks.mobilenetv2 import MobileNetV2
from networks.wideresnet import WideResNet
from networks.resnet import ResNet18
import torch.backends.cudnn as cudnn
import argparse
import torch
import torch.nn as nn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--method', type=str, default='Plain_Madry')
parser.add_argument('--teacher_model', type=str, default='teacher_model')
parser.add_argument('--bs', default=200, type=int)
parser.add_argument('--eps', default=8/255, type=float)
parser.add_argument('--steps', default=10, type=int)
parser.add_argument('--random-start', default=1, type=int)
parser.add_argument('--coeff', default=0.1, type=float)  # for jsma, cw, ela
args = parser.parse_args()


num_classes, train_data, test_data = getData(args.dataset)

trainloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.bs,
    shuffle=True,
    num_workers=4,
    pin_memory=True)
testloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.bs,
    shuffle=False,
    num_workers=4,
    pin_memory=True)


# Model
if args.model == 'mobilenetV2':
    net = MobileNetV2(num_classes=num_classes)
elif args.model == 'resnet18':
    net = ResNet18(num_classes)
elif args.model == 'wideresnet34_10':
    net = WideResNet(num_classes=num_classes)
else:
    raise NotImplementedError


use_cuda = torch.cuda.is_available()
print('use_cuda:%s' % str(use_cuda))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True

model_path = ''  # The model path you want to evaluate
print('model path:', model_path)

net.load_state_dict(torch.load(model_path)['net'])
net.to(device)
net.eval()

print(args.model)
if args.method not in ['Plain_Madry', 'TRADES']:
    print(args.teacher_model)

print('Evaluate FGSM:')
test_robust(net, attack_type='fgsm', c=args.eps, num_classes=num_classes,
            testloader=testloader, loss_fn=nn.CrossEntropyLoss(), req_count=10000)
print('Evaluate PGD:')
test_robust(net, attack_type='pgd', c=args.eps, num_classes=num_classes,
            testloader=testloader, loss_fn=nn.CrossEntropyLoss(), req_count=10000)
print('Evaluate CW:')
test_robust(net, attack_type='cw', c=args.coeff, num_classes=num_classes,
            testloader=testloader, loss_fn=nn.CrossEntropyLoss(), req_count=10000)
print('Evaluate AA:')
test_autoattack(net, testloader, norm='Linf', eps=args.eps,
                version='standard', verbose=False)
