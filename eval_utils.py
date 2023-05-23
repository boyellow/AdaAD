import torch
import torch.nn as nn


from autoattack import AutoAttack

from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack,\
                               LinfPGDAttack, LinfMomentumIterativeAttack, \
                               CarliniWagnerL2Attack, JacobianSaliencyMapAttack, ElasticNetL1Attack 


def test_autoattack(model, testloader, norm='Linf', eps=8/255, version='standard', verbose=True):
    
    adversary = AutoAttack(model, norm=norm, eps=eps, version=version, verbose=verbose)

    if version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
        adversary.apgd.n_restarts = 1
        adversary.apgd_targeted.n_restarts = 1

    x_test = [x for (x,y) in testloader]
    x_test = torch.cat(x_test, 0)
    y_test = [y for (x,y) in testloader]
    y_test = torch.cat(y_test, 0)


    with torch.no_grad():
        x_adv, y_adv = adversary.run_standard_evaluation(x_test, y_test, bs=testloader.batch_size, return_labels=True)

    adv_correct = torch.sum(y_adv==y_test).data
    total = y_test.shape[0]

    rob_acc = adv_correct / total
    print('Attack Strength:%.4f \t  AutoAttack Acc:%.3f (%d/%d)'%(eps, rob_acc, adv_correct, total))
    
                 
def test_robust(model, attack_type, c, num_classes, testloader, loss_fn=nn.CrossEntropyLoss(), req_count=10000, is_return=False):

    if (attack_type == "pgd"):
        adversary = LinfPGDAttack(
            model, loss_fn=loss_fn, eps=c,
            nb_iter=10, eps_iter=c/4, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False)
    elif (attack_type == "fgsm"):
        adversary = GradientSignAttack(
            model, loss_fn=loss_fn, eps=c,
            clip_min=0., clip_max=1., targeted=False)
    elif (attack_type == "mim"):
        adversary = LinfMomentumIterativeAttack(
            model, loss_fn=loss_fn, eps=c,
            nb_iter=40, eps_iter=c/10, clip_min=0., clip_max=1.,
            targeted=False)
    elif (attack_type == "bim"):
        adversary = LinfBasicIterativeAttack(
            model, loss_fn=loss_fn, eps=c,
            nb_iter=40, eps_iter=c/10, clip_min=0., clip_max=1.,
            targeted=False)
    elif (attack_type == "ela"):
        adversary = ElasticNetL1Attack(
            model, initial_const=c, confidence=0.1, max_iterations=100, clip_min=0., clip_max=1.,
            targeted=False, num_classes=10)
    elif (attack_type == "jsma"):
        adversary = JacobianSaliencyMapAttack(
            model, clip_min=0., clip_max=1., num_classes=10, gamma=c)
    elif (attack_type == "cw"):
        adversary = CarliniWagnerL2Attack(
                        model, confidence=0.01, max_iterations=1000, clip_min=0., clip_max=1., learning_rate=0.01,
                        targeted=False, num_classes=num_classes, binary_search_steps=1, initial_const=c)
        
    else:
        raise NotImplementedError

    ori_correct = 0
    adv_correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx < int(req_count/testloader.batch_size):
            inputs, targets = inputs.cuda(), targets.cuda()
            total += targets.size(0)

            ori_outputs = adversary.predict(inputs)
            ori_preds = ori_outputs.max(dim=1, keepdim=False)[1]
            ori_correct += ori_preds.eq(targets.data).cpu().sum()
            nat_acc = 100. * float(ori_correct) / total
            
            advs = adversary.perturb(inputs, targets)
            adv_outputs = adversary.predict(advs)
            adv_preds = adv_outputs.max(dim=1, keepdim=False)[1]
            adv_correct += adv_preds.eq(targets.data).cpu().sum()
            rob_acc = 100. * float(adv_correct) / total

    print('Attack Strength:%.4f \t Nat Acc:%.4f \t  %s Acc:%.3f (%d/%d)'%(c, nat_acc, attack_type, rob_acc, adv_correct, total))

    if is_return:
        return nat_acc, rob_acc
    

def test_transfer_robust(surrogate_model, target_model, attack_type, c, num_classes, testloader, loss_fn=nn.CrossEntropyLoss(), req_count=10000, is_return=False):

    if (attack_type == "pgd"):
        adversary = LinfPGDAttack(
            surrogate_model, loss_fn=loss_fn, eps=c,
            nb_iter=10, eps_iter=c/4, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False)
    elif (attack_type == "fgsm"):
        adversary = GradientSignAttack(
            surrogate_model, loss_fn=loss_fn, eps=c,
            clip_min=0., clip_max=1., targeted=False)
    elif (attack_type == "mim"):
        adversary = LinfMomentumIterativeAttack(
            surrogate_model, loss_fn=loss_fn, eps=c,
            nb_iter=40, eps_iter=c/10, clip_min=0., clip_max=1.,
            targeted=False)
    elif (attack_type == "bim"):
        adversary = LinfBasicIterativeAttack(
            surrogate_model, loss_fn=loss_fn, eps=c,
            nb_iter=40, eps_iter=c/10, clip_min=0., clip_max=1.,
            targeted=False)
    elif (attack_type == "ela"):
        adversary = ElasticNetL1Attack(
            surrogate_model, initial_const=c, confidence=0.1, max_iterations=100, clip_min=0., clip_max=1.,
            targeted=False, num_classes=10)
    elif (attack_type == "jsma"):
        adversary = JacobianSaliencyMapAttack(
            surrogate_model, clip_min=0., clip_max=1., num_classes=10, gamma=0.05, theta=c)
    elif (attack_type == "cw"):
        adversary = CarliniWagnerL2Attack(
                        surrogate_model, confidence=0.01, max_iterations=1000, clip_min=0., clip_max=1., learning_rate=0.01,
                        targeted=False, num_classes=num_classes, binary_search_steps=1, initial_const=c)
        
    else:
        raise NotImplementedError

    ori_correct = 0
    adv_correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx < int(req_count/testloader.batch_size):
            inputs, targets = inputs.cuda(), targets.cuda()
            total += targets.size(0)

            ori_outputs = target_model(inputs)
            ori_preds = ori_outputs.max(dim=1, keepdim=False)[1]
            ori_correct += ori_preds.eq(targets.data).cpu().sum()
            nat_acc = 100. * float(ori_correct) / total
            
            advs = adversary.perturb(inputs, targets)
            adv_outputs = target_model(advs.cuda())
            adv_preds = adv_outputs.max(dim=1, keepdim=False)[1]
            adv_correct += adv_preds.eq(targets.data).cpu().sum()
            rob_acc = 100. * float(adv_correct) / total

    print('Attack Strength:%.4f \t Nat Acc:%.4f \t  %s Acc:%.3f (%d/%d)'%(c, nat_acc, attack_type, rob_acc, adv_correct, total))

    if is_return:
        return nat_acc, rob_acc
    
