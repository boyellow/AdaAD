# Boosting Accuracy and Robustness of Student Models via Adaptive Adversarial Distillation 

Code for the paper [Boosting Accuracy and Robustness of Student Models via Adaptive Adversarial Distillation](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Boosting_Accuracy_and_Robustness_of_Student_Models_via_Adaptive_Adversarial_CVPR_2023_paper.html) (CVPR 2023).



## Environment settings and libraries we used in our experiments

The codes are evaluated under the following environment settings and libraries:
- OS: Ubuntu
- GPU: NVIDIA GTX3090
- Cuda: 11.1, Cudnn: v8.2
- Python: 3.9
- PyTorch: 1.8.1
- Torchvision: 0.9.0
- AutoAttack
- advertorch

## Acknowledgement
The teacher model [WideResNet-34-20](https://arxiv.org/abs/2111.02331) and [WideResNet-34-10](https://arxiv.org/abs/2111.02331) are used on CIFAR10 and CIFAR100, respectively, in our experiments.

## Requirements
- Install or download [AutoAttack](https://github.com/fra31/auto-attack):
```.bash
pip install git+https://github.com/fra31/auto-attack
```

- Install or download [advertorch](https://github.com/BorealisAI/advertorch):
```.bash
pip install advertorch
```

## Training Commands
Run [`main.py`] for reproducing the experimental results reported in the paper. For example, train a ResNet-18 model on CIFAR10 via [PGD-AT](https://arxiv.org/abs/1706.06083) with [early stopping](https://arxiv.org/abs/2002.11569) strategy. Run the command:

```.bash
python main.py \
    --dataset CIFAR10 \
    --model resnet18 \
    --method Plain_Madry \
    --epsilon 8 \
    --num_steps 10 \
    --step_size 2 \
    --epochs 110 \
    --bs 128 \
    --lr_max 0.1 \
    --lr_schedule piecewise \
    --gpu_id 0
```

Train a ResNet-18 model on CIFAR10 via the proposed **AdaAD** in the paper by using the teacher model [WideResNet-34-20](https://arxiv.org/abs/2111.02331), run the command:
```.bash
python main.py \
    --dataset CIFAR10 \
    --model resnet18 \
    --method AdaAD \
    --teacher_model Chen2021LTD_WRD34_20 \
    --epsilon 8 \
    --num_steps 10 \
    --step_size 2 \
    --epochs 200 \
    --bs 128 \
    --lr_max 0.1 \
    --lr_schedule piecewise \
    --gpu_id 0
```

The proposed **AdaAD** allows a larger search radius in the inner optimization to achieve better robustness performance, as reported in Section 3.4 and 4.2. Train a ResNet-18 model on CIFAR10 with a larger search radius via the proposed **AdaAD**, run the command:
```.bash
python main.py \
    --dataset CIFAR10 \
    --model resnet18 \
    --method AdaAD \
    --teacher_model Chen2021LTD_WRD34_20 \
    --epsilon 16 \
    --num_steps 10 \
    --step_size 4 \
    --epochs 200 \
    --bs 128 \
    --lr_max 0.1 \
    --lr_schedule piecewise \
    --gpu_id 0
```

Considering that the teacher model may be unreliable on some points, the proposed **AdaIAD** by naturally combining **AdaAD** with [IAD](https://arxiv.org/abs/2106.04928) is to make the distillation process more reliable. Train a ResNet-18 model on CIFAR10 via the proposed AdaIAD in the paper, run the command:
```.bash
python main.py \
    --dataset CIFAR10 \
    --model resnet18 \
    --method AdaAD_with_IAD1 \
    --teacher_model Chen2021LTD_WRD34_20 \
    --epsilon 8 \
    --num_steps 10 \
    --step_size 2 \
    --epochs 200 \
    --bs 128 \
    --lr_max 0.1 \
    --lr_schedule piecewise \
    --gpu_id 0
```

## References
If you find the codes useful for your research, please consider citing
```bib
@InProceedings{Huang_2023_CVPR,
    author    = {Huang, Bo and Chen, Mingyang and Wang, Yi and Lu, Junda and Cheng, Minhao and Wang, Wei},
    title     = {Boosting Accuracy and Robustness of Student Models via Adaptive Adversarial Distillation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {24668-24677}
}
```
