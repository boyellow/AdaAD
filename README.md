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
- advertorch
- autoattack

## Requirements
- Install or download [AutoAttack](https://github.com/fra31/auto-attack):
```.bash
pip install git+https://github.com/fra31/auto-attack
```

- Install or download [advertorch](https://github.com/BorealisAI/advertorch):
```.bash
pip install advertorch
```
