"""
Configurations

Created by Kunhong Yu
Date: 2021/07/16
"""

import torchvision as tv
import os

class Config(object):
    """
    Args :
        --data_root: default is './data/'
        --dataset: data set name, default is 'cifar10', 'mnist', 'fashion_mnist'

        --distill: True for distillation False then 'train_s' will work
        --baseline: False for training student solely
        --temperature: default is 1.
        --lambd: hyperparameter, default is 0.3
        --alpha: hyperparameter in self-regulation, default is 0.01
        --distill_type: 'kd' f32or Hinton's method, 'cckd_t'/'cckd_l'/'cckd_t_reg'

        --batch_size: default is 32
        --epochs: default is 160
        --optimizer: default is 'adamw', also support 'adam'/'momentum'
        --init_lr: default is 1e-5
        --gamma: learning rate decay rate, default is 0.2
        --milestones: we use steplr decay, default is [30, 60, 90, 120, 150]
        --weight_decay: default is 1e-5

        --fgsm: for adversarial attack, default is False
        --epsilon: step size for FGSM
        --repeat_error_ver: repeat error verification in Experiment 4.6 in the paper?
            default is False
    """

    ############
    #    Data  #
    ############
    data_root = './data/'
    dataset = 'cifar10'

    ############
    #   Model  #
    ############
    distill = False
    baseline = False
    temperature = 1.
    lambd = 0.3
    alpha = 0.01
    distill_type = 'kd'

    ############
    #   Train  #
    ############
    batch_size = 32
    epochs = 160
    optimizer = 'adam'
    init_lr = 1e-3
    gamma = 0.2
    milestones = [30, 60, 90, 120, 150]
    weight_decay = 1e-5

    ############
    #   Test  #
    ############
    fgsm = False
    epsilon = 0.001
    repeat_error_ver = False

    def parse(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                print(k + 'does not exist, will be added!')

            setattr(self, k, v)

        if getattr(self, 'dataset') == 'cifar10':
            setattr(self, 'model_name', 'alexnet')
            self.train_transform = tv.transforms.Compose([
                tv.transforms.RandomCrop(32, padding = 4),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.RandomRotation(15),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(0.5, 0.5)
            ])

            self.test_transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(0.5, 0.5)
            ])

        elif getattr(self, 'dataset').count('mnist'):
            setattr(self, 'model_name', 'lenet5')
            self.train_transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
            ])

            self.test_transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
            ])

        else:
            raise Exception('You may add new transform in config.py file!')

        if getattr(self, 'distill'): # Teacher model must be pretrained and stored for distillation!
            assert os.path.exists(os.path.join('checkpoints', f"t_model_{getattr(self, 'dataset')}.pth"))


    def print(self):
        for k, _ in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, '...', getattr(self, k))