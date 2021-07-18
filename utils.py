"""
Utilities functions

Created by Kunhong Yu
Date: 2021/07/16
"""

import torch as t
import torchvision as tv
from models import LeNet5, LeNet5Half, AlexNet, AlexNetHalf
import matplotlib.pyplot as plt
from torch.nn import functional as F
import os


def get_data_loader(data_root : str, dataset : str, mode : str, batch_size : int, transform : tv.transforms.Compose) -> t.utils.data.DataLoader:
    """This function is used to get data loader
    Args :
        --data_root: data root
        --dataset: data set's name
        --mode: 'train' else 'test'
        --batch_size
        --transform: tv.transforms.Compose instance
    return :
        --dataloader: t.utils.data.DataLoader instance
    """
    if dataset == 'mnist':
        dataset = tv.datasets.MNIST(root = data_root,
                                    download = True,
                                    train = True if mode == 'train' else False,
                                    transform = transform)

    elif dataset == 'fashion_mnist':
        dataset = tv.datasets.FashionMNIST(root = data_root,
                                           download = True,
                                           train = True if mode == 'train' else False,
                                           transform = transform)

    elif dataset == 'cifar10':
        dataset = tv.datasets.CIFAR10(root = data_root,
                                      download = True,
                                      train = True if mode == 'train' else False,
                                      transform = transform)

    else:
        raise Exception('No other data sets!')

    dataloader = t.utils.data.DataLoader(dataset,
                                         batch_size = batch_size,
                                         shuffle = True if mode == 'train' else False,
                                         drop_last = False)

    return dataloader


def get_models(model_name : str, distill = False, baseline = False):
    """Get model instances and calculate num of params
    Args :
        --model_name: model's name
        --distill: default is False
        --baseline: default is False
    return :
        --t_model: teacher model
        --s_model: student model
    """
    s_model = None
    if model_name == 'lenet5':
        t_model = LeNet5()
        print('LeNet5 model : \n', t_model)
        total = sum(p.numel() for p in t_model.parameters())
        print("Total params for LeNet5: %.2fM" % (total / 1e6))

        if distill or baseline:
            s_model = LeNet5Half()
            print('LeNet5Half model : \n', s_model)
            total = sum(p.numel() for p in s_model.parameters())
            print("Total params for LeNet5Half: %.2fM" % (total / 1e6))

    elif model_name == 'alexnet':
        t_model = AlexNet()
        print('AlexNet model : \n', t_model)
        total = sum(p.numel() for p in t_model.parameters())
        print("Total params for AlexNet: %.2fM" % (total / 1e6))

        if distill or baseline:
            s_model = AlexNetHalf()
            print('AlexNetHalf model : \n', s_model)
            total = sum(p.numel() for p in s_model.parameters())
            print("Total params for AlexNetHalf: %.2fM" % (total / 1e6))

    else:
        raise Exception('No other models!')

    return t_model, s_model


##################
#      FGSM      #
##################
def fgsm(x, y, model, device, epsilon : float, index : int, dataset : str, distill : bool, baseline : bool, distill_type : str):
    """Generate adversarial example using FGSM
    Args :
        --x: input tensor
        --y: output tensor
        --model: trained model instance
        --device
        --epsilon
        --index: batch index
        --dataset
        --distill : bool
        --baseline : bool
        --distill_type : str
    return :
        --acc
    """
    x.requires_grad = True
    base_cost = t.nn.CrossEntropyLoss().to(device)
    real_output = model(x)
    probs = F.softmax(real_output, dim = 1)
    y_ = F.one_hot(y, 10)
    normal_max_probs = t.sum(probs * y_, dim = 1)

    # main code of FGSM
    ce_cost = base_cost(real_output, y)
    ce_cost.backward()
    gradient_sign = t.sign(x.grad)
    x.requires_grad = False
    x_ = x.detach().data + epsilon * gradient_sign.data # FGSM
    if dataset == 'cifar10':
        x_ = t.clamp(x_, -1, 1)
    else:
        x_ = t.clamp(x_, 0, 1)
    fake_output = model(x_)
    probs = F.softmax(fake_output, dim = 1)
    indices = t.argmax(probs, dim = 1, keepdim = False)
    # indices is adv predictions
    num_examples = x.size(0)
    correct = t.sum(indices == y).float()
    acc = correct / num_examples

    fake_max_probs = probs[0, indices]

    # We choose 1 result to display
    gt = y[0]
    normal_max_prob = normal_max_probs[0]
    adv_index = indices[0]
    adv_max_prob = fake_max_probs[0]

    f, ax = plt.subplots(1, 3, figsize = (15, 5))
    X = x[0].data.cpu().numpy()
    gs = gradient_sign[0].data.cpu().numpy()
    x_ = x_[0].data.cpu().numpy()
    if dataset == 'mnist' or dataset == 'fashion_mnist':
        X *= 255.
        x_ *= 255.
    elif dataset == 'cifar10':
        X = X.transpose(1, 2, 0)
        X = X * 0.5 + 0.5
        X *= 255.
        x_ = x_.transpose(1, 2, 0)
        x_ = x_ * 0.5 + 0.5
        x_ *= 255.
        gs = gs.transpose(1, 2, 0)

    ax[0].imshow(X.astype('uint8'))
    ax[0].set_title(
        'gt : ' + str(gt.item()) + '\n' + 'prob : ' + str(round(normal_max_prob.data.cpu().numpy() * 100., 2)) + '%')
    ax[0].grid(False)
    ax[0].axis('off')

    ax[1].imshow(gs.astype('uint8'))
    ax[1].set_title('Gradient sign')
    ax[1].grid(False)
    ax[1].axis('off')

    ax[2].imshow(x_.astype('uint8'))
    ax[2].set_title('adv preds : ' + str(adv_index.item()) + '\n' + 'prob : ' + str(
        round(adv_max_prob.data.cpu().numpy() * 100., 2)) + '%')
    ax[2].grid(False)
    ax[2].axis('off')

    if not distill:
        sign = 't' if not baseline else 's'
        filename = f'{sign}_model_{dataset}_FGSM_{epsilon}_{index + 1}.png'
    else:
        filename = f's_model_{dataset}_{distill_type}_FGSM_{epsilon}_{index + 1}.png'
    dir = os.path.join('./results/', f'fgsm_{dataset}')
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(os.path.join(dir, filename))
    plt.close()

    return acc


def repeat_error_verfication(t_preds, s_preds, y):
    """This function is used to verify in Experiment 4.7 in the paper
    Args :
        --t_pred: teacher preds
        --s_pred: student preds
        --y: ground-truth label
    return :
        --mu_s
        --mu_f
    """
    t_preds_correct = (t_preds == y).unsqueeze(dim = -1)
    t_preds_incorrect = (t_preds != y).unsqueeze(dim = -1)
    s_preds_correct = (s_preds == y).unsqueeze(dim = -1)
    s_preds_incorrect = (s_preds != y).unsqueeze(dim = -1)

    W_T_C_S = t_preds_incorrect * s_preds_correct
    numerator = t.sum(W_T_C_S.squeeze())
    mu_s = numerator / (t.sum(t_preds_incorrect.squeeze()) + 1e-8)

    C_T_W_S = t_preds_correct * s_preds_incorrect
    numerator = t.sum(C_T_W_S.squeeze())
    mu_f = t.log(numerator + 1e-8) - t.log(t.sum(t_preds_correct.squeeze()) + 1e-8) # we use logrithm here to avoid data underflow

    return mu_s, mu_f