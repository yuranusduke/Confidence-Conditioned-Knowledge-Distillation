"""
Train models for teachers and students

Created by Kunhong Yu
Date: 2021/07/15
"""

import torch as t
from utils import get_data_loader, get_models
from config import Config
from tqdm import tqdm
from torch.nn import functional as F
from torch.nn.functional import one_hot
import math
import os
import matplotlib.pyplot as plt

opt = Config()

def train_with_cckd(batch_x, batch_y, model, cls_cost, alpha : float, epoch : int,
                    distill : bool, num_classes : int, t_model = None, distill_type = 'kd',
                    temperature = 1., lambd = None):
    """Core code in training with CCKD or not
    Args :
        --batch_x: input tensor
        --batch_Y: target tensor
        --model: model to be trained
        --alpha: hyperparameter in self-regulation
        --epoch: current epoch
        --cls_cost: cls_cost instance
        --distill: default is False
        --num_classes
        --t_model: default is None
        --distill_type: 'kd' f32or Hinton's method, 'cckd_t'/'cckd_l'/'cckd_t_reg'
        --use_reg: use self-regulation
        --temperature: default is 1
        --lambd: hyperparameter, default is None
    return :
        --cls_i_cost: for baseline training
        --cls_i_loss, distill_loss, loss: for 'kd' and 'cckd_l'
        --loss: for 'cckd_t' and 'cckd_t_reg'
        --out
        --s_out
    """
    def reg_condition(y_hat):
        """This inner function is used to get self-regulation condition
        Args :
            --y_hat: logits prediction from student model
        return :
            --incorrect_indices
            --reg_indices
        """
        preds = t.argmax(y_hat, dim = -1)
        correct_indices = (preds == batch_y)
        incorrect_indices = (preds != batch_y)

        # correct_y_hat = y_hat[correct_indices] # [m, 10]
        res, _ = y_hat.topk(2, 1, True, True)
        delta = res[:, 0 : 1] - res[:, 1 : 2] # [m, 1]
        f_n = 1 - math.exp(-alpha * epoch)

        reg_indices = (delta < f_n) * correct_indices.unsqueeze(dim = -1)

        return incorrect_indices, reg_indices.squeeze()

    if not distill: # train student model baseline or train teacher model solely
        out = model(batch_x)
        cls_i_cost = cls_cost(out, batch_y)

        return cls_i_cost, out

    else: # distill
        assert t_model is not None
        t_out = t_model(batch_x).detach()
        s_out = model(batch_x)
        cls_i_cost = cls_cost(s_out, batch_y)
        if distill_type == 'kd':
            distill_loss = -t.mean(t.sum(F.softmax(t_out / temperature, dim = -1) * t.log(F.softmax(s_out / temperature, dim = -1)), dim = -1))

            loss = distill_loss + cls_i_cost * lambd

            return cls_i_cost, distill_loss, loss, s_out

        else: # CCKD
            if distill_type == 'cckd_l':
                lambd = t.sum(F.softmax(t_out, dim = -1) * one_hot(batch_y, num_classes), dim = -1, keepdim = True) # [m, 1]
                distill_loss = -t.sum(F.softmax(t_out / temperature, dim = -1) * t.log(F.softmax(s_out / temperature, dim = -1)), dim = -1)
                loss = t.mean(lambd * distill_loss + (1 - lambd) * cls_i_cost)

                return cls_i_cost, t.mean(distill_loss), loss, s_out

            elif distill_type == 'cckd_t':
                lambd = t.sum(F.softmax(t_out, dim = -1) * one_hot(batch_y, num_classes), dim = -1, keepdim = True)  # [m, 1]
                y_c = lambd * F.softmax(t_out / temperature, dim = -1) + (1 - lambd) * one_hot(batch_y, num_classes) # [m, 10]
                y_c = y_c / (t.norm(y_c, p = 1, dim = -1, keepdim = True) + 1e-8) # norm to keep distribution
                loss = -t.mean(t.sum(y_c * t.log(F.softmax(s_out / temperature, dim = -1)), dim = -1))

                return loss, s_out

            elif distill_type == 'cckd_t_reg':
                incorrect_indices, reg_indices = reg_condition(F.softmax(s_out / temperature, dim = -1))
                temp_s_out = t.cat((s_out[incorrect_indices], s_out[reg_indices]), dim = 0)
                temp_t_out = t.cat((t_out[incorrect_indices], t_out[reg_indices]), dim = 0)
                temp_batch_y = batch_y.unsqueeze(dim = -1)
                temp_batch_y = t.cat((temp_batch_y[incorrect_indices], temp_batch_y[reg_indices]), dim = 0).squeeze()

                if temp_batch_y.size() == t.Size([0]): # 0-d tensor
                    return None

                # here, we use cckd_t according to paper
                lambd = t.sum(F.softmax(temp_t_out, dim = -1) * one_hot(temp_batch_y, num_classes), dim = -1, keepdim = True)  # [m, 1]
                y_c = lambd * F.softmax(temp_t_out / temperature, dim = -1) + (1 - lambd) * one_hot(temp_batch_y, num_classes)
                y_c = y_c / (t.norm(y_c, p = 1, dim = -1, keepdim = True) + 1e-8)  # norm to keep distribution
                loss = -t.mean(t.sum(y_c * t.log(F.softmax(temp_s_out / temperature, dim = -1) + 1e-8), dim = -1))

                return loss, s_out

            else:
                raise Exception('No other CCKD methods!')


def train(**kwargs):
    """Train the model"""
    opt.parse(**kwargs)
    opt.print()

    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    # Step 0 Decide the structure of the model#
    # Step 1 Load the data set#
    dataloader = get_data_loader(data_root = opt.data_root, dataset = opt.dataset, mode = 'train', batch_size = opt.batch_size, transform = opt.train_transform)
    # Step 2 Reshape the inputs#
    # Step 3 Normalize the inputs#
    # Step 4 Initialize parameters#
    # Step 5 Forward propagation(Vectorization/Activation functions)#
    t_model, s_model = get_models(model_name = opt.model_name, distill = opt.distill, baseline = opt.baseline)
    if not opt.distill:
        if opt.baseline: # train student model baseline
            s_model.to(device)
            model = s_model

        else: # train teacher model
            del s_model
            t_model.to(device)
            model = t_model

    else: # distill
        del t_model
        t_model = t.load(f'./checkpoints/t_model_{opt.dataset}.pth') # load pretrained model
        t_model.to(device)
        t_model.eval()
        s_model.to(device)
        model = s_model

    # Step 6 Compute cost#
    cls_cost = t.nn.CrossEntropyLoss().to(device)

    # Step 7 Backward propagation(Vectorization/Activation functions gradients)#
    if opt.optimizer == 'adamw':
        optimizer = t.optim.AdamW(params = filter(lambda x: x.requires_grad, model.parameters()), lr = opt.init_lr,
                                  weight_decay = opt.weight_decay, amsgrad = False)
    elif opt.optimizer == 'adam':
        optimizer = t.optim.Adam(params = filter(lambda x: x.requires_grad, model.parameters()), lr = opt.init_lr,
                                 weight_decay = opt.weight_decay, amsgrad = False)
    elif opt.optimizer == 'momentum':
        optimizer = t.optim.SGD(params = filter(lambda x: x.requires_grad, model.parameters()), lr = opt.init_lr,
                                weight_decay = opt.weight_decay, momentum = 0.9)
    else:
        raise Exception('No other optimizers!')

    lr_sceduler = t.optim.lr_scheduler.MultiStepLR(optimizer, gamma = opt.gamma, milestones = opt.milestones)

    # Step 8 Update parameters#
    losses = []
    cls_losses = []
    dis_losses = []
    accs = []
    for epoch in tqdm(range(opt.epochs)):
        print('Epoch %d / %d.' % (epoch + 1, opt.epochs))
        for i, (batch_x, batch_y) in enumerate(dataloader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            res = train_with_cckd(batch_x = batch_x, batch_y = batch_y, model = model, cls_cost = cls_cost, alpha = opt.alpha,
                                  epoch = epoch, distill = opt.distill, num_classes = 10,
                                  t_model = t_model, distill_type = opt.distill_type, temperature = opt.temperature, lambd = opt.lambd)

            if not opt.distill:
                loss, out = res
                loss.backward()
            else:
                if opt.distill_type == 'kd' or opt.distill_type == 'cckd_l':
                    cls_loss, dis_loss, loss, out = res
                    loss.backward()
                elif opt.distill_type == 'cckd_t' or opt.distill_type == 'cckd_t_reg':
                    if res is None:
                        pass # because of 'cckd_t_reg'
                    else:
                        loss, out = res
                        loss.backward()
                else:
                    raise()

            optimizer.step()

            if i % opt.batch_size == 0:
                if res is not None:
                    preds = t.argmax(out, dim = -1)
                    correct = t.sum(preds == batch_y).float()
                    acc = correct / batch_x.size(0)
                if not opt.distill:
                    sign = 'Teacher' if not opt.baseline else 'Student'
                    print(f'\t\033[34m{sign} model : \033[0mBatch %d / %d has cls cost : %.2f. || acc : %.2f%%.' % (i + 1, len(dataloader), loss.item(), acc * 100.))
                    losses.append(loss.item())
                else:
                    if opt.distill_type == 'kd' or opt.distill_type == 'cckd_l':
                        print('\t\033[34mStudent model : \033[0m Batch %d / %d has cost : %.2f[cls cost : %.2f & dis cost : %.2f]. || acc : %.2f%%.' % (i + 1, len(dataloader), loss.item(), cls_loss.item(), dis_loss.item(), acc * 100.))
                        losses.append(loss.item())
                        cls_losses.append(cls_loss.item())
                        dis_losses.append(dis_loss.item())
                    elif opt.distill_type == 'cckd_t' or opt.distill_type == 'cckd_t_reg':
                        if res is None:
                            pass
                        else:
                            print('\t\033[34mStudent model : \033[0mBatch %d / %d has cls cost : %.2f. || acc : %.2f%%.' % (i + 1, len(dataloader), loss.item(), acc * 100.))
                            losses.append(loss.item())

                if res is not None:
                    accs.append(acc.item())

        lr_sceduler.step()

    print('Training is done!')
    if not opt.distill:
        sign = 't' if not opt.baseline else 's'
        filename = os.path.join('./checkpoints', f'{sign}_model_{opt.dataset}.pth')
    else:
        filename = os.path.join('./checkpoints', f's_model_{opt.dataset}_{opt.distill_type}.pth')
    t.save(model, filename)

    # now visualize results
    f, ax = plt.subplots(1, 2, figsize = (10, 5))
    ax[0].plot(range(len(losses)), losses, '--r', label = 'loss')
    ax[0].grid(True)
    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel('Values')
    ax[0].set_title('Loss')
    ax[0].legend(loc = 'best')
    if not opt.distill:
        sign = 'Teacher' if not opt.baseline else 'Student'
        f.suptitle(f'{sign} model statistics')

    else:
        if opt.distill_type == 'kd' or opt.distill_type == 'cckd_l':
            f.suptitle(f'Student {opt.distill_type} model statistics')
            ax[0].plot(range(len(cls_losses)), cls_losses, '--b', label = 'cls_loss')
            ax[0].grid(True)
            ax[0].set_xlabel('Steps')
            ax[0].set_ylabel('Values')
            ax[0].set_title('Cls Loss')
            ax[0].legend(loc = 'best')

            ax[0].plot(range(len(dis_losses)), dis_losses, '--y', label = 'dis_loss')
            ax[0].grid(True)
            ax[0].set_xlabel('Steps')
            ax[0].set_ylabel('Values')
            ax[0].set_title('Dis Loss')
            ax[0].legend(loc = 'best')
        else:
            f.suptitle(f'Student {opt.distill_type} model statistics')

    ax[-1].plot(range(len(accs)), accs, '--g', label = 'acc')
    ax[-1].grid(True)
    ax[-1].set_xlabel('Steps')
    ax[-1].set_ylabel('Values')
    ax[-1].set_title('Acc')
    ax[-1].legend(loc = 'best')

    if not opt.distill:
        sign = 't' if not opt.baseline else 's'
        filename = os.path.join('./results', f'{sign}_model_{opt.dataset}.png')
    else:
        filename = os.path.join('./results', f's_model_{opt.dataset}_{opt.distill_type}.png')

    plt.savefig(filename)
    plt.close()