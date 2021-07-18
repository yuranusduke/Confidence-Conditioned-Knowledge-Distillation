"""
Test models

Created by Kunhong Yu
Date: 2021/07/16
"""
import os
import torch as t
from config import Config
from utils import get_data_loader, fgsm, repeat_error_verfication
from datetime import datetime
import sys

opt = Config()

def test(**kwargs):
    """Test models"""
    opt.parse(**kwargs)
    opt.print()

    # device
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    # data
    dataloader = get_data_loader(data_root = opt.data_root, dataset = opt.dataset, mode = 'test', batch_size = opt.batch_size, transform = opt.test_transform)

    # model
    if not opt.distill:
        sign = 't' if not opt.baseline else 's'
        filename = os.path.join('./checkpoints', f'{sign}_model_{opt.dataset}.pth')
    else:
        filename = os.path.join('./checkpoints', f's_model_{opt.dataset}_{opt.distill_type}.pth')
    model = t.load(filename)
    # send model to device
    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))

    if opt.repeat_error_ver:
        t_model = t.load(os.path.join('./checkpoints', f't_model_{opt.dataset}.pth'))
        t_model.eval()
        t_model.to(device)

    # starting evaluation
    print("Starting evaluation")

    model.eval()

    whole_acc = 0.
    whole_adv_acc = 0.
    whole_mu_s = 0.
    whole_mu_f = 0.
    for batch_idx, (batch_data, batch_target) in enumerate(dataloader):
        sys.stdout.write('\r>>Testing batch %d / %d.' % (batch_idx + 1, len(dataloader)))
        sys.stdout.flush()
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        batch_pred = model(batch_data)
        preds = t.argmax(batch_pred, dim = -1)
        correct = t.sum(preds == batch_target).float()
        acc = correct / batch_data.size(0)

        if opt.fgsm:
            adv_acc = fgsm(batch_data, batch_target, model, device, epsilon = opt.epsilon, index = batch_idx, dataset = opt.dataset, distill = opt.distill, baseline = opt.baseline, distill_type = opt.distill_type)
            whole_adv_acc += adv_acc

        if opt.distill and opt.repeat_error_ver:
            t_out = t_model(batch_data)
            mu_s, mu_f = repeat_error_verfication(t.argmax(t_out, dim = -1), preds, batch_target)
            whole_mu_s += mu_s
            whole_mu_f += mu_f

        whole_acc += acc.item()

    whole_acc /= len(dataloader)
    if opt.fgsm:
        whole_adv_acc /= len(dataloader)

    if opt.repeat_error_ver:
        whole_mu_f /= len(dataloader)
        whole_mu_s /= len(dataloader)

    print()

    if not opt.distill:
        sign = 'teacher' if not opt.baseline else 'student'
        eval_string = f"Evaluation of {sign} model on dataset %s with acc %.2f%%." % (opt.dataset, whole_acc * 100.)
    else:
        eval_string = "Evaluation of student model on dataset {:s} with distillation type {:s} acc {:.2f}%.".format(opt.dataset, opt.distill_type, whole_acc * 100.)
        if opt.repeat_error_ver:
            eval_string += '\n\tRepeat error verification : [mu_S : %.2f; mu_F : %.2f].' % (whole_mu_s, whole_mu_f)
    if opt.fgsm:
        eval_string += '\n\tFGSM acc : {:.2f}%.'.format(whole_adv_acc * 100.)

    print('\n\033[36m' + eval_string + '\033[0m')

    filename = os.path.join('./results/', 'saved_test_results.txt')
    string = '*' * 40 + str(datetime.now()) + '*' * 40 + '\n'
    string += "Model capacity : Total params: %.2fM" % (total / 1e6) + '\n'
    for k, _ in opt.__class__.__dict__.items():
        v = getattr(opt, k)
        if not k.startswith('__'):
            string += str(k) + ' --> ' + str(v) + '\n'

    string += '\n' + eval_string + '\n\n'

    with open(filename, 'a+') as f:
        f.write(string)
        f.flush()

    print('Testing is done!')