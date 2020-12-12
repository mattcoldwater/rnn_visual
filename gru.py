# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import time
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import os.path as osp
import csv
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models

import torch.nn.utils.rnn as rnn_utils
from data_gru import NTUDataLoaders, AverageMeter, NTUSmallDataLoaders
from networks import LSTM_Simple, GRU_Simple, GRU_Att

import gc

# python gru.py --aug 0 --experiment debug1 --print_freq 500 --batch_size 256 --lr 0.005 --train 1 --max_epoches 5
# python gru.py --aug 0 --experiment debug1 --print_freq 500 --batch_size 256 --lr 0.005 --train 1 --max_epoches 5 --att 3
# python gru.py --aug 0 --experiment debug1 --print_freq 500 --batch_size 256 --lr 0.005 --train 0 --max_epoches 1 --att 3

##############
# 3 att
# python gru.py --aug 0 --experiment att3_gru0 --print_freq 500 --batch_size 256 --lr 0.005 --train 1 --max_epoches 100 --att 3
# python gru.py --aug 0 --experiment att3_gru0 --print_freq 500 --batch_size 256 --lr 0.005 --train 0 --max_epoches 1 --att 3

# 0 att
# python gru.py --aug 0 --experiment att0_gru3 --print_freq 500 --batch_size 256 --lr 0.005 --train 1 --max_epoches 100
# python gru.py --aug 0 --experiment att0_gru3 --print_freq 500 --batch_size 256 --lr 0.005 --train 0 --max_epoches 1

# 1 att
# python gru.py --aug 0 --experiment att1_gru2 --print_freq 500 --batch_size 256 --lr 0.005 --train 1 --max_epoches 100 --att 1
# python gru.py --aug 0 --experiment att1_gru2 --print_freq 500 --batch_size 256 --lr 0.005 --train 0 --max_epoches 1 --att 1

# 2 att
# python gru.py --aug 0 --experiment att2_gru1 --print_freq 500 --batch_size 256 --lr 0.005 --train 1 --max_epoches 100 --att 2
# python gru.py --aug 0 --experiment att2_gru1 --print_freq 500 --batch_size 256 --lr 0.005 --train 0 --max_epoches 1 --att 2

# 4 att
# python gru.py --aug 0 --experiment att4_gru0 --print_freq 500 --batch_size 256 --lr 0.005 --train 1 --max_epoches 100 --att 4 --nlayer 4
# python gru.py --aug 0 --experiment att4_gru0 --print_freq 500 --batch_size 256 --lr 0.005 --train 0 --max_epoches 1 --att 4 --nlayer 4
##################
##############
# 0 att
# python gru.py --dropout 0.5 --aug 0 --experiment att0_gru3_d --print_freq 500 --batch_size 256 --lr 0.005 --train 1 --max_epoches 100
# python gru.py --dropout 0.5 --aug 0 --experiment att0_gru3_d --print_freq 500 --batch_size 256 --lr 0.005 --train 0 --max_epoches 1

# 3 att
# python gru.py --dropout 0.5 --aug 0 --experiment att3_gru0_d --print_freq 500 --batch_size 256 --lr 0.005 --train 1 --max_epoches 100 --att 3
# python gru.py --dropout 0.5 --aug 0 --experiment att3_gru0_d --print_freq 500 --batch_size 256 --lr 0.005 --train 0 --max_epoches 1 --att 3
##################

# python gru.py --aug 0 --experiment debug1 --print_freq 500 --batch_size 4 --lr 0.005 --train 1 --max_epoches 1 --stop_i 3 --debug
# python gru.py --aug 0 --experiment debug1 --print_freq 500 --batch_size 4 --lr 0.005 --train 1 --max_epoches 1 --stop_i 3 --debug --att 3
# python gru.py --aug 0 --experiment debug1 --print_freq 500 --batch_size 4 --lr 0.005 --train 1 --max_epoches 1 --stop_i 3 --debug --att 3 --dropout 0.5
# python gru.py --aug 0 --experiment debug1 --print_freq 500 --batch_size 4 --lr 0.005 --train 1 --max_epoches 1 --stop_i 3 --debug --dropout 0.5 --nlayer 4 --att 4

args = argparse.ArgumentParser(description='Rnn Visual')
## For debug Only!
args.add_argument('--stop_i', type=int, default=-1,
                  help='for debug')
args.add_argument('--debug', action='store_true',
                  help='use samller dataset')
## For release,
args.add_argument('--experiment', type=str, default='debug1',
                  help='the experiment name')
args.add_argument('--data_path', type=str, default='/content/ntu/',
                  help='NTU Data Path')
args.add_argument('--max_epoches', type=int, default=200,
                  help='start number of epochs to run')
args.add_argument('--lr', type=float, default=0.005,
                  help='initial learning rate')

args.add_argument('--dropout', type=float, default=0,
                  help='dropout rate')
args.add_argument('--nlayer', type=int, default=3,
                  help='nlayer')
args.add_argument('--att', type=int, default=0,
                  help='attention layer num')

args.add_argument('--lr_factor', type=float, default=0.1,
                  help='the ratio to reduce lr on each step')
args.add_argument('--optimizer', type=str, default='Adam',
                  help='the optimizer type')
args.add_argument('--print_freq', '-p', type=int, default=20,
                  help='print frequency (default: 20)')
args.add_argument('-b', '--batch_size', type=int, default=32,
                  help='mini-batch size (default: 256)')
args.add_argument('--case', type=int, default=0,
                  help='select which case')
args.add_argument('--aug', type=int, default=1,
                  help='data augmentation')
args.add_argument('--workers', type=int, default=8,
                  help='number of data loading workers')
args.add_argument('--train', type=int, default=1,
                  help='train or test')
args = args.parse_args()


def main(results):

    num_classes = 120
    n_gru_layers = args.nlayer
    gru_hidden_size = 100
    feature_size = 150

    dropout = args.dropout

    atten = [False, False, False, False]
    if args.att == 1:
        atten = [True, False, False, False]
    elif args.att == 2:
        atten = [True, True, False, False]
    elif args.att == 3:
        atten = [True, True, True, False]
    elif args.att == 4:
        atten = [True, True, True, True]

    batch_first = True

    # model = LSTM_Simple(num_classes=num_classes)
    model = GRU_Att(num_classes=num_classes, layers=n_gru_layers, hidden_size=gru_hidden_size, 
                        input_size=feature_size, atten=atten, batch_first=batch_first, dropout=dropout)
    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best = -np.Inf

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_factor,
                                patience=2, cooldown=2, verbose=True)

    # Data loading
    gc.collect()
    if args.debug: # Debug, smaller dataset
        ntu_loaders = NTUSmallDataLoaders(args.case, args.aug, data_path=args.data_path)
    else:
        ntu_loaders = NTUDataLoaders(args.case, args.aug, data_path=args.data_path)
    train_loader = ntu_loaders.get_train_loader(args.batch_size, args.workers)
    val_loader = ntu_loaders.get_val_loader(args.batch_size, args.workers)
    test_loader = ntu_loaders.get_test_loader(args.batch_size, args.workers)
    train_size = ntu_loaders.get_train_size()
    val_size = ntu_loaders.get_val_size()
    test_size = ntu_loaders.get_test_size()
    print('Train on %d samples, validate on %d samples, test on %d samples' %
          (train_size, val_size, test_size))
    if not args.debug: # Debug
        assert (len(train_loader) + len(val_loader) + len(test_loader) ) * args.batch_size >= 100000

    best_epoch = 0
    best_hidden = None
    output_dir = root_path

    checkpoint = osp.join(output_dir, '%s_best.pth' % args.case)

    pred_dir = osp.join(output_dir, '%s_pred.txt' % args.case)
    label_dir = osp.join(output_dir, '%s_label.txt' % args.case)

    att_dir = osp.join(output_dir, '%s_att.pkl' % args.case)
    len_dir = osp.join(output_dir, '%s_len.pkl' % args.case)
    x_dir = osp.join(output_dir, '%s_x.pkl' % args.case)
    y_dir = osp.join(output_dir, '%s_y.pkl' % args.case)
    visual_dirs = [att_dir, len_dir, x_dir, y_dir]

    earlystop_cnt = 0
    csv_file = osp.join(output_dir, '%s_log.csv' % args.case)
    log_res = list()

    # Training
    if args.train == 1:
        for epoch in range(args.max_epoches):
            # train for one epoch
            t_start = time.time()
            train_loss, train_acc, train_hidden = train(train_loader, model, criterion, optimizer, epoch, best_hidden=best_hidden)
            # evaluate on validation set
            val_loss, val_acc = validate(val_loader, model, criterion)

            log_res += [[train_loss, train_acc, val_loss, val_acc]]

            print('Epoch-{:<3d} {:.1f}s\t'
                  'Train: loss {:.4f}\taccu {:.4f}\tValid: loss {:.4f}\taccu {:.4f}'
                  .format(epoch + 1, time.time() - t_start, train_loss, train_acc, val_loss, val_acc))

            current = val_acc.cpu()
            if np.greater(current, best):
                print('Epoch %d: val_acc improved from %.4f to %.4f, '
                      'saving model to %s'
                      % (epoch + 1, best, current, checkpoint))
                best = current
                best_epoch = epoch + 1
                best_hidden = train_hidden
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best': best,
                    'monitor': 'val_acc',
                    'optimizer': optimizer.state_dict(),
                    'best_hidden': best_hidden,
                }, checkpoint)
                earlystop_cnt = 0
            else:
                print('Epoch %d: val_acc did not improve' % (epoch + 1))
                earlystop_cnt += 1
            scheduler.step(current)
            if earlystop_cnt > 8:
                print('Epoch %d: early stopping' % (epoch + 1))
                break

        print('Best val_acc: %.4f from epoch-%d' % (best, best_epoch))
        # save log
        with open(csv_file, 'w') as fw:
            cw = csv.writer(fw)
            cw.writerow(['loss', 'acc', 'val_loss', 'val_acc'])
            cw.writerows(log_res)
        print('Save train and validation log into into %s' % csv_file)

    # Testing
    test(test_loader, model, checkpoint, results, pred_dir, label_dir, visual_dirs, atten)


def train(train_loader, model, criterion, optimizer, epoch, best_hidden=None):

    losses = AverageMeter()
    acces = AverageMeter()

    model.train()
    h = model.init_hidden(args.batch_size, best_hidden)

    for i, (inputs, x_len, target) in enumerate(train_loader):
        # https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426
        h = h.detach() # h.data
        
        model.zero_grad()
        optimizer.zero_grad()  # clear gradients out before each mini-batch

        output, h = model(inputs, x_len, h)
        target = target.cuda(non_blocking=True)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # backward
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()  # update parameters

        if (i + 1) % args.print_freq == 0:
            print('Epoch-{:<3d} {:3d} batches\t'
                  'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'accu {acc.val:.3f} ({acc.avg:.3f})'.format(
                   epoch + 1, i + 1, loss=losses, acc=acces))
        
        if args.stop_i == i: break
    
    return losses.avg, acces.avg, h.detach().cpu()


def validate(val_loader, model, criterion, best_hidden=None):
    losses = AverageMeter()
    acces = AverageMeter()

    h = model.init_hidden(args.batch_size, best_hidden)

    # switch to evaluation mode
    model.eval()

    for i, (inputs, x_len, target) in enumerate(val_loader):
        with torch.no_grad():
            h = h.detach()
            output, h = model(inputs, x_len, h)

        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

    return losses.avg, acces.avg


def test(test_loader, model, checkpoint, results, path, label_path, visual_dirs, atten):
    acces = AverageMeter()
    # load learnt model that obtained best performance on validation set
    model.load_state_dict(torch.load(checkpoint)['state_dict'], strict=False)
    best_hidden = torch.load(checkpoint)['best_hidden']
    # print(best_hidden.shape) # 3, 4, 100
    
    # switch to evaluation mode

    h = model.init_hidden(args.batch_size, best_hidden)

    model.eval()

    preds, label = list(), list()
    t_start = time.time()
    for i, (inputs, x_len, target) in enumerate(test_loader):
        with torch.no_grad():
            h = h.detach()
            if i == 19 and atten[0]:
                output, h, attentions = model(inputs, x_len, h, visual=True)

                [att_dir, len_dir, x_dir, y_dir] = visual_dirs

                with open(att_dir, 'wb') as ff:
                    # batch * seq_len * 150
                    pickle.dump(attentions.detach().cpu().numpy(), ff, pickle.HIGHEST_PROTOCOL)

                with open(len_dir, 'wb') as ff:
                    # batch
                    pickle.dump(x_len.detach().cpu().numpy(), ff, pickle.HIGHEST_PROTOCOL)

                with open(x_dir, 'wb') as ff:
                    # batch * seq_len * 150
                    pickle.dump(inputs.detach().cpu().numpy(), ff, pickle.HIGHEST_PROTOCOL)

                with open(y_dir, 'wb') as ff:
                    # batch
                    pickle.dump(target.detach().cpu().numpy(), ff, pickle.HIGHEST_PROTOCOL)

            else:
                output, h = model(inputs, x_len, h)

        output = output.cpu()
        pred = output.data.numpy()
        target = target.numpy()

        preds = preds + list(pred)
        label = label + list(target)

    preds = np.array(preds)
    label = np.array(label)

    preds_label = np.argmax(preds, axis=-1)
    total = ((label-preds_label)==0).sum()
    total = float(total)

    print("Model Accuracy:%.2f" % (total / len(label)*100))

    results.append(round(float(total/len(label)*100),2))
    np.savetxt(path, preds, fmt = '%f')
    np.savetxt(label_path, label, fmt = '%f')


def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)

    return correct.mul_(100.0 / batch_size)


def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':

    root_path = '/content/results_'+args.experiment
    if not osp.exists(root_path):
        os.mkdir(root_path)
    
    # get the number of total cases of certain dataset
    cases = 2 # 'C-Subject' 'C-Setup'
    results = list()

    for case in range(cases):
        print('case', case)
        args.case = case
        main(results)
        print()
    np.savetxt(root_path + '/result.txt', results, fmt = '%f')

    print(results)
    print('ave:', np.array(results).mean())