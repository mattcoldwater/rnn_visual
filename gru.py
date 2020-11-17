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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
from data_gru import NTUDataLoaders, AverageMeter
import torch.nn.utils.rnn as rnn_utils

args = argparse.ArgumentParser(description='View adaptive')
args.add_argument('--stop_i', type=int, default=-1,
                  help='for debug')
args.add_argument('--model', type=str, default='VA',
                  help='the neural network to use')
args.add_argument('--dataset', type=str, default='NTU',
                  help='select dataset to evlulate')
args.add_argument('--max_epoches', type=int, default=200,
                  help='start number of epochs to run')
args.add_argument('--lr', type=float, default=0.005,
                  help='initial learning rate')
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
args.add_argument('--monitor', type=str, default='val_acc',
                  help='quantity to monitor (default: val_acc)')
args.add_argument('--train', type=int, default=1,
                  help='train or test')
args = args.parse_args()

class LSTM_Simple(nn.Module):
    def __init__(self, num_classes=120):
      super(LSTM_Simple, self).__init__()
      # self.layer1 = nn.LSTMCell(input_size=150, hidden_size=10) # input_size: feature
      self.lstm1 = nn.LSTM(input_size=150, hidden_size=100, num_layers=3, batch_first=True)
      # self.lstm2 = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
      self.layer2 = nn.Linear(in_features=100, out_features=num_classes)

    def forward(self, x_pack, hidden):
      # print(x.shape) # [batch_size, seq_len=129, features=150]

      # x_pack = rnn_utils.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
      # print(x_pack.data.shape) # (seq_len=273, features=150)


      out, hidden = self.lstm1(x_pack, hidden)
      # print(out.data.shape) # (seq_len=273, num_hiddens=10)
      # print(h1.data.shape) # [num_layers*num_directions, batch_size,  num_hiddens]
      # print(c1.data.shape) # [num_layers*num_directions, batch_size,  num_hiddens]

      # out, (h1, c1) = self.lstm2(out)
      # print(out.data.shape) # (seq_len=273, num_hiddens=20)

      out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first=True)
      # print(out_pad.shape) # [batch_size, seq_len=129, num_hiddens=10]
      
      feat = torch.mean(out_pad, dim=1)
      # print(feat.shape) # (batch_size, num_hiddens=10)

      outs = self.layer2(feat)
      # print(outs.shape) # (batch_size, num_classes=120)

      return outs, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(3, batch_size, 100).zero_().cuda(),
                      weight.new(3, batch_size, 100).zero_().cuda())
        return hidden


def main(results):

    num_classes = 120

    model = LSTM_Simple(num_classes=num_classes)
    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.monitor == 'val_acc':
        mode = 'max'
        monitor_op = np.greater
        best = -np.Inf
        str_op = 'improve'
    elif args.monitor == 'val_loss':
        mode = 'min'
        monitor_op = np.less
        best = np.Inf
        str_op = 'reduce'

    scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=args.lr_factor,
                                patience=2, cooldown=2, verbose=True)

    # Data loading
    ntu_loaders = NTUDataLoaders(args.case, args.aug)
    train_loader = ntu_loaders.get_train_loader(args.batch_size, args.workers)
    val_loader = ntu_loaders.get_val_loader(args.batch_size, args.workers)
    test_loader = ntu_loaders.get_test_loader(args.batch_size, args.workers)
    train_size = ntu_loaders.get_train_size()
    val_size = ntu_loaders.get_val_size()
    print('Train on %d samples, validate on %d samples' %
          (train_size, val_size))
    assert (len(train_loader) + len(val_loader) + len(test_loader) ) * args.batch_size >= 100000

    best_epoch = 0
    output_dir = root_path

    checkpoint = osp.join(output_dir, '%s_best.pth' % args.case)

    pred_dir = osp.join(output_dir, '%s_pred.txt' % args.case)
    label_dir = osp.join(output_dir, '%s_label.txt' % args.case)

    earlystop_cnt = 0
    csv_file = osp.join(output_dir, '%s_log.csv' % args.case)
    log_res = list()

    # Training
    if args.train == 1:
        for epoch in range(args.max_epoches):
            # train for one epoch
            t_start = time.time()
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            val_loss, val_acc = validate(val_loader, model, criterion)

            log_res += [[train_loss, train_acc, val_loss, val_acc]]

            print('Epoch-{:<3d} {:.1f}s\t'
                  'Train: loss {:.4f}\taccu {:.4f}\tValid: loss {:.4f}\taccu {:.4f}'
                  .format(epoch + 1, time.time() - t_start, train_loss, train_acc, val_loss, val_acc))

            current = val_loss if mode == 'min' else val_acc

            current = current.cpu()
            if monitor_op(current, best):
                print('Epoch %d: %s %sd from %.4f to %.4f, '
                      'saving model to %s'
                      % (epoch + 1, args.monitor, str_op, best, current, checkpoint))
                best = current
                best_epoch = epoch + 1
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best': best,
                    'monitor': args.monitor,
                    'optimizer': optimizer.state_dict(),
                }, checkpoint)
                earlystop_cnt = 0
            else:
                print('Epoch %d: %s did not %s' % (epoch + 1, args.monitor, str_op))
                earlystop_cnt += 1
            scheduler.step(current)
            if earlystop_cnt > 15:
                print('Epoch %d: early stopping' % (epoch + 1))
                break

        print('Best %s: %.4f from epoch-%d' % (args.monitor, best, best_epoch))
        # save log
        with open(csv_file, 'w') as fw:
            cw = csv.writer(fw)
            cw.writerow(['loss', 'acc', 'val_loss', 'val_acc'])
            cw.writerows(log_res)
        print('Save train and validation log into into %s' % csv_file)

    # Testing
    test(test_loader, model, checkpoint, results, pred_dir, label_dir)


def train(train_loader, model, criterion, optimizer, epoch):

    losses = AverageMeter()
    acces = AverageMeter()

    model.train()
    h = model.init_hidden(args.batch_size)

    for i, (inputs, x_len, target) in enumerate(train_loader):
        h = tuple([e.data for e in h])
        x_pack = rnn_utils.pack_padded_sequence(inputs, x_len, batch_first=True, enforce_sorted=False)
        
        model.zero_grad()
        optimizer.zero_grad()  # clear gradients out before each mini-batch

        output, h = model(x_pack.cuda(), h)
        target = target.cuda(non_blocking=True)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # backward
        loss.backward()
        optimizer.step()  # update parameters

        if (i + 1) % args.print_freq == 0:
            print('Epoch-{:<3d} {:3d} batches\t'
                  'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'accu {acc.val:.3f} ({acc.avg:.3f})'.format(
                   epoch + 1, i + 1, loss=losses, acc=acces))
        
        if args.stop_i == i: break
    return losses.avg, acces.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    acces = AverageMeter()

    h = model.init_hidden(args.batch_size)

    # switch to evaluation mode
    model.eval()

    for i, (inputs, x_len, target) in enumerate(val_loader):
        with torch.no_grad():
            h = tuple([each.data for each in h])
            x_pack = rnn_utils.pack_padded_sequence(inputs, x_len, batch_first=True, enforce_sorted=False)
            output, h = model(x_pack.cuda(), h)

        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

    return losses.avg, acces.avg


def test(test_loader, model, checkpoint, results, path, label_path):
    acces = AverageMeter()
    # load learnt model that obtained best performance on validation set
    model.load_state_dict(torch.load(checkpoint)['state_dict'], strict=False)
    # switch to evaluation mode

    h = model.init_hidden(args.batch_size)

    model.eval()

    preds, label = list(), list()
    t_start = time.time()
    for i, (inputs, x_len, target) in enumerate(test_loader):
        with torch.no_grad():
            h = tuple([each.data for each in h])
            x_pack = rnn_utils.pack_padded_sequence(inputs, x_len, batch_first=True, enforce_sorted=False)
            output, h = model(x_pack.cuda(), h)

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

    root_path = '/content/results_'+args.model
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