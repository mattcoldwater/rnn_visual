# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import h5py
import os.path as osp
import sys
import scipy.misc
import pickle
import torch.nn.utils.rnn as rnn_utils
import gc

# still need changes:
#######################################
# def collate_fn_aug(self,batch):
#####################################

class NTUDataset(Dataset):
    """
    NTU Skeleton Dataset.

    Args:
        x (list): Input dataset, each element in the list is an ndarray corresponding to
        a joints matrix of a skeleton sequence sample
        y (list): Action labels
    """

    def __init__(self, x, y):
        self.x = x
        self.y = np.array(y, dtype='int')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return [self.x[index], int(self.y[index])]

class NTUDataLoaders(object):
    def __init__(self, case = 1, aug = 0, data_path='/content/ntu/'):
        self.transformed_path = osp.join(data_path, 'transformed_data')
        self.maxn_f = 'no idea' # max num frames

        self.case = case
        self.aug = aug
        self.create_datasets()
        self.train_set = NTUDataset(self.train_X, self.train_Y)
        self.val_set = NTUDataset(self.val_X, self.val_Y)
        self.test_set = NTUDataset(self.test_X, self.test_Y)

    def get_train_loader(self, batch_size, num_workers):
        if self.aug == 1:
            return DataLoader(self.train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, drop_last=True,
                              collate_fn=self.collate_fn_aug, pin_memory=True)
        else:
            return DataLoader(self.train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, drop_last=True,
                              collate_fn=self.collate_fn, pin_memory=True)

    def get_val_loader(self, batch_size, num_workers):
        return DataLoader(self.val_set, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          collate_fn=self.collate_fn, pin_memory=True)

    def get_test_loader(self, batch_size, num_workers):
        return DataLoader(self.test_set, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          collate_fn=self.collate_fn, pin_memory=True)

    def collate_fn_aug(self,batch):
        x, y = zip(*batch)
        x = self.align_padding(x)
        x = torch.stack([torch.from_numpy(x[i]) for i in range(len(x))], 0)
        print(x.shape)
        ########################### augmentation
        x = _transform(x)
        ###########################
        print(x.shape)
        x = torch.stack([torch.from_numpy(x[i]) for i in range(len(x))], 0)
        print(x.shape)
        y = torch.LongTensor(y)
        return [x, y]

    def collate_fn(self, batch):
        # len(batch) = batch_size
        # len(batch[i]) = 2
        # batch[i][0] (60,150)
        # batch[i][1] int

        x, y = zip(*batch)
        x = [torch.from_numpy(x[i]) for i in range(len(x))]
        x_len = torch.LongTensor([_.shape[0] for _ in x])
        x = rnn_utils.pad_sequence(x, batch_first=True) 
        # [10(batch), 174(sequence length, can vary, max seems to be 300), 150]
        y = torch.LongTensor(y) # [10,]
        return x, x_len, y

    def get_train_size(self):
        return len(self.train_Y)

    def get_val_size(self):
        return len(self.val_Y)

    def get_test_size(self):
        return len(self.test_Y)

    def create_datasets(self):
        if self.case == 0:
            self.metric = 'C-Subject'
        else:
            self.metric = 'C-Setup'

        # original code for reference
        # criterion = nn.CrossEntropyLoss().cuda()
        # for i, (inputs, maxmin, target) in enumerate(train_loader):
        # loss = criterion(output, target)

        gc.collect()
        save_path = self.transformed_path
        evaluation = self.metric

        x_pkl = osp.join(save_path, '%s_x.pkl' % (evaluation))
        y_pkl = osp.join(save_path, '%s_y.pkl' % (evaluation))
        valid_x_pkl = osp.join(save_path, '%s_valid_x.pkl' % (evaluation))
        valid_y_pkl = osp.join(save_path, '%s_valid_y.pkl' % (evaluation))
        test_x_pkl = osp.join(save_path, '%s_test_x.pkl' % (evaluation))
        test_y_pkl = osp.join(save_path, '%s_test_y.pkl' % (evaluation))

        with open(x_pkl, 'rb') as f:
            self.train_X = pickle.load(f)
        
        # print(sorted([s.shape[0] for s in self.train_X]))

        with open(y_pkl, 'rb') as f:
            self.train_Y = pickle.load(f)

        with open(valid_x_pkl, 'rb') as f:
            self.val_X = pickle.load(f)

        with open(valid_y_pkl, 'rb') as f:
            self.val_Y = pickle.load(f)

        with open(test_x_pkl, 'rb') as f:
            self.test_X = pickle.load(f)

        with open(test_y_pkl, 'rb') as f:
            self.test_Y = pickle.load(f)

class NTUSmallDataLoaders(NTUDataLoaders):
    def create_datasets(self):
        if self.case == 0:
            self.metric = 'C-Subject'
        else:
            self.metric = 'C-Setup'

        # original code for reference
        # criterion = nn.CrossEntropyLoss().cuda()
        # for i, (inputs, maxmin, target) in enumerate(train_loader):
        # loss = criterion(output, target)

        gc.collect()
        save_path = self.transformed_path
        evaluation = self.metric

        valid_x_pkl = osp.join(save_path, '%s_valid_x.pkl' % (evaluation))
        valid_y_pkl = osp.join(save_path, '%s_valid_y.pkl' % (evaluation))

        with open(valid_x_pkl, 'rb') as f:
            self.val_X = pickle.load(f)
            self.train_X = self.val_X
            self.test_X = self.val_X

        with open(valid_y_pkl, 'rb') as f:
            self.val_Y = pickle.load(f)
            self.test_Y = self.val_Y
            self.train_Y = self.val_Y

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def _rot(rot):
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = rot.new(rot.size()[:2] + (1,)).zero_()
    ones = rot.new(rot.size()[:2] + (1,)).fill_(1)

    r1 = torch.stack((ones, zeros, zeros),dim=-1)
    rx2 = torch.stack((zeros, cos_r[:,:,0:1], sin_r[:,:,0:1]), dim = -1)
    rx3 = torch.stack((zeros, -sin_r[:,:,0:1], cos_r[:,:,0:1]), dim = -1)
    rx = torch.cat((r1, rx2, rx3), dim = 2)

    ry1 = torch.stack((cos_r[:,:,1:2], zeros, -sin_r[:,:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,:,1:2], zeros, cos_r[:,:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 2)

    rz1 = torch.stack((cos_r[:,:,2:3], sin_r[:,:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,:,2:3], cos_r[:,:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 2)

    rot = rz.matmul(ry).matmul(rx)

    return rot

def _transform(x):
    x = x.contiguous().view(x.size()[:2] + (-1, 3))

    rot = x.new(x.size()[0],3).uniform_(-0.3, 0.3)

    rot = rot.repeat(1, x.size()[1])
    rot = rot.contiguous().view((-1, x.size()[1], 3))
    rot = _rot(rot)
    x = torch.transpose(x, 2, 3)
    x = torch.matmul(rot, x)
    x = torch.transpose(x, 2, 3)

    x = x.contiguous().view(x.size()[:2] + (-1,))
    return x

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp