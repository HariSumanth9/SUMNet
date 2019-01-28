#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 00:30:44 2018

@author: sumanthnandamuri
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage.morphology import distance_transform_edt

def dice_coefficient(pred, target):
    smooth = 1e-15
    num = pred.size()[0]
    m1 = pred.view(num, -1).float()
    m2 = target.view(num, -1).float()
    intersection = (m1*m2).sum(1)
    union = (m1 + m2).sum(1) + smooth - intersection
    score = intersection/union
    return score.mean()

def get_weights(labels_batch):
    weights = np.array([])
    labels_batch_numpy = labels_batch.numpy()
    n = labels_batch_numpy.shape[0]
    labels_batch_numpy = labels_batch_numpy.astype('uint8')
    for i in range(n):
        label = labels_batch_numpy[i][0]
        trnsf = distance_transform_edt(label)
        trnsf = ((np.abs((trnsf.max() - trnsf))/trnsf.max())*(trnsf>=1.)+1)
        trnsf = trnsf.flatten()
        weights = np.concatenate((weights, trnsf))
    weights = torch.from_numpy(weights).float()
    return weights

def make_loader(twisted_labels, fake_labels, one, zero, bs = 24):
    images = torch.cat([twisted_labels, fake_labels], 0)
    labels = torch.cat([one, zero], 0)
    discriminatorDataset = TensorDataset(images, labels)
    discriminatorDataLoader = DataLoader(discriminatorDataset, batch_size = bs, shuffle = True, num_workers = 4, pin_memory = True)
    return discriminatorDataLoader