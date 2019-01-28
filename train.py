#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 00:27:04 2018

@author: sumanthnandamuri
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import tqdm
import time
import gc
from get_data_loader import data_loaders
from SUMNet import SUMNet
from utils import dice_coefficient, make_loader

images_dir = 'ThyroidData/data/'
labels_dir = 'ThyroidData/groundtruth/'
trainDataLoader, validDataLoader = data_loaders(images_dir, labels_dir, bs = 12)

net = SUMNet()
use_gpu = torch.cuda.is_available()
if use_gpu:
    net = netS.cuda()
    
optimizer = optim.Adam(netS.parameters(), lr = 1e-3)
criterion = nn.BCELoss()

def train(trainDataLoader, validDataLoader, net, optimizer, criterion, use_gpu):
    epochs = 50
    trainLoss = []
    validLoss = []
    trainDiceCoeff = []
    validDiceCoeff = []
    start = time.time()
    bestValidDice = 0
    
    for epoch in range(epochs):
        epochStart = time.time()
        trainRunningLoss = 0
        validRunningLoss = 0
        trainBatches = 0
        validBatches = 0
        trainDice = 0
        validDice = 0
        
        net.train(True)
        for data in tqdm.tqdm(trainDataLoader):
            inputs, labels = data
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            probs = net(inputs)
            loss = criterion(probs.view(-1), labels.view(-1))
            preds = (probs > 0.5).float()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trainRunningLoss += loss.item()
            trainDice += dice_coefficient(preds, labels).item()
            trainBatches += 1
        trainLoss.append(trainRunningLoss/trainBatches)
        trainDiceCoeff.append(trainDice/trainBatches)
        
        net.train(False)
        for data in validDataLoader:
            inputs, labels = data
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            probs = net(inputs)
            loss = criterion(probs.view(-1), labels.view(-1))
            preds = (probs > 0.5).float()
            validDice += dice_coefficient(preds, labels).item()
            validRunningLoss += loss.item()
            validBatches += 1
        validLoss.append(validRunningLoss/validBatches)
        validDiceCoeff.append(validDice/validBatches)
        if validDice > bestValidDice:
            bestValidDice = validDice
            torch.save(net.state_dict(), 'SUMNet.pt')

        epochEnd = time.time()-epochStart
        print('Epoch: {:.0f}/{:.0f} | Train Loss: {:.3f} | Valid Loss: {:.3f} | Train Dice: {:.3f} | Valid Dice: {:.3f}'\
              .format(epoch+1, epochs, trainRunningLoss/trainBatches, validRunningLoss/validBatches, trainDice/trainBatches, validDice/validBatches))
        print('Time: {:.0f}m {:.0f}s'.format(epochEnd//60,epochEnd%60))
    end = time.time()-start
    print('Training completed in {:.0f}m {:.0f}s'.format(end//60,end%60))
    trainLoss = np.array(trainLoss)
    validLoss = np.array(validLoss)
    trainDiceCoeff = np.array(trainDiceCoeff)
    validDiceCoeff = np.array(validDiceCoeff)
    DF = pd.DataFrame({'Train Loss': trainLoss, 'Valid Loss': validLoss, 'Train Dice': trainDiceCoeff, 'Valid Dice': validDiceCoeff})
    return DF

DF = train(trainDataLoader, validDataLoader, net, optimizer, criterion, use_gpu)
DF.to_csv('SUMNet.csv')

