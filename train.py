import time
import numpy as np
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn


# train function per batch
def train(args, train_loader, model, criterion, optimizer, epoch, 
          batch_iter, total_batch, train_batch, use_cuda=False):

    model.train()

    loader_length = len(train_loader)
    
    total_loss = 0.
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_start = time.time()

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)
        losses = criterion(outputs, targets)

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()

        total_loss += losses.data.item()

        batch_time = time.time() - batch_start

        batch_iter += 1

        if epoch % args.step_display == 0:
            print("[Training] Time: {} Epoch: [{}/{}] batch_idx: [{}/{}] batch_iter: [{}/{}] batch_losses: {:.4f} BatchTime: {:.4f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch,
                    args.num_epochs,
                    batch_idx,
                    train_batch,
                    batch_iter,
                    total_batch,
                    losses.data.item(),
                    batch_time
                ))


    return batch_iter, np.mean(total_loss)


# Validation phase
def val(args, val_loader, model, criterion, epoch, val_total_batch, use_cuda=False):
    
    model.eval()

    correct = 0.
    total = 0.
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):

            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            batch_size = inputs.shape[0]

            outputs = model(inputs)
            losses = criterion(outputs, targets)

            total_loss += losses.data.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == targets).sum().item()
        
    acc = 100 * correct / total
    epoch_loss = np.mean(total_loss)
           
    if epoch % args.step_display == 0:
        print(f"Validation Epoch: [{epoch}/{args.num_epochs}] Loss: {epoch_loss} Accuracy: {acc}")
    
    return epoch_loss, acc


  
    
    
    
    
    
