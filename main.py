import os
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW

from utils import setup_seed
from train import train, val
from data.Datasets import build_dataset
from models.crossformer import CrossFormer
from models.vit import ViT


parser = argparse.ArgumentParser()

# ----- data
parser.add_argument('--data-path', default='./data/', type=str, help='dataset path')
parser.add_argument('--data-set', default='CIFAR10', choices=['CIFAR10', 'CIFAR100'], type=str, 
                                  help='Image Net dataset path')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--val_size', type=float,  default=0.1)
parser.add_argument('--color_jitter', type=float,  default=0.4)
parser.add_argument('--train-interpolation', type=str, default='bicubic',
                    help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

# ----- model
parser.add_argument('-model-name', default='vit', choices=['vit', 'crossformer'], type=str, 
                                   help='Choose the adequate model.')

# ---- optimizer
parser.add_argument('--optimizer_name', default="adamw", type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--weight_decay', default=5e-2, type=float)

# ---- train
parser.add_argument('--warmup_epochs', default=5, type=int)
parser.add_argument('--num_epochs', default=90, type=int)
parser.add_argument('--step_display', default=1, type=int)
parser.add_argument('--use-gpu', action='store_true')

def main(args):

    # Precise device
    use_cuda = True if torch.cuda.is_available() and args.use_gpu else False
    print("On Cuda!" if use_cuda else "On CPU")
    
    # Prepare Datasets
    train_loader, val_loader, nb_classes, len_train, len_val = build_dataset(args)

    # model
    if args.model_name == "crossformer":
        model = CrossFormer(num_classes = nb_classes)
    elif args.model_name == "vit":
        model = ViT(image_size=args.img_size, patch_size=16, num_classes=nb_classes, 
                    dim=768, depth=12, heads=12, mlp_dim=64)
    else:
        raise NotImplementedError(f"The model {args.model_name} has not been implemented!")
    
    #print(f"=============== model architecture ===============")
    #print(model)
    
    if use_cuda:
        model.cuda()

    # Choose an Optimizer
    if args.optimizer_name == "sgd":
        optimizer = SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, 
                                                     momentum=0.9)
    elif args.optimizer_name == "adam":
        optimizer = Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_name == "adamw":
        optimizer = AdamW(model.parameters(), args.lr, betas=(0.9, 0.95), 
                                                       weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f"{args.optimizer_name} optimizer has not been implemented!")
    

    start_epoch = 1
    batch_iter = 0
    ngpus_per_node = torch.cuda.device_count()
    train_batch = math.ceil(len_train / (args.batch_size * ngpus_per_node))
    total_batch = train_batch * args.num_epochs

    val_batch = math.ceil(len_val / (args.batch_size * ngpus_per_node))
    val_total_batch = val_batch * args.num_epochs

    criterion = nn.CrossEntropyLoss()
    
    val_losses = []
    train_losses = []
    accuracy = []
    
    # training loop
    print()
    print("Start training...")
    for epoch in range(start_epoch, args.num_epochs + 1):
        batch_iter, tr_loss = train(args, train_loader, model, criterion, optimizer, epoch, 
                                    batch_iter, total_batch, train_batch, use_cuda)
        val_loss, acc = val(args, val_loader, model, criterion, epoch, val_total_batch, use_cuda)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        accuracy.append(acc)
        
if __name__ == "__main__":
    args = parser.parse_args()
    setup_seed()
    main(args)
