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

from models.vit import ViT
from models.cvt import CvT
from models.cross_vit import CrossViT
from models.swin import SwinTransformer
from models.crossformer import CrossFormer



parser = argparse.ArgumentParser()

# ----- data
parser.add_argument('--data-path', default='./data/', type=str, help='dataset path')
parser.add_argument('--data-set', default='CIFAR10', choices=['CIFAR10', 'CIFAR100'], type=str, 
                                  help='Image Net dataset path')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--patch_size', type=int, default=16)
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
    elif args.model_name == "swin":
        model = SwinTransformer(img_size=args.img_size, patch_size=4, in_chans=3,
                                num_classes=nb_classes,
                                embed_dim=192, depths=[ 2, 2, 6, 2 ], num_heads=[ 3, 6, 12, 24 ],
                                window_size=7, mlp_ratio=4.,
                                qkv_bias=True, qk_scale=None,
                                drop_rate=0., drop_path_rate=0.1,
                                ape=False, patch_norm=True, use_checkpoint=False)
    elif args.model_name == "cross_vit":
        model = CrossViT(image_size = args.img_size,
                         num_classes = nb_classes,
                         depth = 4,          # number of multi-scale encoding blocks
                         sm_dim = 192,       # high res dimension
                         sm_patch_size = 16, # high res patch size (smaller than lg_patch_size)
                         sm_enc_depth = 2,        # high res depth
                         sm_enc_heads = 8,        # high res heads
                         sm_enc_mlp_dim = 2048,   # high res feedforward dimension
                         lg_dim = 384,            # low res dimension
                         lg_patch_size = 64,      # low res patch size
                         lg_enc_depth = 3,        # low res depth
                         lg_enc_heads = 8,        # low res heads
                         lg_enc_mlp_dim = 2048,   # low res feedforward dimensions
                         cross_attn_depth = 2,    # cross attention rounds
                         cross_attn_heads = 8,    # cross attention heads
                         dropout = 0.1,
                         emb_dropout = 0.1
                        )
    elif args.model_name == "cvt":
        model = CvT( num_classes = 1000,
                     s1_emb_dim = 64,        # stage 1 - dimension
                     s1_emb_kernel = 7,      # stage 1 - conv kernel
                     s1_emb_stride = 4,      # stage 1 - conv stride
                     s1_proj_kernel = 3,     # stage 1 - attention ds-conv kernel size
                     s1_kv_proj_stride = 2,  # stage 1 - attention key / value projection stride
                     s1_heads = 1,           # stage 1 - heads
                     s1_depth = 1,           # stage 1 - depth
                     s1_mlp_mult = 4,        # stage 1 - feedforward expansion factor
                     s2_emb_dim = 192,       # stage 2 - (same as above)
                     s2_emb_kernel = 3,
                     s2_emb_stride = 2,
                     s2_proj_kernel = 3,
                     s2_kv_proj_stride = 2,
                     s2_heads = 3,
                     s2_depth = 2,
                     s2_mlp_mult = 4,
                     s3_emb_dim = 384,       # stage 3 - (same as above)
                     s3_emb_kernel = 3,
                     s3_emb_stride = 2,
                     s3_proj_kernel = 3,
                     s3_kv_proj_stride = 2,
                     s3_heads = 4,
                     s3_depth = 10,
                     s3_mlp_mult = 4,
                     dropout = 0.
                    )
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
