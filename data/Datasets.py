import os
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# Prepare datasets
def build_dataset(args):

    # Applying Transformations
    train_transform = build_transform(args, is_train=True)
    val_transform = build_transform(args, is_train=True)
    
    # load the dataset
    if args.data_set == 'CIFAR10':
        train_dataset = datasets.CIFAR10(args.data_path, train=True, 
                                          download=True, transform=train_transform)
        val_dataset = datasets.CIFAR10(args.data_path, train=False, 
                                          download=True, transform=val_transform)
        nb_classes = 10
    elif args.data_set == 'CIFAR100':
        train_dataset = datasets.CIFAR100(args.data_path, train=True, 
                                          download=True, transform=train_transform)
        val_dataset = datasets.CIFAR100(args.data_path, train=False, 
                                          download=True, transform=val_transform)
        nb_classes = 100

    """
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.val_size * num_train))

    np.random.seed(100)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    """
    
    # dataloader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              #sampler=train_sampler,
                              pin_memory=True,
                              drop_last=True,
                             )

    validation_loader = DataLoader(dataset=val_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.num_workers,
                                   #sampler=valid_sampler,
                                   pin_memory=True,
                                   drop_last=False
                                  )

    len_train = len(train_dataset)
    len_val = len(val_dataset)

    print("Trainig dataset length: ", len_train)
    print("Validation dataset length: ", len_val)

    return train_loader, validation_loader, nb_classes, len_train, len_val

def build_transform(args, is_train):
    resize_im = args.img_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform( input_size=args.img_size,
                                      is_training=True,
                                      color_jitter=args.color_jitter,
                                      #auto_augment=args.aa,
                                      interpolation=args.train_interpolation,
                                      re_prob=0.25,
                                      re_mode='pixel',
                                      re_count=1,
                                    )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with RandomCrop
            transform.transforms[0] = transforms.RandomCrop(args.img_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.img_size)
        # to maintain same ratio w.r.t. 224 images
        t.append(transforms.Resize(size, interpolation=3), )
        t.append(transforms.CenterCrop(args.img_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


