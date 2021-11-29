# Transformers-Nets
Implmentation of several transformer-based architectures compared on a classification task.

### Implemented networks
:heavy_check_mark: [ViT](https://arxiv.org/abs/2010.11929)
:white_check_mark: [CrossViT](https://arxiv.org/pdf/2103.14899.pdf)
:heavy_check_mark: [CrossFormer](https://arxiv.org/abs/2108.00154)
:white_check_mark: [CvT](https://arxiv.org/pdf/2103.15808.pdf)
:white_check_mark: [SwinTransformer](https://arxiv.org/pdf/2103.14030.pdf)


### Usage
To train and test these models, you can simply put the following command into your terminal after adjusting the necessary parameters:
```
python3 main.py [--data-path DATA_PATH] [--data-set {CIFAR10,CIFAR100}]
                [--img_size IMG_SIZE] [--crop_size CROP_SIZE]
                [--patch_size PATCH_SIZE] [--val_size VAL_SIZE]
                [--color_jitter COLOR_JITTER]
                [--train-interpolation TRAIN_INTERPOLATION]
                [-model-name {vit,crossformer}]
                [--optimizer_name OPTIMIZER_NAME] [--lr LR]
                [--momentum MOMENTUM] [--batch_size BATCH_SIZE]
                [--num_workers NUM_WORKERS] [--weight_decay WEIGHT_DECAY]
                [--warmup_epochs WARMUP_EPOCHS] [--num_epochs NUM_EPOCHS]
                [--step_display STEP_DISPLAY] [--use-gpu]
```

### Acknowledgment
Our implmenetation is based on some modules from [lucidrains](https://github.com/lucidrains/vit-pytorch).
