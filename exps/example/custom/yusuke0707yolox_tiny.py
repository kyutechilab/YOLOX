#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False
                # Define yourself dataset path
        self.data_dir = "./datasets/0707_Extracted/"
        self.train_ann = "train_annotations.json"
        self.val_ann = "val_annotations.json"

        self.num_classes = 9
        self.eval_interval =1
        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1
        self.strict=False
        
# --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
