#!/home/dongjai/anaconda3/envs/tensorflow2/bin/python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : config.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:06:54
#   Description :
#
#================================================================

from easydict import EasyDict as edict

__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

# Set the class name
__C.YOLO.CLASSES              = '~/catkin_ws/src/act_recognizer/src/data/classes/coco.names'
#__C.YOLO.CLASSES              = "./data/classes/yymnist.names"
__C.YOLO.ANCHORS              = '~/catkin_ws/src/act_recognizer/src/data/anchors/basline_anchors.txt'
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5




