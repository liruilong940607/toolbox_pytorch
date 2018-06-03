from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Monkey-patch because I trained with a newer version.
# This can be removed once PyTorch 0.4.x is out.
# See https://discuss.pytorch.org/t/question-about-rebuild-tensor-v2/14560
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image


import sys
import os 
import time
import numpy as np
import cv2
import argparse
import yaml
import json
import random
import math
import copy
import shutil
import logging
import scipy.sparse
from tqdm import tqdm
from collections import namedtuple
from easydict import EasyDict as edict

from config import CONFIG, config_load

# Load CONFIG
parser = argparse.ArgumentParser(description='Training code')
parser.add_argument('--config', default='config.yaml', type=str, help='yaml config file')
args = parser.parse_args()
config_load(args.config)
# config_load('config.yaml')
print ('==> CONFIG is: \n', CONFIG, '\n')

# Set logger
logger = logging.getLogger(__name__)
format = logging.Formatter("%(asctime)s - %(message)s")    # output format 
sh = logging.StreamHandler(stream=sys.stdout)    # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)
if CONFIG.DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# Create LOG_DIR and SNAPSHOT_DIR
LOGDIR = os.path.join(CONFIG.LOGS.LOG_DIR, '%s_%d'%(CONFIG.NAME, int(time.time())))
SNAPSHOTDIR = os.path.join(CONFIG.LOGS.SNAPSHOT_DIR, '%s_%d'%(CONFIG.NAME, int(time.time())))
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
if not os.path.exists(SNAPSHOTDIR):
    os.makedirs(SNAPSHOTDIR)
    
# Store the code into LOG_DIR/shutil
if CONFIG.LOGS.LOG_SHUTIL_ON:
    SHUTILDIR = os.path.join(LOGDIR, 'shutil')
    if os.path.exists(SHUTILDIR):
        shutil.rmtree(SHUTILDIR)
    SHUTIL_IGNORELIST = [CONFIG.LOGS.SNAPSHOT_DIR, CONFIG.LOGS.LOG_DIR] + \
                        CONFIG.LOGS.LOG_SHUTIL_IGNORELIST
    if os.path.exists(CONFIG.LOGS.LOG_SHUTIL_IGNOREFILE):
        lines = open(CONFIG.LOGS.LOG_SHUTIL_IGNOREFILE).readlines()
    SHUTIL_IGNORELIST += [l.strip() for l in lines]
    print ('==> Shutil Code to File: %s \n'%(SHUTILDIR))
    print ('==> Shutil Ignore Patterns: ', SHUTIL_IGNORELIST, '\n')
    shutil.copytree('./', SHUTILDIR, ignore=shutil.ignore_patterns(*SHUTIL_IGNORELIST))

####################################################################################################
#                                         COCO Dataset
####################################################################################################
from datasets import CocoDatasetMiniBatch, MinibatchSampler

####################################################################################################
#                                         Network Model
####################################################################################################
from resnetXtFPN import resnet50C4
from generate_anchors import generate_anchors

class RPN(nn.Module):
    def __init__(self, dim_in, spatial_scale, pretrainfile=None):
        super(RPN, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_in
        spatial_scale = 1.0 / CONFIG.RPN.STRIDE
        anchors = generate_anchors(
            stride=CONFIG.RPN.STRIDE,
            sizes=CONFIG.RPN.SIZES,
            aspect_ratios=CONFIG.RPN.ASPECT_RATIOS)
        num_anchors = anchors.shape[0]

        # RPN hidden representation
        self.RPN_conv = nn.Conv2d(self.dim_in, self.dim_out, 3, 1, 1)
        # Proposal classification scores
        self.n_score_out =  num_anchors # for sigmoid.
        self.RPN_cls_score = nn.Conv2d(self.dim_out, self.n_score_out, 1, 1, 0)
        # Proposal bbox regression deltas
        self.RPN_bbox_pred = nn.Conv2d(self.dim_out, num_anchors * 4, 1, 1, 0)

        #self.RPN_GenerateProposals = GenerateProposalsOp(anchors, spatial_scale)
        #self.RPN_GenerateProposalLabels = GenerateProposalLabelsOp()
        
    def _init_weights(self):
        init.normal_(self.RPN_conv.weight, std=0.01)
        init.constant_(self.RPN_conv.bias, 0)
        init.normal_(self.RPN_cls_score.weight, std=0.01)
        init.constant_(self.RPN_cls_score.bias, 0)
        init.normal_(self.RPN_bbox_pred.weight, std=0.01)
        init.constant_(self.RPN_bbox_pred.bias, 0)
        
class MaskRCNN(nn.Module):
    def __init__(self, pretrainfile=None):
        super(MaskRCNN, self).__init__()
        
        # Backbone
        if CONFIG.MODEL.CONV_BODY == 'resnet50C4':
            self.Conv_Body = resnet50C4(pretrained=True, num_classes=None)
            spatial_scale = 1. / 16.
            dim_out = 1024
        elif CONFIG.MODEL.CONV_BODY == 'resnet50FPN':
            # FPN : out is in the order of [p1, p2, p3, p4]. from 1./4. to 1./32. 
            # The order is different here. **REMENBER** to transpose the order in the forward().
            self.Conv_Body = resnet50C4(pretrained=True, num_classes=None)
            spatial_scale = (1. / 32., 1. / 16., 1. / 8., 1. / 4.)
            dim_out = (2048, 1024, 512, 256)
        else:
            raise NotImplementedError 
            
        # TODO: here
        # Region Proposal Network
        if CONFIG.RPN.RPN_ON:
            self.rpn = RPN(dim_in, spatial_scale)
        
        if CONFIG.FPN.FPN_ON:
            # Only supports case when RPN and ROI min levels are the same
            assert CONFIG.FPN.RPN_MIN_LEVEL == CONFIG.FPN.ROI_MIN_LEVEL
            # RPN max level can be >= to ROI max level
            assert CONFIG.FPN.RPN_MAX_LEVEL >= CONFIG.FPN.ROI_MAX_LEVEL
            # FPN RPN max level might be > FPN ROI max level in which case we
            # need to discard some leading conv blobs (blobs are ordered from
            # max/coarsest level to min/finest level)
            self.num_roi_levels = CONFIG.FPN.ROI_MAX_LEVEL - CONFIG.FPN.ROI_MIN_LEVEL + 1

            # Retain only the spatial scales that will be used for RoI heads. `Conv_Body.spatial_scale`
            # may include extra scales that are used for RPN proposals, but not for RoI heads.
            self.Conv_Body.spatial_scale = self.Conv_Body.spatial_scale[-self.num_roi_levels:]

        # BBOX Branch
        if not CONFIG.MODEL.RPN_ONLY:
            self.Box_Head = get_func(CONFIG.FAST_RCNN.ROI_BOX_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(
                self.Box_Head.dim_out)

        # Mask Branch
        if CONFIG.MODEL.MASK_ON:
            self.Mask_Head = get_func(CONFIG.MRCNN.ROI_MASK_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Mask_Head, 'SHARE_RES5', False):
                self.Mask_Head.share_res5_module(self.Box_Head.res5)
            self.Mask_Outs = mask_rcnn_heads.mask_rcnn_outputs(self.Mask_Head.dim_out)

        # Keypoints Branch
        if CONFIG.MODEL.KEYPOINTS_ON:
            self.Keypoint_Head = get_func(CONFIG.KRCNN.ROI_KEYPOINTS_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                self.Keypoint_Head.share_res5_module(self.Box_Head.res5)
            self.Keypoint_Outs = keypoint_rcnn_heads.keypoint_outputs(self.Keypoint_Head.dim_out)

        self._init_modules()
        
        self.init(pretrainfile)
        
    def init(self, pretrainfile=None):
        if pretrainfile is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, .1)
                    m.bias.data.zero_()
        else:
            self.load_state_dict(torch.load(pretrainfile, map_location=lambda storage, loc: storage))
            print ('==> load self-train weights as pretrain.')

    
    def forward(self, input_local, input_global):
        pass
        
    
    def calc_loss(self, pred, gt):
        loss = nn.BCEWithLogitsLoss()(pred, gt)
        return loss
    
## main ##
if __name__ == '__main__':
    print ('===>dataset proprocess.... ')
    dataset = CocoDatasetMiniBatch(CONFIG.MYDATASET.TRAIN_DIR, CONFIG.MYDATASET.TRAIN_ANNOFILE,
                          gt=True, crowd_filter_thresh=CONFIG.TRAIN.CROWD_FILTER_THRESH)
    sampler = MinibatchSampler(dataset.ratio_list, dataset.ratio_index)
    print ('===>done. ')
    
    batch_size = len(CONFIG.MYSOLVER.GPU_IDS) * CONFIG.TRAIN.IMS_PER_BATCH
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=CONFIG.MYSOLVER.WORKERS,
        collate_fn=dataset.collate_minibatch)
    
    print ('===>model building .... ')
    maskRCNN = MaskRCNN(pretrainfile=None)
    print ('===>done. ')
    
    print ('===>start training .... ')
    for input_data in tqdm(dataloader):
        for key in input_data:
            if key != 'roidb': # roidb is a list of ndarrays with inconsistent length
                input_data[key] = list(map(Variable, input_data[key]))

        # net_outputs = maskRCNN(**input_data)
        # loss = net_outputs['total_loss']
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        pass