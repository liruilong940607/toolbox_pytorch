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

FieldOfAnchors = namedtuple(
    'FieldOfAnchors', [
        'field_of_anchors', 'num_cell_anchors', 'stride', 'field_size'
    ]
)

_threadlocal_foa = {}

from generate_anchors import generate_anchors

def get_field_of_anchors(stride, anchor_sizes, anchor_aspect_ratios):
    global _threadlocal_foa
    cache_key = str(stride) + str(anchor_sizes) + str(anchor_aspect_ratios)
    if cache_key in _threadlocal_foa.keys():
        return _threadlocal_foa[cache_key]

    # Anchors at a single feature cell
    cell_anchors = generate_anchors(
        stride=stride, sizes=anchor_sizes, aspect_ratios=anchor_aspect_ratios
    )
    num_cell_anchors = cell_anchors.shape[0]

    # Generate canonical proposals from shifted anchors
    # Enumerate all shifted positions on the (H, W) grid
    fpn_max_size = CONFIG.FPN.COARSEST_STRIDE * np.ceil(
        CONFIG.TRAIN.MAX_SIZE / float(CONFIG.FPN.COARSEST_STRIDE)
    )
    field_size = int(np.ceil(fpn_max_size / float(stride)))
    shifts = np.arange(0, field_size) * stride
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.ravel()
    shift_y = shift_y.ravel()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()

    # Broacast anchors over shifts to enumerate all anchors at all positions
    # in the (H, W) grid:
    #   - add A cell anchors of shape (1, A, 4) to
    #   - K shifts of shape (K, 1, 4) to get
    #   - all shifted anchors of shape (K, A, 4)
    #   - reshape to (K*A, 4) shifted anchors
    A = num_cell_anchors
    K = shifts.shape[0]
    field_of_anchors = (
        cell_anchors.reshape((1, A, 4)) +
        shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    )
    # print ('fpn_max_size:', fpn_max_size)
    # print ('field_size:', field_size)
    # print ('stride:', stride)
    # print ('shift_x:', shift_x.shape)
    # print ('shifts:', shifts.shape)
    # print ('field_of_anchors:', field_of_anchors.shape)
    
    field_of_anchors = field_of_anchors.reshape((K * A, 4))
    foa = FieldOfAnchors(
        field_of_anchors=field_of_anchors.astype(np.float32),
        num_cell_anchors=num_cell_anchors,
        stride=stride,
        field_size=field_size,
    )
    _threadlocal_foa[cache_key] = foa
    return foa

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask
import pycocotools.mask as mask_util


class CocoDataset():
    def __init__(self, ImageRoot, AnnoFile):
        self.imgroot = ImageRoot
        self.COCO = COCO(AnnoFile)
        # Set up dataset classes
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.COCO.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }
        # print ('self.category_to_id_map:', self.category_to_id_map)
        # print ('self.classes:', self.classes)
        # print ('self.num_classes:', self.num_classes)
        # print ('self.json_category_id_to_contiguous_id:', self.json_category_id_to_contiguous_id)
        # print ('self.contiguous_category_id_to_json_id:', self.contiguous_category_id_to_json_id)
        # logger.info('self.num_classes: %d' % self.num_classes)
        
        
        self.image_ids = self.COCO.getImgIds()
        self.image_ids.sort()
        if CONFIG.DEBUG:
            self.image_ids = self.image_ids[0:128]
        
#         self.filterednum = 0
#         roidb = []
#         for i in tqdm(range(self.__len__())):
#             roidb += [self.get_data(i)]
#         self._compute_and_log_stats(roidb)
#         print (self.__len__())
#         logger.info('Filtered %d roidb entries'%self.filterednum)
#         print (self.__len__()-self.filterednum)
        
        self.foa = get_field_of_anchors(CONFIG.RPN.STRIDE, CONFIG.RPN.SIZES, CONFIG.RPN.ASPECT_RATIOS)
        self.all_anchors = self.foa.field_of_anchors
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        rawdata = self.get_data(idx)
        im_blob, im_scales = self.get_image_blob(rawdata)
        
        # get_minibatch_blob_names(), get_rpn_blob_names()
        # Single level RPN blobs
        blobs = {
            'im_info': [],
            'rpn_labels_int32_wide': [],
            'rpn_bbox_targets_wide': [],
            'rpn_bbox_inside_weights_wide': [],
            'rpn_bbox_outside_weights_wide': []
        }
        blobs['data'] = im_blob
        valid = self.add_rpn_blobs(blobs, im_scales, rawdata)
        
        # Squeeze batch dim
        for key in blobs:
            if key != 'roidb':
                blobs[key] = blobs[key].squeeze(axis=0)
                        
        # blobs['roidb'] = blob_utils.serialize(blobs['roidb']) ???
        return blobs
    
    def collate_minibatch(self, list_of_blobs):
        """Stack samples seperately and return a list of minibatches
        A batch contains NUM_GPUS minibatches and image size in different minibatch may be different.
        Hence, we need to stack smaples from each minibatch seperately.
        """
        def pad_image_data(list_of_blobs):
            max_shape = np.array([blobs['data'].shape[1:] for blobs in list_of_blobs]).max(axis=0)
            output_list = []
            for blobs in list_of_blobs:
                data_padded = np.zeros((3, max_shape[0], max_shape[1]), dtype=np.float32)
                _, h, w = blobs['data'].shape
                data_padded[:, :h, :w] = blobs['data']
                blobs['data'] = data_padded
                output_list.append(blobs)
            return output_list
        
        Batch = {key: [] for key in list_of_blobs[0]}
        # Because roidb consists of entries of variable length, it can't be batch into a tensor.
        # So we keep roidb in the type of "list of ndarray".
        list_of_roidb = [blobs.pop('roidb') for blobs in list_of_blobs]
        for i in range(0, len(list_of_blobs), CONFIG.SOLVER.IMS_PER_BATCH):
            mini_list = list_of_blobs[i:(i + CONFIG.SOLVER.IMS_PER_BATCH)]
            # Pad image data
            mini_list = pad_image_data(mini_list)
            minibatch = default_collate(mini_list)
            minibatch['roidb'] = list_of_roidb[i:(i + CONFIG.SOLVER.IMS_PER_BATCH)]
            for key in minibatch:
                Batch[key].append(minibatch[key])

        return Batch

    def _get_rpn_blobs(self, im_height, im_width, foas, all_anchors, gt_boxes):
        total_anchors = all_anchors.shape[0]
        straddle_thresh = CONFIG.TRAIN.RPN_STRADDLE_THRESH

        if straddle_thresh >= 0:
            # Only keep anchors inside the image by a margin of straddle_thresh
            # Set TRAIN.RPN_STRADDLE_THRESH to -1 (or a large value) to keep all
            # anchors
            inds_inside = np.where(
                (all_anchors[:, 0] >= -straddle_thresh) &
                (all_anchors[:, 1] >= -straddle_thresh) &
                (all_anchors[:, 2] < im_width + straddle_thresh) &
                (all_anchors[:, 3] < im_height + straddle_thresh)
            )[0]
            # keep only inside anchors
            anchors = all_anchors[inds_inside, :]
        else:
            inds_inside = np.arange(all_anchors.shape[0])
            anchors = all_anchors
        num_inside = len(inds_inside)

        # logger.debug('total_anchors: %d', total_anchors)
        # logger.debug('inds_inside: %d', num_inside)
        # logger.debug('anchors.shape: %s', str(anchors.shape))

        # Compute anchor labels:
        # label=1 is positive, 0 is negative, -1 is don't care (ignore)
        labels = np.empty((num_inside, ), dtype=np.int32)
        labels.fill(-1)
        if len(gt_boxes) > 0:
            # Compute overlaps between the anchors and the gt boxes overlaps
            anchor_by_gt_overlap = bbox_overlaps(anchors, gt_boxes)
            # Map from anchor to gt box that has highest overlap
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
            # For each anchor, amount of overlap with most overlapping gt box
            anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_inside),
                                                    anchor_to_gt_argmax]

            # Map from gt box to an anchor that has highest overlap
            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
            # For each gt box, amount of overlap with most overlapping anchor
            gt_to_anchor_max = anchor_by_gt_overlap[
                gt_to_anchor_argmax,
                np.arange(anchor_by_gt_overlap.shape[1])
            ]
            # Find all anchors that share the max overlap amount
            # (this includes many ties)
            anchors_with_max_overlap = np.where(
                anchor_by_gt_overlap == gt_to_anchor_max
            )[0]

            # Fg label: for each gt use anchors with highest overlap
            # (including ties)
            labels[anchors_with_max_overlap] = 1
            # Fg label: above threshold IOU
            labels[anchor_to_gt_max >= CONFIG.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        # subsample positive labels if we have too many
        num_fg = int(CONFIG.TRAIN.RPN_FG_FRACTION * CONFIG.TRAIN.RPN_BATCH_SIZE_PER_IM)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = np.random.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False
            )
            labels[disable_inds] = -1
        fg_inds = np.where(labels == 1)[0]

        # subsample negative labels if we have too many
        # (samples with replacement, but since the set of bg inds is large most
        # samples will not have repeats)
        num_bg = CONFIG.TRAIN.RPN_BATCH_SIZE_PER_IM - np.sum(labels == 1)
        bg_inds = np.where(anchor_to_gt_max < CONFIG.TRAIN.RPN_NEGATIVE_OVERLAP)[0]
        if len(bg_inds) > num_bg:
            enable_inds = bg_inds[np.random.randint(len(bg_inds), size=num_bg)]
            labels[enable_inds] = 0
        bg_inds = np.where(labels == 0)[0]

        bbox_targets = np.zeros((num_inside, 4), dtype=np.float32)
        bbox_targets[fg_inds, :] = bbox_transform_inv(
            anchors[fg_inds, :], gt_boxes[anchor_to_gt_argmax[fg_inds], :]).astype(np.float32, copy=False)

        # Bbox regression loss has the form:
        #   loss(x) = weight_outside * L(weight_inside * x)
        # Inside weights allow us to set zero loss on an element-wise basis
        # Bbox regression is only trained on positive examples so we set their
        # weights to 1.0 (or otherwise if config is different) and 0 otherwise
        bbox_inside_weights = np.zeros((num_inside, 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = (1.0, 1.0, 1.0, 1.0)

        # The bbox regression loss only averages by the number of images in the
        # mini-batch, whereas we need to average by the total number of example
        # anchors selected
        # Outside weights are used to scale each element-wise loss so the final
        # average over the mini-batch is correct
        bbox_outside_weights = np.zeros((num_inside, 4), dtype=np.float32)
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)
        bbox_outside_weights[labels == 1, :] = 1.0 / num_examples
        bbox_outside_weights[labels == 0, :] = 1.0 / num_examples

        # Map up to original set of anchors
        labels = self._unmap(labels, total_anchors, inds_inside, fill=-1)
        
        bbox_targets = self._unmap(
            bbox_targets, total_anchors, inds_inside, fill=0
        )
        bbox_inside_weights = self._unmap(
            bbox_inside_weights, total_anchors, inds_inside, fill=0
        )
        bbox_outside_weights = self._unmap(
            bbox_outside_weights, total_anchors, inds_inside, fill=0
        )

        # Split the generated labels, etc. into labels per each field of anchors
        blobs_out = []
        start_idx = 0
        for foa in foas:
            H = foa.field_size
            W = foa.field_size
            A = foa.num_cell_anchors
            end_idx = start_idx + H * W * A
            _labels = labels[start_idx:end_idx]
            _bbox_targets = bbox_targets[start_idx:end_idx, :]
            _bbox_inside_weights = bbox_inside_weights[start_idx:end_idx, :]
            _bbox_outside_weights = bbox_outside_weights[start_idx:end_idx, :]
            start_idx = end_idx

            # labels output with shape (1, A, height, width)
            _labels = _labels.reshape((1, H, W, A)).transpose(0, 3, 1, 2)
            # bbox_targets output with shape (1, 4 * A, height, width)
            _bbox_targets = _bbox_targets.reshape(
                (1, H, W, A * 4)).transpose(0, 3, 1, 2)
            # bbox_inside_weights output with shape (1, 4 * A, height, width)
            _bbox_inside_weights = _bbox_inside_weights.reshape(
                (1, H, W, A * 4)).transpose(0, 3, 1, 2)
            # bbox_outside_weights output with shape (1, 4 * A, height, width)
            _bbox_outside_weights = _bbox_outside_weights.reshape(
                (1, H, W, A * 4)).transpose(0, 3, 1, 2)
            blobs_out.append(
                dict(
                    rpn_labels_int32_wide=_labels,
                    rpn_bbox_targets_wide=_bbox_targets,
                    rpn_bbox_inside_weights_wide=_bbox_inside_weights,
                    rpn_bbox_outside_weights_wide=_bbox_outside_weights
                )
            )
        return blobs_out[0] if len(blobs_out) == 1 else blobs_out

    
    def add_rpn_blobs(self, blobs, im_scales, rawdata):
        """Add blobs needed training RPN-only and end-to-end Faster R-CNN models."""
        im_i = 0
        scale = im_scales[im_i]
        im_height = np.round(rawdata['height'] * scale)
        im_width = np.round(rawdata['width'] * scale)
        gt_inds = np.where(
            (rawdata['gt_classes'] > 0) & (rawdata['is_crowd'] == 0)
        )[0]
        gt_rois = rawdata['boxes'][gt_inds, :] * scale
        # TODO(rbg): gt_boxes is poorly named;
        # should be something like 'gt_rois_info'
        gt_boxes = np.zeros((len(gt_inds), 6), dtype=np.float32)
        gt_boxes[:, 0] = im_i  # batch inds
        gt_boxes[:, 1:5] = gt_rois
        gt_boxes[:, 5] = rawdata['gt_classes'][gt_inds]
        im_info = np.array([[im_height, im_width, scale]], dtype=np.float32)
        blobs['im_info'].append(im_info)

        # Add RPN targets
        # Classical RPN, applied to a single feature level
        rpn_blobs = self._get_rpn_blobs(im_height, im_width, [self.foa], self.all_anchors, gt_rois)
        for k, v in rpn_blobs.items():
            blobs[k].append(v)
        
        #
        for k, v in blobs.items():
            if isinstance(v, list) and len(v) > 0:
                blobs[k] = np.concatenate(v)
            
        valid_keys = [
            'has_visible_keypoints', 'boxes', 'segms', 'seg_areas', 'gt_classes',
            'gt_overlaps', 'is_crowd', 'box_to_gt_ind_map', 'gt_keypoints'
        ]
        minimal_roidb = [{} for _ in range(1)]
        i = 0
        e = rawdata
        for k in valid_keys:
            if k in e:
                minimal_roidb[i][k] = e[k]
        # blobs['roidb'] = blob_utils.serialize(minimal_roidb)
        blobs['roidb'] = minimal_roidb
        
        
        # Always return valid=True, since RPN minibatches are valid by design
        return True
    
    def get_image_blob(self, rawdata):
        """Builds an input blob from the images in the roidb at the specified
        scales.
        """
        # Sample random scales to use for each image in this batch
        scale_inds = np.random.randint(
            0, high=len(CONFIG.TRAIN.SCALES), size=1)
        processed_ims = []
        im_scales = []
        
        im = cv2.imread(rawdata['image'])
        if rawdata['flipped']:
            im = im[:, ::-1, :]
        target_size = CONFIG.TRAIN.SCALES[scale_inds[0]]
        im, im_scale = self._prep_im_for_blob(
            im, CONFIG.DATASET.MEAN, [target_size], CONFIG.TRAIN.MAX_SIZE)
        
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

        # Create a blob to hold the input images [n, c, h, w]
        blob = self._im_list_to_blob(processed_ims)

        return blob, im_scales

    def _im_list_to_blob(self, ims):
        """Convert a list of images into a network input. Assumes images were
        prepared using prep_im_for_blob or equivalent: i.e.
          - BGR channel order
          - pixel means subtracted
          - resized to the desired input size
          - float32 numpy ndarray format
        Output is a 4D HCHW tensor of the images concatenated along axis 0 with
        shape.
        """
        if not isinstance(ims, list):
            ims = [ims]
        max_shape = np.array([im.shape[:2] for im in ims]).max(axis=0) # get_max_shape()

        num_images = len(ims)
        blob = np.zeros(
            (num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
        for i in range(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
        # Move channels (axis 3) to axis 1
        # Axis order will become: (batch elem, channel, height, width)
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        return blob
    
    def _prep_im_for_blob(self, im, pixel_means, target_sizes, max_size):
        """Prepare an image for use as a network input blob. Specially:
          - Subtract per-channel pixel mean
          - Convert to float32
          - Rescale to each of the specified target size (capped at max_size)
        Returns a list of transformed images, one for each target size. Also returns
        the scale factors that were used to compute each returned image.
        """
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        ims = []
        im_scales = []
        for target_size in target_sizes:
            im_scale = self._get_target_scale(im_size_min, im_size_max, target_size, max_size)
            im_resized = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                                    interpolation=cv2.INTER_LINEAR)
            ims.append(im_resized)
            im_scales.append(im_scale)
        return ims, im_scales
    
    def _get_target_scale(self, im_size_min, im_size_max, target_size, max_size):
        """Calculate target resize scale
        """
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        return im_scale

    def get_data(self, idx):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        image_id = self.image_ids[idx]
        datainfo = self.COCO.loadImgs(image_id)[0]
        rawdata = {
            'id': image_id,
            'filename': datainfo['file_name'],
            'image': os.path.join(self.imgroot, datainfo['file_name']),
            'width': datainfo['width'],
            'height': datainfo['height'],
            
            'flipped': False,
            'boxes': np.empty((0, 4), dtype=np.float32),
            'segms': [],
            'gt_classes': np.empty((0), dtype=np.int32),
            'seg_areas': np.empty((0), dtype=np.float32),
            'gt_overlaps': scipy.sparse.csr_matrix(
                            np.empty((0, self.num_classes), dtype=np.float32)
                            ),
            'is_crowd': np.empty((0), dtype=np.bool),
            'box_to_gt_ind_map': np.empty((0), dtype=np.int32)
        }
        self._add_gt_annotations(rawdata)
        self._add_class_assignments(rawdata)
        if CONFIG.AUGS.FLIP_ON and random.random()<0.5:
            self._flip(rawdata)
        if not self._is_valid(rawdata):
            # self.filterednum += 1
            return self.get_data(idx+1) # DIFF: may get same data twice.
        rawdata['bbox_targets'] = compute_targets(rawdata)
        
        return rawdata
        
    def _compute_and_log_stats(self, roidb):
        classes = self.classes
        char_len = np.max([len(c) for c in classes])
        hist_bins = np.arange(len(classes) + 1)

        # Histogram of ground-truth objects
        gt_hist = np.zeros((len(classes)), dtype=np.int)
        for rawdata in roidb:
            gt_inds = np.where(
                (rawdata['gt_classes'] > 0) & (rawdata['is_crowd'] == 0))[0]
            gt_classes = rawdata['gt_classes'][gt_inds]
            gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
        logger.debug('Ground-truth class histogram:')
        for i, v in enumerate(gt_hist):
            logger.debug(
                '{:d}{:s}: {:d}'.format(
                    i, classes[i].rjust(char_len), v))
        logger.debug('-' * char_len)
        logger.debug(
            '{:s}: {:d}'.format(
                'total'.rjust(char_len), np.sum(gt_hist)))

    def _unmap(self, data, count, inds, fill=0):
        """Unmap a subset of item (data) back to the original set of items (of
        size count)"""
        if count == len(inds):
            return data

        if len(data.shape) == 1:
            ret = np.empty((count, ), dtype=data.dtype)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count, ) + data.shape[1:], dtype=data.dtype)
            ret.fill(fill)
            ret[inds, :] = data
        return ret 
    
    def _is_valid(self, rawdata):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = rawdata['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= CONFIG.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < CONFIG.TRAIN.BG_THRESH_HI) &
                           (overlaps >= CONFIG.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid
    
    def _flip(self, rawdata):
        width = rawdata['width']
        boxes = rawdata['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = width - oldx2 - 1
        boxes[:, 2] = width - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        rawdata['boxes'] = boxes
        rawdata['segms'] = self.flip_segms(
            rawdata['segms'], rawdata['height'], rawdata['width']
        )
        rawdata['flipped'] = True
         
    def _add_class_assignments(self, rawdata):
        """Compute object category assignment for each box associated with each
        roidb entry.
        """
        gt_overlaps = rawdata['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        rawdata['max_classes'] = max_classes
        rawdata['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

    def _add_gt_annotations(self, rawdata):
        ann_ids = self.COCO.getAnnIds(imgIds=rawdata['id'], iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []
        width = rawdata['width']
        height = rawdata['height']
        for obj in objs:
            # crowd regions are RLE encoded and stored as dicts
            if isinstance(obj['segmentation'], list):
                # Valid polygons have >= 3 points, so require >= 6 coordinates
                obj['segmentation'] = [
                    p for p in obj['segmentation'] if len(p) >= 6
                ]
            if obj['area'] < CONFIG.TRAIN.GT_MIN_AREA:
                continue
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
            x1, y1, x2, y2 = xywh_to_xyxy(obj['bbox'])
            x1, y1, x2, y2 = self.clip_xyxy_to_image(
                x1, y1, x2, y2, height, width
            )
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                valid_segms.append(obj['segmentation'])
        num_valid_objs = len(valid_objs)
        
        boxes = np.zeros((num_valid_objs, 4), dtype=rawdata['boxes'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=rawdata['gt_classes'].dtype)
        gt_overlaps = np.zeros(
            (num_valid_objs, self.num_classes),
            dtype=rawdata['gt_overlaps'].dtype
        )
        seg_areas = np.zeros((num_valid_objs), dtype=rawdata['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=rawdata['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            (num_valid_objs), dtype=rawdata['box_to_gt_ind_map'].dtype
        )
        
        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            is_crowd[ix] = obj['iscrowd']
            box_to_gt_ind_map[ix] = ix
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                gt_overlaps[ix, :] = -1.0
            else:
                gt_overlaps[ix, cls] = 1.0
        rawdata['boxes'] = np.append(rawdata['boxes'], boxes, axis=0)
        rawdata['segms'].extend(valid_segms)
        rawdata['gt_classes'] = np.append(rawdata['gt_classes'], gt_classes)
        rawdata['seg_areas'] = np.append(rawdata['seg_areas'], seg_areas)
        rawdata['gt_overlaps'] = np.append(
            rawdata['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        rawdata['gt_overlaps'] = scipy.sparse.csr_matrix(rawdata['gt_overlaps'])
        rawdata['is_crowd'] = np.append(rawdata['is_crowd'], is_crowd)
        rawdata['box_to_gt_ind_map'] = np.append(
            rawdata['box_to_gt_ind_map'], box_to_gt_ind_map
        )
        
    
        
    def flip_segms(self, segms, height, width):
        """Left/right flip each mask in a list of masks."""

        def _flip_poly(poly, width):
            flipped_poly = np.array(poly)
            flipped_poly[0::2] = width - np.array(poly[0::2]) - 1
            return flipped_poly.tolist()

        def _flip_rle(rle, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                # Magic RLE format handling painfully discovered by looking at the
                # COCO API showAnns function.
                rle = mask_util.frPyObjects([rle], height, width)
            mask = mask_util.decode(rle)
            mask = mask[:, ::-1, :]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        flipped_segms = []
        for segm in segms:
            if type(segm) == list:
                # Polygon format
                flipped_segms.append([_flip_poly(poly, width) for poly in segm])
            else:
                # RLE format
                assert type(segm) == dict
                flipped_segms.append(_flip_rle(segm, height, width))
        return flipped_segms
        
    def clip_xyxy_to_image(self, x1, y1, x2, y2, height, width):
        """Clip coordinates to an image with the given height and width."""
        x1 = np.minimum(width - 1., np.maximum(0., x1))
        y1 = np.minimum(height - 1., np.maximum(0., y1))
        x2 = np.minimum(width - 1., np.maximum(0., x2))
        y2 = np.minimum(height - 1., np.maximum(0., y2))
        return x1, y1, x2, y2
        
        
        
        
####################################################################################################
#                                         Network Model
####################################################################################################
from resnetXtFPN import resnet50C4
  
class GenerateProposalLabelsOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rpn_rois, roidb, im_info):
        """Op for generating training labels for RPN proposals. This is used
        when training RPN jointly with Fast/Mask R-CNN (as in end-to-end
        Faster R-CNN training).

        blobs_in:
          - 'rpn_rois': 2D tensor of RPN proposals output by GenerateProposals
          - 'roidb': roidb entries that will be labeled
          - 'im_info': See GenerateProposals doc.

        blobs_out:
          - (variable set of blobs): returns whatever blobs are required for
            training the model. It does this by querying the data loader for
            the list of blobs that are needed.
        """
        im_scales = im_info.data.numpy()[:, 2]

        # get_fast_rcnn_blob_names()
        output_blob_names = ['rois', 
                      'labels_int32', 'bbox_targets', 'bbox_inside_weights', 'bbox_outside_weights',
                      'mask_rois', 'roi_has_mask_int32', 'masks_int32']
        
        # For historical consistency with the original Faster R-CNN
        # implementation we are *not* filtering crowd proposals.
        # This choice should be investigated in the future (it likely does
        # not matter).
        # Note: crowd_thresh=0 will ignore _filter_crowd_proposals
        self.add_proposals(roidb, rpn_rois, im_scales, crowd_thresh=0)
        blobs = {k: [] for k in output_blob_names}
        self.add_fast_rcnn_blobs(blobs, im_scales, roidb)

        return blobs
    
    def add_fast_rcnn_blobs(self, blobs, im_scales, roidb):
        """Add blobs needed for training Fast R-CNN style models."""
        # Sample training RoIs from each image and append them to the blob lists
        for im_i, entry in enumerate(roidb):
            frcn_blobs = self._sample_rois(entry, im_scales[im_i], im_i)
            for k, v in frcn_blobs.items():
                blobs[k].append(v)
        # Concat the training blob lists into tensors
        for k, v in blobs.items():
            if isinstance(v, list) and len(v) > 0:
                blobs[k] = np.concatenate(v)
                
        # Perform any final work and validity checks after the collating blobs for
        # all minibatch images
        valid = True
        
        return valid
    
    def _sample_rois(self, roidb, im_scale, batch_idx):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        rois_per_image = int(CONFIG.TRAIN.BATCH_SIZE_PER_IM)
        fg_rois_per_image = int(np.round(CONFIG.TRAIN.FG_FRACTION * rois_per_image))
        max_overlaps = roidb['max_overlaps']

        # Select foreground RoIs as those with >= FG_THRESH overlap
        fg_inds = np.where(max_overlaps >= CONFIG.TRAIN.FG_THRESH)[0]
        # Guard against the case when an image has fewer than fg_rois_per_image
        # foreground RoIs
        fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
        # Sample foreground regions without replacement
        if fg_inds.size > 0:
            fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((max_overlaps < CONFIG.TRAIN.BG_THRESH_HI) &
                           (max_overlaps >= CONFIG.TRAIN.BG_THRESH_LO))[0]
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_inds.size)
        # Sample foreground regions without replacement
        if bg_inds.size > 0:
            bg_inds = np.random.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

        # The indices that we're selecting (both fg and bg)
        keep_inds = np.append(fg_inds, bg_inds)
        # Label is the class each RoI has max overlap with
        sampled_labels = roidb['max_classes'][keep_inds]
        sampled_labels[fg_rois_per_this_image:] = 0  # Label bg RoIs with class 0
        sampled_boxes = roidb['boxes'][keep_inds]

        if 'bbox_targets' not in roidb:
            gt_inds = np.where(roidb['gt_classes'] > 0)[0]
            gt_boxes = roidb['boxes'][gt_inds, :]
            gt_assignments = gt_inds[roidb['box_to_gt_ind_map'][keep_inds]]
            bbox_targets = compute_targets(
                sampled_boxes, gt_boxes[gt_assignments, :], sampled_labels)
            bbox_targets, bbox_inside_weights = expand_bbox_targets(bbox_targets)
        else:
            bbox_targets, bbox_inside_weights = expand_bbox_targets(
                roidb['bbox_targets'][keep_inds, :])

        bbox_outside_weights = np.array(
            bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype)

        # Scale rois and format as (batch_idx, x1, y1, x2, y2)
        sampled_rois = sampled_boxes * im_scale
        repeated_batch_idx = batch_idx * np.ones((sampled_rois.shape[0], 1), dtype=np.float32)
        sampled_rois = np.hstack((repeated_batch_idx, sampled_rois))

        # Base Fast R-CNN blobs
        blob_dict = dict(
            labels_int32=sampled_labels.astype(np.int32, copy=False),
            rois=sampled_rois,
            bbox_targets=bbox_targets,
            bbox_inside_weights=bbox_inside_weights,
            bbox_outside_weights=bbox_outside_weights)

        # Optionally add Mask R-CNN blobs
        roi_data.mask_rcnn.add_mask_rcnn_blobs(blob_dict, sampled_boxes, roidb,
                                               im_scale, batch_idx)

        return blob_dict
    
    def add_proposals(self, roidb, rois, scales, crowd_thresh):
        """Add proposal boxes (rois) to an roidb that has ground-truth annotations
        but no proposals. If the proposals are not at the original image scale,
        specify the scale factor that separate them in scales.
        """
        box_list = []
        for i in range(len(roidb)):
            inv_im_scale = 1. / scales[i]
            idx = np.where(rois[:, 0] == i)[0]
            box_list.append(rois[idx, 1:] * inv_im_scale)
        self._merge_proposal_boxes_into_roidb(roidb, box_list)
        if crowd_thresh > 0:
            self._filter_crowd_proposals(roidb, crowd_thresh)
        self._add_class_assignments(roidb)
        
    def _merge_proposal_boxes_into_roidb(self, roidb, box_list):
        """Add proposal boxes to each roidb entry."""
        assert len(box_list) == len(roidb)
        for i, entry in enumerate(roidb):
            boxes = box_list[i]
            num_boxes = boxes.shape[0]
            gt_overlaps = np.zeros(
                (num_boxes, entry['gt_overlaps'].shape[1]),
                dtype=entry['gt_overlaps'].dtype
            )
            box_to_gt_ind_map = -np.ones(
                (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
            )

            # Note: unlike in other places, here we intentionally include all gt
            # rois, even ones marked as crowd. Boxes that overlap with crowds will
            # be filtered out later (see: _filter_crowd_proposals).
            gt_inds = np.where(entry['gt_classes'] > 0)[0]
            if len(gt_inds) > 0:
                gt_boxes = entry['boxes'][gt_inds, :]
                gt_classes = entry['gt_classes'][gt_inds]
                proposal_to_gt_overlaps = bbox_overlaps(
                    boxes.astype(dtype=np.float32, copy=False),
                    gt_boxes.astype(dtype=np.float32, copy=False)
                )
                # Gt box that overlaps each input box the most
                # (ties are broken arbitrarily by class order)
                argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
                # Amount of that overlap
                maxes = proposal_to_gt_overlaps.max(axis=1)
                # Those boxes with non-zero overlap with gt boxes
                I = np.where(maxes > 0)[0]
                # Record max overlaps with the class of the appropriate gt box
                gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
                box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]
            entry['boxes'] = np.append(
                entry['boxes'],
                boxes.astype(entry['boxes'].dtype, copy=False),
                axis=0
            )
            entry['gt_classes'] = np.append(
                entry['gt_classes'],
                np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
            )
            entry['seg_areas'] = np.append(
                entry['seg_areas'],
                np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
            )
            entry['gt_overlaps'] = np.append(
                entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
            )
            entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
            entry['is_crowd'] = np.append(
                entry['is_crowd'],
                np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
            )
            entry['box_to_gt_ind_map'] = np.append(
                entry['box_to_gt_ind_map'],
                box_to_gt_ind_map.astype(
                    entry['box_to_gt_ind_map'].dtype, copy=False
                )
            )

    def _filter_crowd_proposals(self, roidb, crowd_thresh):
        """Finds proposals that are inside crowd regions and marks them as
        overlap = -1 with each ground-truth rois, which means they will be excluded
        from training.
        """
        for entry in roidb:
            gt_overlaps = entry['gt_overlaps'].toarray()
            crowd_inds = np.where(entry['is_crowd'] == 1)[0]
            non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
            if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
                continue
            crowd_boxes = xyxy_to_xywh(entry['boxes'][crowd_inds, :])
            non_gt_boxes = xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
            iscrowd_flags = [int(True)] * len(crowd_inds)
            ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
            bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
            gt_overlaps[non_gt_inds[bad_inds], :] = -1
            entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)

    def _add_class_assignments(self, roidb):
        """Compute object category assignment for each box associated with each
        roidb entry.
        """
        for entry in roidb:
            gt_overlaps = entry['gt_overlaps'].toarray()
            # max overlap with gt over classes (columns)
            max_overlaps = gt_overlaps.max(axis=1)
            # gt class that had the max overlap
            max_classes = gt_overlaps.argmax(axis=1)
            entry['max_classes'] = max_classes
            entry['max_overlaps'] = max_overlaps
            # sanity checks
            # if max overlap is 0, the class must be background (class 0)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)
            # if max overlap > 0, the class must be a fg class (not class 0)
            nonzero_inds = np.where(max_overlaps > 0)[0]
            assert all(max_classes[nonzero_inds] != 0)
    

class GenerateProposalsOp(nn.Module):
    def __init__(self, anchors, spatial_scale):
        super().__init__()
        self._anchors = anchors
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = 1. / spatial_scale

    def forward(self, rpn_cls_prob, rpn_bbox_pred, im_info):
        """Op for generating RPN porposals.

        blobs_in:
          - 'rpn_cls_probs': 4D tensor of shape (N, A, H, W), where N is the
            number of minibatch images, A is the number of anchors per
            locations, and (H, W) is the spatial size of the prediction grid.
            Each value represents a "probability of object" rating in [0, 1].
          - 'rpn_bbox_pred': 4D tensor of shape (N, 4 * A, H, W) of predicted
            deltas for transformation anchor boxes into RPN proposals.
          - 'im_info': 2D tensor of shape (N, 3) where the three columns encode
            the input image's [height, width, scale]. Height and width are
            for the input to the network, not the original image; scale is the
            scale factor used to scale the original image to the network input
            size.

        blobs_out:
          - 'rpn_rois': 2D tensor of shape (R, 5), for R RPN proposals where the
            five columns encode [batch ind, x1, y1, x2, y2]. The boxes are
            w.r.t. the network input, which is a *scaled* version of the
            original image; these proposals must be scaled by 1 / scale (where
            scale comes from im_info; see above) to transform it back to the
            original input image coordinate system.
          - 'rpn_roi_probs': 1D tensor of objectness probability scores
            (extracted from rpn_cls_probs; see above).
        """
        # 1. for each location i in a (H, W) grid:
        #      generate A anchor boxes centered on cell i
        #      apply predicted bbox deltas to each of the A anchors at cell i
        # 2. clip predicted boxes to image
        # 3. remove predicted boxes with either height or width < threshold
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take the top pre_nms_topN proposals before NMS
        # 6. apply NMS with a loose threshold (0.7) to the remaining proposals
        # 7. take after_nms_topN proposals after NMS
        # 8. return the top proposals
        
        """Type conversion"""
        # predicted probability of fg object for each RPN anchor
        scores = rpn_cls_prob.data.cpu().numpy()
        # predicted achors transformations
        bbox_deltas = rpn_bbox_pred.data.cpu().numpy()
        # input image (height, width, scale), in which scale is the scale factor
        # applied to the original dataset image to get the network input image
        im_info = im_info.data.cpu().numpy()

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]
        # Enumerate all shifted positions on the (H, W) grid
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y, copy=False)
        # Convert to (K, 4), K=H*W, where the columns are (dx, dy, dx, dy)
        # shift pointing to each grid location
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                            shift_y.ravel())).transpose()

        # Broacast anchors over shifts to enumerate all anchors at all positions
        # in the (H, W) grid:
        #   - add A anchors of shape (1, A, 4) to
        #   - K shifts of shape (K, 1, 4) to get
        #   - all shifted anchors of shape (K, A, 4)
        #   - reshape to (K*A, 4) shifted anchors
        num_images = scores.shape[0]
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = self._anchors[np.newaxis, :, :] + shifts[:, np.newaxis, :]
        all_anchors = all_anchors.reshape((K * A, 4))
        # all_anchors = torch.from_numpy(all_anchors).type_as(scores)

        rois = np.empty((0, 5), dtype=np.float32)
        roi_probs = np.empty((0, 1), dtype=np.float32)
        for im_i in range(num_images):
            im_i_boxes, im_i_probs = self.proposals_for_one_image(
                im_info[im_i, :], all_anchors, bbox_deltas[im_i, :, :, :],
                scores[im_i, :, :, :])
            batch_inds = im_i * np.ones(
                (im_i_boxes.shape[0], 1), dtype=np.float32)
            im_i_rois = np.hstack((batch_inds, im_i_boxes))
            rois = np.append(rois, im_i_rois, axis=0)
            roi_probs = np.append(roi_probs, im_i_probs, axis=0)

        return rois, roi_probs  # Note: ndarrays

    def proposals_for_one_image(self, im_info, all_anchors, bbox_deltas, scores):
        # Get mode-dependent configuration
        CONFIG_key = 'TRAIN' if self.training else 'TEST'
        pre_nms_topN = CONFIG[CONFIG_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = CONFIG[CONFIG_key].RPN_POST_NMS_TOP_N
        nms_thresh = CONFIG[CONFIG_key].RPN_NMS_THRESH
        min_size = CONFIG[CONFIG_key].RPN_MIN_SIZE
        # print('generate_proposals:', pre_nms_topN, post_nms_topN, nms_thresh, min_size)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #   - bbox deltas will be (4 * A, H, W) format from conv output
        #   - transpose to (H, W, 4 * A)
        #   - reshape to (H * W * A, 4) where rows are ordered by (H, W, A)
        #     in slowest to fastest order to match the enumerated anchors
        bbox_deltas = bbox_deltas.transpose((1, 2, 0)).reshape((-1, 4))

        # Same story for the scores:
        #   - scores are (A, H, W) format from conv output
        #   - transpose to (H, W, A)
        #   - reshape to (H * W * A, 1) where rows are ordered by (H, W, A)
        #     to match the order of anchors and bbox_deltas
        scores = scores.transpose((1, 2, 0)).reshape((-1, 1))
        # print('pre_nms:', bbox_deltas.shape, scores.shape)

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        if pre_nms_topN <= 0 or pre_nms_topN >= len(scores):
            order = np.argsort(-scores.squeeze())
        else:
            # Avoid sorting possibly large arrays; First partition to get top K
            # unsorted and then sort just those (~20x faster for 200k scores)
            inds = np.argpartition(-scores.squeeze(),
                                   pre_nms_topN)[:pre_nms_topN]
            order = np.argsort(-scores[inds].squeeze())
            order = inds[order]
        bbox_deltas = bbox_deltas[order, :]
        all_anchors = all_anchors[order, :]
        scores = scores[order]

        # Transform anchors into proposals via bbox transformations
        proposals = bbox_transform(all_anchors, bbox_deltas, (1.0, 1.0, 1.0, 1.0))

        # 2. clip proposals to image (may result in proposals with zero area
        # that will be removed in the next step)
        proposals = self.clip_tiled_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < min_size
        keep = self._filter_boxes(proposals, min_size, im_info)
        proposals = proposals[keep, :]
        scores = scores[keep]
        # print('pre_nms:', proposals.shape, scores.shape)

        # 6. apply loose nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        if nms_thresh > 0:
            keep = nms_gpu(np.hstack((proposals, scores)), nms_thresh)
            # print('nms keep:', keep.shape)
            if post_nms_topN > 0:
                keep = keep[:post_nms_topN]
            proposals = proposals[keep, :]
            scores = scores[keep]
        # print('final proposals:', proposals.shape, scores.shape)
        return proposals, scores


    def _filter_boxes(self, boxes, min_size, im_info):
        """Only keep boxes with both sides >= min_size and center within the image.
      """
        # Scale min_size to match image scale
        min_size *= im_info[2]
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        x_ctr = boxes[:, 0] + ws / 2.
        y_ctr = boxes[:, 1] + hs / 2.
        keep = np.where((ws >= min_size) & (hs >= min_size) &
                        (x_ctr < im_info[1]) & (y_ctr < im_info[0]))[0]
        return keep
    
    def _clip_tiled_boxes(self, boxes, im_shape):
        """Clip boxes to image boundaries. im_shape is [height, width] and boxes
        has shape (N, 4 * num_tiled_boxes)."""
        assert boxes.shape[1] % 4 == 0, \
            'boxes.shape[1] is {:d}, but must be divisible by 4.'.format(
            boxes.shape[1]
        )
        # x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes
    
class RPN(nn.Module)
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

        self.RPN_GenerateProposals = GenerateProposalsOp(anchors, spatial_scale)
        self.RPN_GenerateProposalLabels = GenerateProposalLabelsOp()
        
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
        
        # resnet50C4: stride = 16, outplane = 1024
        self.backbone = resnet50C4(pretrained=True, num_classes=None)
        self.rpn = RPN(dim_in = 1024, spatial_scale = 1.0 / CONFIG.RPN.STRIDE)
        
        self.proposal_layer
        self.roialign_layer
        self.head_bbox
        self.head_mask
        
        
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
    dataset = CocoDataset(CONFIG.DATASET.TRAIN_DIR, CONFIG.DATASET.TRAIN_ANNOFILE)
    batch_size = len(CONFIG.SOLVER.GPU_IDS) * CONFIG.SOLVER.IMS_PER_BATCH
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        # sampler=sampler,
        num_workers=CONFIG.SOLVER.WORKERS,
        collate_fn=dataset.collate_minibatch)
    for input_data in tqdm(dataloader):
        for key in input_data:
            if key != 'roidb': # roidb is a list of ndarrays with inconsistent length
                input_data[key] = list(map(Variable, input_data[key]))

        net_outputs = maskRCNN(**input_data)
        # loss = net_outputs['total_loss']
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        pass