from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.utils.data.dataloader import default_collate
import torch.utils.data.sampler as torch_sampler

import sys
import os 
import time
import numpy as np
import cv2
import random
import math
import copy
import logging
import scipy.sparse
from tqdm import tqdm
from collections import namedtuple

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask
import pycocotools.mask as mask_util

import box_utils
import segm_utils
import keypoint_utils
import blob_utils

from generate_anchors import generate_anchors
from config import CONFIG

# Set logger
logger = logging.getLogger(__name__)
format = logging.Formatter("%(asctime)s - %(message)s")    # output format 
sh = logging.StreamHandler(stream=sys.stdout)    # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)
logger.setLevel(logging.DEBUG)


####################################################################################################
#                                 Anchors (data_utils.py)
####################################################################################################

# octave and aspect fields are only used on RetinaNet. Octave corresponds to the
# scale of the anchor and aspect denotes which aspect ratio is used in the range
# of aspect ratios
FieldOfAnchors = namedtuple(
    'FieldOfAnchors', [
        'field_of_anchors', 'num_cell_anchors', 'stride', 'field_size',
        'octave', 'aspect'
    ]
)

def get_field_of_anchors(
    stride, anchor_sizes, anchor_aspect_ratios, octave=None, aspect=None
):
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
    field_of_anchors = field_of_anchors.reshape((K * A, 4))
    foa = FieldOfAnchors(
        field_of_anchors=field_of_anchors.astype(np.float32),
        num_cell_anchors=num_cell_anchors,
        stride=stride,
        field_size=field_size,
        octave=octave,
        aspect=aspect
    )
    return foa


def unmap(data, count, inds, fill=0):
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


def compute_targets(ex_rois, gt_rois, weights=(1.0, 1.0, 1.0, 1.0)):
    """Compute bounding-box regression targets for an image."""
    return box_utils.bbox_transform_inv(ex_rois, gt_rois, weights).astype(
        np.float32, copy=False
    )



####################################################################################################
#                                         COCO Processing Tools
####################################################################################################

def add_proposals(roidb, rois, scales, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_proposal_boxes_into_roidb(roidb, box_list)
    if crowd_thresh > 0:
        _filter_crowd_proposals(roidb, crowd_thresh)
    _add_class_assignments(roidb)


def _merge_proposal_boxes_into_roidb(roidb, box_list):
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
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
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


def _filter_crowd_proposals(roidb, crowd_thresh):
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
        crowd_boxes = box_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = box_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        iscrowd_flags = [int(True)] * len(crowd_inds)
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        gt_overlaps[non_gt_inds[bad_inds], :] = -1
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)


def _add_class_assignments(roidb):
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


def _sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]


####################################################################################################
#                                         COCO Dataset
####################################################################################################

def _compute_and_log_stats(roidb):
    classes = roidb[0]['dataset'].classes
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

def rank_for_training(roidb):
    """Rank the roidb entries according to image aspect ration and mark for cropping
    for efficient batching if image is too long.

    Returns:
        ratio_list: ndarray, list of aspect ratios from small to large
        ratio_index: ndarray, list of roidb entry indices correspond to the ratios
    """
    RATIO_HI = CONFIG.TRAIN.ASPECT_HI  # largest ratio to preserve.
    RATIO_LO = CONFIG.TRAIN.ASPECT_LO  # smallest ratio to preserve.

    need_crop_cnt = 0

    ratio_list = []
    for entry in roidb:
        width = entry['width']
        height = entry['height']
        ratio = width / float(height)

        if CONFIG.TRAIN.ASPECT_CROPPING:
            if ratio > RATIO_HI:
                entry['need_crop'] = True
                ratio = RATIO_HI
                need_crop_cnt += 1
            elif ratio < RATIO_LO:
                entry['need_crop'] = True
                ratio = RATIO_LO
                need_crop_cnt += 1
            else:
                entry['need_crop'] = False
        else:
            entry['need_crop'] = False

        ratio_list.append(ratio)

    if CONFIG.TRAIN.ASPECT_CROPPING:
        logging.info('Number of entries that need to be cropped: %d. Ratio bound: [%.2f, %.2f]',
                     need_crop_cnt, RATIO_LO, RATIO_HI)
    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index

class CocoDatasetInfo():
    def __init__(self, ImageRoot, AnnoFile, 
                 gt=False, proposal_file=None, min_proposal_size=2, proposal_limit=-1, crowd_filter_thresh=0):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        self.imgroot = ImageRoot
        self.COCO = COCO(AnnoFile)
        self.gt = gt
        self.proposal_file = proposal_file
        self.min_proposal_size = min_proposal_size
        self.proposal_limit = proposal_limit
        self.crowd_filter_thresh = crowd_filter_thresh
        
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
        
        self.image_ids = self.COCO.getImgIds()
        self.image_ids.sort()
        if CONFIG.MYDEBUG:
            self.image_ids = self.image_ids[0:128]
        
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'

        # _init_keypoints()
        """Initialize COCO keypoint information."""
        self.keypoints = None
        self.keypoint_flip_map = None
        self.keypoints_to_id_map = None
        self.num_keypoints = 0
        # Thus far only the 'person' category has keypoints
        if 'person' in self.category_to_id_map:
            cat_info = self.COCO.loadCats([self.category_to_id_map['person']])
            
            # Check if the annotations contain keypoint data or not
            if 'keypoints' in cat_info[0]:
                keypoints = cat_info[0]['keypoints']
                self.keypoints_to_id_map = dict(
                    zip(keypoints, range(len(keypoints))))
                self.keypoints = keypoints
                self.num_keypoints = len(keypoints)
                if CONFIG.KRCNN.NUM_KEYPOINTS != -1:
                    assert CONFIG.KRCNN.NUM_KEYPOINTS == self.num_keypoints, \
                        "number of keypoints should equal when using multiple datasets"
                else:
                    CONFIG.KRCNN.NUM_KEYPOINTS = self.num_keypoints
                self.keypoint_flip_map = {
                    'left_eye': 'right_eye',
                    'left_ear': 'right_ear',
                    'left_shoulder': 'right_shoulder',
                    'left_elbow': 'right_elbow',
                    'left_wrist': 'right_wrist',
                    'left_hip': 'right_hip',
                    'left_knee': 'right_knee',
                    'left_ankle': 'right_ankle'}
        
        # Check. total should be 849901
        # roidb = [self.__getitem__(idx) for idx in tqdm(range(self.__len__()))]
        # roidb = [item for item in roidb if item is not None]
        # _compute_and_log_stats(roidb)
        
        self.roidb = None # Pre-load
        if CONFIG.TRAIN.ASPECT_GROUPING or CONFIG.TRAIN.ASPECT_CROPPING:
            logger.info('Computing image aspect ratios and ordering the ratios...')
            self.roidb = [self.getitem(idx) for idx in tqdm(range(self.__len__()))]
            self.ratio_list, self.ratio_index = rank_for_training(self.roidb)
            logger.info('done')
        else:
            self.ratio_list, self.ratio_index = None, None
        
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        if self.roidb is None:
            return self.getitem(idx)
        else:
            return self.roidb[idx]
        
    def getitem(self, idx):
        '''
        combined_roidb_for_training(dataset_names, proposal_files)
        '''
        # ---------------------------
        # _prep_roidb_entry()    
        # ---------------------------
        image_id = self.image_ids[idx]
        datainfo = self.COCO.loadImgs(image_id)[0]
        rawdata = {
            'dataset': self,
            'id': image_id,
            'filename': datainfo['file_name'],
            'image': os.path.join(self.imgroot, datainfo['file_name']),
            'width': datainfo['width'],
            'height': datainfo['height'],
            
            'flipped': False,
            'has_visible_keypoints': False,
            'boxes': np.empty((0, 4), dtype=np.float32),
            'segms': [],
            'gt_classes': np.empty((0), dtype=np.int32),
            'seg_areas': np.empty((0), dtype=np.float32),
            'gt_overlaps': scipy.sparse.csr_matrix(
                            np.empty((0, self.num_classes), dtype=np.float32)
                            ),
            'is_crowd': np.empty((0), dtype=np.bool),
            # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
            # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
            'box_to_gt_ind_map': np.empty((0), dtype=np.int32)
        }
        if self.keypoints is not None:
            rawdata['gt_keypoints'] = np.empty((0, 3, self.num_keypoints), dtype=np.int32)
            
        # ---------------------------
        # _add_gt_annotations()
        # ---------------------------
        if self.gt:
            # Include ground-truth object annotations
            """Add ground truth annotation metadata to an roidb entry."""
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
                x1, y1, x2, y2 = box_utils.xywh_to_xyxy(obj['bbox'])
                x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
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
            if self.keypoints is not None:
                gt_keypoints = np.zeros(
                    (num_valid_objs, 3, self.num_keypoints),
                    dtype=rawdata['gt_keypoints'].dtype
                )

            im_has_visible_keypoints = False
            for ix, obj in enumerate(valid_objs):
                cls = self.json_category_id_to_contiguous_id[obj['category_id']]
                boxes[ix, :] = obj['clean_bbox']
                gt_classes[ix] = cls
                seg_areas[ix] = obj['area']
                is_crowd[ix] = obj['iscrowd']
                box_to_gt_ind_map[ix] = ix
                if self.keypoints is not None:
                    gt_keypoints[ix, :, :] = self._get_gt_keypoints(obj)
                    if np.sum(gt_keypoints[ix, 2, :]) > 0:
                        im_has_visible_keypoints = True
                if obj['iscrowd']:
                    # Set overlap to -1 for all classes for crowd objects
                    # so they will be excluded during training
                    gt_overlaps[ix, :] = -1.0
                else:
                    gt_overlaps[ix, cls] = 1.0
            rawdata['boxes'] = np.append(rawdata['boxes'], boxes, axis=0)
            rawdata['segms'].extend(valid_segms)
            # To match the original implementation:
            # rawdata['boxes'] = np.append(
            #     rawdata['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
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
            if self.keypoints is not None:
                rawdata['gt_keypoints'] = np.append(
                    rawdata['gt_keypoints'], gt_keypoints, axis=0
                )
                rawdata['has_visible_keypoints'] = im_has_visible_keypoints
        
        # ---------------------------
        # _add_proposals_from_file(roidb, proposal_file, min_proposal_size, proposal_limit,
        #                           crowd_filter_thresh):
        #   _sort_proposals()
        #   _merge_proposal_boxes_into_roidb()
        #   _filter_crowd_proposals()
        # ---------------------------
        if self.proposal_file is not None:
            """Add proposals from a proposals file to an roidb."""
            logger.info('Loading proposals from: {}'.format(self.proposal_file))
            with open(self.proposal_file, 'r') as f:
                proposals = pickle.load(f)
            id_field = 'indexes' if 'indexes' in proposals else 'ids'  # compat fix
            _sort_proposals(proposals, id_field)
            box_list = []

            boxes = proposals['boxes'][i]
            # Sanity check that these boxes are for the correct image id
            assert rawdata['id'] == proposals[id_field][i]
            # Remove duplicate boxes and very small boxes and then take top k
            boxes = box_utils.clip_boxes_to_image(
                boxes, rawdata['height'], rawdata['width']
            )
            keep = box_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = box_utils.filter_small_boxes(boxes, self.min_proposal_size)
            boxes = boxes[keep, :]
            if self.proposal_limit > 0:
                boxes = boxes[:self.proposal_limit, :]
            box_list.append(boxes)
            
            _merge_proposal_boxes_into_roidb([rawdata], box_list)
            if self.crowd_filter_thresh > 0:
                _filter_crowd_proposals([rawdata], self.crowd_filter_thresh)

        # ---------------------------
        # _add_class_assignments():
        # ---------------------------
        _add_class_assignments([rawdata])
        
        
        # ---------------------------
        # extend_with_flipped_entries(roidb, ds)
        # ---------------------------
        if CONFIG.TRAIN.USE_FLIPPED and random.random()<0.5:
            width = rawdata['width']
            boxes = rawdata['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = width - oldx2 - 1
            boxes[:, 2] = width - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()

            rawdata['boxes'] = boxes
            rawdata['segms'] = segm_utils.flip_segms(
                rawdata['segms'], rawdata['height'], rawdata['width']
            )
            if self.keypoints is not None:
                rawdata['gt_keypoints'] = keypoint_utils.flip_keypoints(
                    self.keypoints, self.keypoint_flip_map,
                    rawdata['gt_keypoints'], rawdata['width']
                )
            rawdata['flipped'] = True
            
        # ---------------------------
        # filter_for_training(roidb)
        # ---------------------------
        """Remove roidb entries that have no usable RoIs based on config settings.
        """
        def is_valid(rawdata):
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
            if CONFIG.MODEL.KEYPOINTS_ON:
                # If we're training for keypoints, exclude images with no keypoints
                valid = valid and rawdata['has_visible_keypoints']
            return valid

        if not is_valid(rawdata):
            # return None
            return self.__getitem__(idx+1) # DIFF: may get same data twice.
        
        # ---------------------------
        # rank_for_training(roidb)
        # ---------------------------
        # implemented in self.__init__()
        
        # ---------------------------
        # add_bbox_regression_targets(roidb):
        #   _compute_targets(entry)
        # ---------------------------
        """Compute bounding-box regression targets for an image."""
        # Indices of ground-truth ROIs
        rois = rawdata['boxes']
        overlaps = rawdata['max_overlaps']
        labels = rawdata['max_classes']
        gt_inds = np.where((rawdata['gt_classes'] > 0) & (rawdata['is_crowd'] == 0))[0]
        # Targets has format (class, tx, ty, tw, th)
        targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
        if len(gt_inds) == 0:
            # Bail if the image has no ground-truth ROIs
            return targets

        # Indices of examples for which we try to make predictions
        ex_inds = np.where(overlaps >= CONFIG.TRAIN.BBOX_THRESH)[0]

        # Get IoU overlap between each ex ROI and gt ROI
        ex_gt_overlaps = box_utils.bbox_overlaps(
            rois[ex_inds, :].astype(dtype=np.float32, copy=False),
            rois[gt_inds, :].astype(dtype=np.float32, copy=False))

        # Find which gt ROI each ex ROI has max overlap with:
        # this will be the ex ROI's gt target
        gt_assignment = ex_gt_overlaps.argmax(axis=1)
        gt_rois = rois[gt_inds[gt_assignment], :]
        ex_rois = rois[ex_inds, :]
        # Use class "1" for all boxes if using class_agnostic_bbox_reg
        targets[ex_inds, 0] = (
            1 if CONFIG.MODEL.CLS_AGNOSTIC_BBOX_REG else labels[ex_inds])
        targets[ex_inds, 1:] = box_utils.bbox_transform_inv(
            ex_rois, gt_rois, CONFIG.MODEL.BBOX_REG_WEIGHTS)
        
        rawdata['bbox_targets'] = targets
        
        return rawdata
    
####################################################################################################
#                                  COCO Dataset MiniBatch Loader
####################################################################################################
def get_fast_rcnn_blob_names(is_training=True):
    # Fast R-CNN like models trained on precomputed proposals
    raise NotImplementedError

def get_rpn_blob_names(is_training=True):
    """Blob names used by RPN."""
    # im_info: (height, width, image scale)
    blob_names = ['im_info']
    if is_training:
        # gt boxes: (batch_idx, x1, y1, x2, y2, cls)
        blob_names += ['roidb']
        if CONFIG.FPN.FPN_ON and CONFIG.FPN.MULTILEVEL_RPN:
            # Same format as RPN blobs, but one per FPN level
            for lvl in range(CONFIG.FPN.RPN_MIN_LEVEL, CONFIG.FPN.RPN_MAX_LEVEL + 1):
                blob_names += [
                    'rpn_labels_int32_wide_fpn' + str(lvl),
                    'rpn_bbox_targets_wide_fpn' + str(lvl),
                    'rpn_bbox_inside_weights_wide_fpn' + str(lvl),
                    'rpn_bbox_outside_weights_wide_fpn' + str(lvl)
                ]
        else:
            # Single level RPN blobs
            blob_names += [
                'rpn_labels_int32_wide',
                'rpn_bbox_targets_wide',
                'rpn_bbox_inside_weights_wide',
                'rpn_bbox_outside_weights_wide'
            ]
    return blob_names

def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data']
    if CONFIG.RPN.RPN_ON:
        # RPN-only or end-to-end Faster R-CNN
        blob_names += get_rpn_blob_names(is_training=is_training)
    elif CONFIG.RETINANET.RETINANET_ON:
        raise NotImplementedError
    else:
        # Fast R-CNN like models trained on precomputed proposals
        blob_names += get_fast_rcnn_blob_names(
            is_training=is_training
        )
    return blob_names

def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(CONFIG.TRAIN.SCALES), size=num_images)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])
        # If NOT using opencv to read in images, uncomment following lines
        # if len(im.shape) == 2:
        #     im = im[:, :, np.newaxis]
        #     im = np.concatenate((im, im, im), axis=2)
        # # flip the channel, since the original one using cv2
        # # rgb -> bgr
        # im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = CONFIG.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, CONFIG.PIXEL_MEANS, [target_size], CONFIG.TRAIN.MAX_SIZE)
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales

def add_fast_rcnn_blobs(blobs, im_scales, roidb):
    # Fast R-CNN like models trained on precomputed proposals
    raise NotImplementedError
    
def add_rpn_blobs(blobs, im_scales, roidb):
    """Add blobs needed training RPN-only and end-to-end Faster R-CNN models."""
    if CONFIG.FPN.FPN_ON and CONFIG.FPN.MULTILEVEL_RPN:
        # RPN applied to many feature levels, as in the FPN paper
        k_max = CONFIG.FPN.RPN_MAX_LEVEL
        k_min = CONFIG.FPN.RPN_MIN_LEVEL
        foas = []
        for lvl in range(k_min, k_max + 1):
            field_stride = 2.**lvl
            anchor_sizes = (CONFIG.FPN.RPN_ANCHOR_START_SIZE * 2.**(lvl - k_min), )
            anchor_aspect_ratios = CONFIG.FPN.RPN_ASPECT_RATIOS
            foa = get_field_of_anchors(
                field_stride, anchor_sizes, anchor_aspect_ratios
            )
            foas.append(foa)
        all_anchors = np.concatenate([f.field_of_anchors for f in foas])
    else:
        foa = get_field_of_anchors(CONFIG.RPN.STRIDE, CONFIG.RPN.SIZES,
                                              CONFIG.RPN.ASPECT_RATIOS)
        all_anchors = foa.field_of_anchors

    for im_i, rawdata in enumerate(roidb):
        scale = im_scales[im_i]
        im_height = np.round(rawdata['height'] * scale)
        im_width = np.round(rawdata['width'] * scale)
        gt_inds = np.where(
            (rawdata['gt_classes'] > 0) & (rawdata['is_crowd'] == 0)
        )[0]
        gt_rois = rawdata['boxes'][gt_inds, :] * scale
        # TODO(rbg): gt_boxes is poorly named;
        # should be something like 'gt_rois_info'
        gt_boxes = blob_utils.zeros((len(gt_inds), 6))
        gt_boxes[:, 0] = im_i  # batch inds
        gt_boxes[:, 1:5] = gt_rois
        gt_boxes[:, 5] = rawdata['gt_classes'][gt_inds]
        im_info = np.array([[im_height, im_width, scale]], dtype=np.float32)
        blobs['im_info'].append(im_info)

        # Add RPN targets
        if CONFIG.FPN.FPN_ON and CONFIG.FPN.MULTILEVEL_RPN:
            # RPN applied to many feature levels, as in the FPN paper
            rpn_blobs = _get_rpn_blobs(
                im_height, im_width, foas, all_anchors, gt_rois
            )
            for i, lvl in enumerate(range(k_min, k_max + 1)):
                for k, v in rpn_blobs[i].items():
                    blobs[k + '_fpn' + str(lvl)].append(v)
        else:
            # Classical RPN, applied to a single feature level
            rpn_blobs = _get_rpn_blobs(
                im_height, im_width, [foa], all_anchors, gt_rois
            )
            for k, v in rpn_blobs.items():
                blobs[k].append(v)

    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)

    valid_keys = [
        'has_visible_keypoints', 'boxes', 'segms', 'seg_areas', 'gt_classes',
        'gt_overlaps', 'is_crowd', 'box_to_gt_ind_map', 'gt_keypoints'
    ]
    minimal_roidb = [{} for _ in range(len(roidb))]
    for i, e in enumerate(roidb):
        for k in valid_keys:
            if k in e:
                minimal_roidb[i][k] = e[k]
    # blobs['roidb'] = blob_utils.serialize(minimal_roidb)
    blobs['roidb'] = minimal_roidb

    # Always return valid=True, since RPN minibatches are valid by design
    return True

def _get_rpn_blobs(im_height, im_width, foas, all_anchors, gt_boxes):
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
        anchor_by_gt_overlap = box_utils.bbox_overlaps(anchors, gt_boxes)
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
    bbox_targets[fg_inds, :] = compute_targets(
        anchors[fg_inds, :], gt_boxes[anchor_to_gt_argmax[fg_inds], :]
    )

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
    labels = unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = unmap(
        bbox_targets, total_anchors, inds_inside, fill=0
    )
    bbox_inside_weights = unmap(
        bbox_inside_weights, total_anchors, inds_inside, fill=0
    )
    bbox_outside_weights = unmap(
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

def get_minibatch(roidb):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    im_blob, im_scales = _get_image_blob(roidb)
    blobs['data'] = im_blob
    if CONFIG.RPN.RPN_ON:
        # RPN-only or end-to-end Faster/Mask R-CNN
        valid = add_rpn_blobs(blobs, im_scales, roidb)
    elif CONFIG.RETINANET.RETINANET_ON:
        raise NotImplementedError
    else:
        # Fast R-CNN like models trained on precomputed proposals
        valid = add_fast_rcnn_blobs(blobs, im_scales, roidb)
    return blobs, valid


class CocoDatasetMiniBatch():
    # ---------------------------
    # class RoiDataLoader(data.Dataset)
    # ---------------------------
    def __init__(self, ImageRoot, AnnoFile, 
                 gt=False, proposal_file=None, min_proposal_size=2, proposal_limit=-1, crowd_filter_thresh=0):
        self._roidb = CocoDatasetInfo(ImageRoot, AnnoFile, 
                                gt, proposal_file, min_proposal_size, proposal_limit, crowd_filter_thresh)
        
        self._num_classes = self._roidb.num_classes
        self.DATA_SIZE = len(self._roidb)
        self.ratio_list = self._roidb.ratio_list
        self.ratio_index = self._roidb.ratio_index

    def __getitem__(self, index_tuple):
        index, ratio = index_tuple
        single_db = [self._roidb[index]]
        blobs, valid = get_minibatch(single_db)
        # TODO: Check if minibatch is valid ? If not, abandon it.
        # Need to change _worker_loop in torch.utils.data.dataloader.py.

        # Squeeze batch dim
        for key in blobs:
            if key != 'roidb':
                blobs[key] = blobs[key].squeeze(axis=0)

        #if self._roidb[index]['need_crop']: DIFF
        if 'need_crop' in single_db[0].keys():
            if not single_db[0]['need_crop']:
                return blobs
            self.crop_data(blobs, ratio)
            # Check bounding box
            rawdata = blobs['roidb'][0]
            boxes = rawdata['boxes']
            invalid = (boxes[:, 0] == boxes[:, 2]) | (boxes[:, 1] == boxes[:, 3])
            valid_inds = np.nonzero(~ invalid)[0]
            if len(valid_inds) < len(boxes):
                for key in ['boxes', 'gt_classes', 'seg_areas', 'gt_overlaps', 'is_crowd',
                            'box_to_gt_ind_map', 'gt_keypoints']:
                    if key in rawdata:
                        rawdata[key] = rawdata[key][valid_inds]
                rawdata['segms'] = [rawdata['segms'][ind] for ind in valid_inds]

        # blobs['roidb'] = blob_utils.serialize(blobs['roidb'])  # CHECK: maybe we can serialize in collate_fn ??

        return blobs
    
    def __len__(self):
        return self.DATA_SIZE
    
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
        for i in range(0, len(list_of_blobs), CONFIG.TRAIN.IMS_PER_BATCH):
            mini_list = list_of_blobs[i:(i + CONFIG.TRAIN.IMS_PER_BATCH)]
            # Pad image data
            mini_list = pad_image_data(mini_list)
            minibatch = default_collate(mini_list)
            minibatch['roidb'] = list_of_roidb[i:(i + CONFIG.TRAIN.IMS_PER_BATCH)]
            for key in minibatch:
                Batch[key].append(minibatch[key])

        return Batch

    def crop_data(self, blobs, ratio):
        data_height, data_width = map(int, blobs['im_info'][:2])
        boxes = blobs['roidb'][0]['boxes']
        if ratio < 1:  # width << height, crop height
            size_crop = math.ceil(data_width / ratio)  # size after crop
            min_y = math.floor(np.min(boxes[:, 1]))
            max_y = math.floor(np.max(boxes[:, 3]))
            box_region = max_y - min_y + 1
            if min_y == 0:
                y_s = 0
            else:
                if (box_region - size_crop) < 0:
                    y_s_min = max(max_y - size_crop, 0)
                    y_s_max = min(min_y, data_height - size_crop)
                    y_s = y_s_min if y_s_min == y_s_max else \
                        np.random.choice(range(y_s_min, y_s_max + 1))
                else:
                    # CHECK: rethinking the mechnism for the case box_region > size_crop
                    # Now, the crop is biased on the lower part of box_region caused by
                    # // 2 for y_s_add
                    y_s_add = (box_region - size_crop) // 2
                    y_s = min_y if y_s_add == 0 else \
                        np.random.choice(range(min_y, min_y + y_s_add + 1))
            # Crop the image
            blobs['data'] = blobs['data'][:, y_s:(y_s + size_crop), :,]
            # Update im_info
            blobs['im_info'][0] = size_crop
            # Shift and clamp boxes ground truth
            boxes[:, 1] -= y_s
            boxes[:, 3] -= y_s
            np.clip(boxes[:, 1], 0, size_crop - 1, out=boxes[:, 1])
            np.clip(boxes[:, 3], 0, size_crop - 1, out=boxes[:, 3])
            blobs['roidb'][0]['boxes'] = boxes
        else:  # width >> height, crop width
            size_crop = math.ceil(data_height * ratio)
            min_x = math.floor(np.min(boxes[:, 0]))
            max_x = math.floor(np.max(boxes[:, 2]))
            box_region = max_x - min_x + 1
            if min_x == 0:
                x_s = 0
            else:
                if (box_region - size_crop) < 0:
                    x_s_min = max(max_x - size_crop, 0)
                    x_s_max = min(min_x, data_width - size_crop)
                    x_s = x_s_min if x_s_min == x_s_max else \
                        np.random.choice(range(x_s_min, x_s_max + 1))
                else:
                    x_s_add = (box_region - size_crop) // 2
                    x_s = min_x if x_s_add == 0 else \
                        np.random.choice(range(min_x, min_x + x_s_add + 1))
            # Crop the image
            blobs['data'] = blobs['data'][:, :, x_s:(x_s + size_crop)]
            # Update im_info
            blobs['im_info'][1] = size_crop
            # Shift and clamp boxes ground truth
            boxes[:, 0] -= x_s
            boxes[:, 2] -= x_s
            np.clip(boxes[:, 0], 0, size_crop - 1, out=boxes[:, 0])
            np.clip(boxes[:, 2], 0, size_crop - 1, out=boxes[:, 2])
            blobs['roidb'][0]['boxes'] = boxes
            
####################################################################################################
#                               COCO Dataset MiniBatch Sampler
####################################################################################################

def cal_minibatch_ratio(ratio_list):
    """Given the ratio_list, we want to make the RATIO same for each minibatch on each GPU.
    Note: this only work for 1) CONFIG.TRAIN.MAX_SIZE is ignored during `prep_im_for_blob` 
    and 2) CONFIG.TRAIN.SCALES containing SINGLE scale.
    Since all prepared images will have same min side length of CONFIG.TRAIN.SCALES[0], we can
     pad and batch images base on that.
    """
    DATA_SIZE = len(ratio_list)
    ratio_list_minibatch = np.empty((DATA_SIZE,))
    num_minibatch = int(np.ceil(DATA_SIZE / CONFIG.TRAIN.IMS_PER_BATCH))  # Include leftovers
    for i in range(num_minibatch):
        left_idx = i * CONFIG.TRAIN.IMS_PER_BATCH
        right_idx = min((i+1) * CONFIG.TRAIN.IMS_PER_BATCH - 1, DATA_SIZE - 1)

        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1

        ratio_list_minibatch[left_idx:(right_idx+1)] = target_ratio
    return ratio_list_minibatch


class MinibatchSampler(torch_sampler.Sampler):
    def __init__(self, ratio_list, ratio_index):
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.num_data = len(ratio_list)

        if CONFIG.TRAIN.ASPECT_GROUPING:
            # Given the ratio_list, we want to make the ratio same
            # for each minibatch on each GPU.
            self.ratio_list_minibatch = cal_minibatch_ratio(ratio_list)

    def __iter__(self):
        if CONFIG.TRAIN.ASPECT_GROUPING:
            # indices for aspect grouping awared permutation
            n, rem = divmod(self.num_data, CONFIG.TRAIN.IMS_PER_BATCH)
            round_num_data = n * CONFIG.TRAIN.IMS_PER_BATCH
            indices = np.arange(round_num_data)
            np.random.shuffle(indices.reshape(-1, CONFIG.TRAIN.IMS_PER_BATCH))  # inplace shuffle
            if rem != 0:
                indices = np.append(indices, np.arange(round_num_data, round_num_data + rem))
            ratio_index = self.ratio_index[indices]
            ratio_list_minibatch = self.ratio_list_minibatch[indices]
        else:
            rand_perm = np.random.permutation(self.num_data)
            ratio_list = self.ratio_list[rand_perm]
            ratio_index = self.ratio_index[rand_perm]
            # re-calculate minibatch ratio list
            ratio_list_minibatch = cal_minibatch_ratio(ratio_list)

        return iter(zip(ratio_index.tolist(), ratio_list_minibatch.tolist()))

    def __len__(self):
        return self.num_data
    

## main ##
if __name__ == '__main__':
    import torch.utils.data
    
    # dataset = CocoDatasetInfo(CONFIG.MYDATASET.TRAIN_DIR, CONFIG.MYDATASET.TRAIN_ANNOFILE,
    #                       gt=True, crowd_filter_thresh=CONFIG.TRAIN.CROWD_FILTER_THRESH)
    dataset = CocoDatasetMiniBatch(CONFIG.MYDATASET.TRAIN_DIR, CONFIG.MYDATASET.TRAIN_ANNOFILE,
                          gt=True, crowd_filter_thresh=CONFIG.TRAIN.CROWD_FILTER_THRESH)
    sampler = MinibatchSampler(dataset.ratio_list, dataset.ratio_index)
    
    # dataset[(0, 1.0)]
    
    batch_size = len(CONFIG.MYSOLVER.GPU_IDS) * CONFIG.TRAIN.IMS_PER_BATCH
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=CONFIG.MYSOLVER.WORKERS,
        collate_fn=dataset.collate_minibatch)
    
    for input_data in tqdm(dataloader):
        pass
    