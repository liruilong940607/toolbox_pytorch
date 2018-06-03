from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import os
import yaml
from easydict import EasyDict as edict
CONFIG = edict()

# ---------------------------------------------------------------------------- #
# mask heads or keypoint heads that share res5 stage weights and
# training forward computation with box head.
# ---------------------------------------------------------------------------- #
_SHARE_RES5_HEADS = set(
    [
        'mask_rcnn_heads.mask_rcnn_fcn_head_v0upshare',
    ]
)

def config_load(yaml_file):
    _CONFIG = edict(yaml.load(open(yaml_file, 'r')))
    _merge_a_into_b(_CONFIG, CONFIG)

def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, dict), 'Argument `a` must be an dict'
    assert isinstance(b, dict), 'Argument `b` must be an dict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        v = copy.deepcopy(v_)
        # Add new key to dict b
        if k not in b:
            b[k] = v
            continue
        # Recursively merge dicts
        if isinstance(v, dict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v
            

def assert_and_infer_cfg(cfg):
    from packaging import version
    import torch
    from torch.nn import init
    
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """
    if cfg.MODEL.RPN_ONLY or cfg.MODEL.FASTER_RCNN:
        cfg.RPN.RPN_ON = True
    if cfg.RPN.RPN_ON or cfg.RETINANET.RETINANET_ON:
        cfg.TEST.PRECOMPUTED_PROPOSALS = False
    if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
        assert cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS, \
            "Path to the weight file must not be empty to load imagenet pertrained resnets."
    if set([cfg.MRCNN.ROI_MASK_HEAD, cfg.KRCNN.ROI_KEYPOINTS_HEAD]) & _SHARE_RES5_HEADS:
        cfg.MODEL.SHARE_RES5 = True
    if version.parse(torch.__version__) < version.parse('0.4.0'):
        cfg.PYTORCH_VERSION_LESS_THAN_040 = True
        # create alias for PyTorch version less than 0.4.0
        init.uniform_ = init.uniform
        init.normal_ = init.normal
        init.constant_ = init.constant
        # nn.GroupNorm = nnlib.GroupNorm # DIFF: not support GroupNorm yet.
            
# Default: config_default.yaml
if os.path.exists('config_default.yaml'):
    config_load('config_default.yaml')


# For convenient debug
config_load('config.yaml')
assert_and_infer_cfg(CONFIG)
