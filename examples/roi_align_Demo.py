import os
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import pylab
% matplotlib inline
plt.rcParams['figure.figsize'] = (15, 15)

def to_varabile(arr, requires_grad=False, is_cuda=True):
    if type(arr) == np.ndarray:
        tensor = torch.from_numpy(arr)
    else:
        tensor = arr
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var

def myshow(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
# test roialign
from roi_align.modules.roi_align import RoIAlign
img = cv2.imread('/home/dalong/testimg.png')
features = np.float32(img.transpose(2, 0, 1)[np.newaxis, :, :, :])
features = to_varabile(features, True, True)
bbox = [0, 10, 10, 400, 400] # [batch_ind, x1, y1, x2, y2]
rois = np.array([bbox], dtype=np.float32)
rois = to_varabile(rois, False, True)
aligned_height = 500
aligned_width = 500
spatial_scale = 1.0
sampling_ratio = 0.0

alignlayer = RoIAlign(aligned_height, aligned_width, spatial_scale, sampling_ratio)

res = alignlayer(features, rois)
vis = res.cpu().detach().numpy()[0].transpose(1,2,0)
print vis.shape
print np.max(vis)
print np.min(vis)
myshow(np.uint8(vis[:,:,::-1]))
myshow(np.uint8(img[bbox[2]:bbox[4],bbox[1]:bbox[3],::-1]))