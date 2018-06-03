import cv2
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from .pose_align import PoseAffineTemplate

def to_varabile(arr, requires_grad=False, is_cuda=True):
    if type(arr) == np.ndarray:
        tensor = torch.from_numpy(arr)
    else:
        tensor = arr
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var

# ===========================
# AffineAlignLayer operations
# ===========================

'''
Calculate Align Matrix Based on Estimated Keypoints And Template
Usage:
    -> H_feat, W_feat: size of the map needed to be aligned.
    -> keypoints: normalized. support 17 and 29. a (N, npart, 3) size float array 
    -> align_size: the size of output. 
    -> template_size: in the center of output, how large is the template.
    -> Haug: support using augument matrix. a (2, 3) size float array 
    
    <- Hs: align matrix. a (N, 2, 3) size numpy array 
    
'''
def calcAffineMatrix(H_feat, W_feat, keypoints, align_size, template_size, Haug=None):
    offset = (align_size-template_size)/2
    H_offset = np.array([[1.0, 0.0, offset], [-0.0, 1.0, offset]])

    N, npart, _ = keypoints.shape
    keypoints_f = keypoints*[W_feat, H_feat, 1.0]

    template = PoseAffineTemplate(npart, template_size, template_size)

    Hs = []
    for keypoint in keypoints_f:
        H = template.estimateH(keypoint, W_feat, H_feat) # (W_feat, H_feat) -> (template_size, template_size)
        H = H_offset.dot(np.vstack((H, np.array([0,0,1], dtype=np.float32))))
        if Haug is not None: # (2, 3)
            H = np.dot(Haug, np.vstack((H, np.array([0,0,1], dtype=np.float32))))
        Hs.append(H)
    Hs = np.array(Hs)
    return Hs

'''
Affine align operation. based on cv2.warpAffine Using cv2.INTER_LINEAR
Usage:
    -> features: (bz, H, W, 3) | (bz, H, W) float
    -> idxs: (N, )
    -> align_size: the size of output. 
    -> Hs: result of calcAffineMatrix(). a (N, 2, 3) size numpy array 
    
    <- rois: feature rois after align. (N, align_size, align_size, 3) | (N, align_size, align_size)
Example:
    NN, H_feat, W_feat = features.shape[0:3]
    Hs = calcAffineMatrix(H_feat, W_feat, keypoints, align_size, template_size, Haug)
    features_roi = affine_align(features, indexs, align_size, Hs)
'''
def affine_align(features, idxs, align_size, Hs):
    rois = []
    for idx, H in zip(idxs, Hs):
        feature = features[idx].copy()
        roi = cv2.warpAffine(feature, H, (align_size, align_size), borderValue=0, flags=cv2.INTER_LINEAR) 
        rois.append(roi)
    rois = np.array(rois)
    return rois

'''
Affine align operation GPU version. based on torch.nn.functional.affine_grid() Using bilinear
Usage:
    -> features: (bz, 3, H, W) | (bz, 1, H, W) Variable
    -> idxs: (N, ) numpy array
    -> align_size: the size of output. 
    -> Hs: result of calcAffineMatrix(). a (N, 2, 3) size numpy array 
    
    <- rois: feature rois after align. (N, 3, align_size, align_size) | (N, 1, align_size, align_size) Variable
Example:
    NN, _, H_feat, W_feat = features.shape[0:4]
    Hs = calcAffineMatrix(H_feat, W_feat, keypoints, align_size, template_size, Haug)
    features_roi = affine_align_gpu(features_var, indexs, align_size, Hs)
'''
def affine_align_gpu(features, idxs, align_size, Hs):
    
    def _transform_matrix(Hs, w, h):
        _Hs = np.zeros(Hs.shape, dtype = np.float32)
        for i, H in enumerate(Hs):
            try:
                H0 = np.concatenate((H, np.array([[0, 0, 1]])), axis=0)
                A = np.array([[2.0 / w, 0, -1], [0, 2.0 / h, -1], [0, 0, 1]])
                A_inv = np.array([[w / 2.0, 0, w / 2.0], [0, h / 2.0, h/ 2.0], [0, 0, 1]])
                H0 = A.dot(H0).dot(A_inv)
                H0 = np.linalg.inv(H0)
                _Hs[i] = H0[:-1]
            except:
                print ('[error in (affine_align_gpu)]', H)
        return _Hs
    
#     def _transform_matrix(Hs, w, h):
#         '''
#         Hs : (N x 2 x 3), used by cv2.warpAffine()
#         _Hs : (N x 2 x 3), used by torch.nn.functional.affine_grid()
        
#         The coordinate system used by two functions are different. So need to transform
#         [dx]   [A,  B]   [-w/2.0]   [w/2.0]   [w/2.0*_dx]
#              -         *          -         = 
#         [dy]   [-B, A]   [-h/2.0]   [h/2.0]   [h/2.0*_dy]
#         '''
#         for i, H in enumerate(Hs):
#             try:
#                 H33 = np.eye(3, dtype = np.float32)
#                 H33[0:2, :] = H
#                 _dx, _dy = (H[0:2, 2] - H[0:2, 0:2].dot(np.array([-w, -h]))/2 - np.array([w,h])/2) * 2 / np.array([w,h])
#                 H33[0:2, 2] = np.array([_dx, _dy]) 
#                 H_inv = np.linalg.inv(H33)
#                 Hs[i,:,:] = H_inv[0:2, :]
#             except:
#                 print ('[error in (affine_align_gpu)]', H)
#         return Hs
    
    bz, C_feat, H_feat, W_feat = features.size()
    N = len(idxs)
    feature_select = features[idxs] # (N, feature_channel, feature_size, feature_size)
    # transform coordinate system
    Hs_new = _transform_matrix(Hs, w=W_feat, h=H_feat) # return (N, 2, 3)
    Hs_var = to_varabile(Hs_new, requires_grad=False, is_cuda=True)
    '''
    theta (Variable) – input batch of affine matrices (N x 2 x 3)
    size (torch.Size) – the target output image size (N x C x H x W) 
    output Tensor of size (N x H x W x 2)
    '''
    flow = F.affine_grid(theta=Hs_var, size=(N, C_feat, H_feat, W_feat)).float().cuda()
    flow = flow[:,:align_size, :align_size, :]
    '''
    input (Variable) – input batch of images (N x C x IH x IW)
    grid (Variable) – flow-field of size (N x OH x OW x 2)
    padding_mode (str) – padding mode for outside grid values ‘zeros’ | ‘border’. Default: ‘zeros’
    '''
    rois = F.grid_sample(feature_select, flow, mode='bilinear', padding_mode='border') # 'zeros' | 'border' 
    return rois
    
    
    