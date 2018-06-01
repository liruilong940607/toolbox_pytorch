import cv2
import sys
import os
import numpy as np
import math
import random
import json

import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 10)




def get_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*sx    -sin(a + shear)*sy     0]
    #                              [ sin(a)*sx    cos(a + shear)*sy     0]
    #                              [     0                  0          1]

    angle = math.radians(angle)
    shear = math.radians(shear)
    
    T = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]]).astype(np.float32)
    C = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]]).astype(np.float32)
    RSS = np.array([[ math.cos(angle)*scale[0], -math.sin(angle + shear)*scale[1], 0],
                    [ math.sin(angle)*scale[0],  math.cos(angle + shear)*scale[1], 0],
                    [ 0, 0, 1]]).astype(np.float32)
    C_inv = np.linalg.inv(np.mat(C))
    M = T.dot(C).dot(RSS).dot(C_inv)
    return M

def get_inverse_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*sx    -sin(a + shear)*sy     0]
    #                              [ sin(a)*sx    cos(a + shear)*sy     0]
    #                              [     0                  0          1]
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    angle = math.radians(angle)
    shear = math.radians(shear)
    
    T = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]]).astype(np.float32)
    C = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]]).astype(np.float32)
    RSS = np.array([[ math.cos(angle)*scale[0], -math.sin(angle + shear)*scale[1], 0],
                    [ math.sin(angle)*scale[0],  math.cos(angle + shear)*scale[1], 0],
                    [ 0, 0, 1]]).astype(np.float32)
    T_inv = np.linalg.inv(np.mat(T))
    RSS_inv = np.linalg.inv(np.mat(RSS))
    C_inv = np.linalg.inv(np.mat(C))
    M = C.dot(RSS_inv).dot(C_inv).dot(T_inv)
    return M


def masks2bboxes(masks):
    '''
    masks: (N, H, W) or [N](H, W)
    '''
    bboxes = []
    for mask in masks:
        if np.max(mask)<=0.5:
            continue
        idxs = np.where(mask>0.5)
        ymax = np.max(idxs[0])
        ymin = np.min(idxs[0])
        xmax = np.max(idxs[1])
        xmin = np.min(idxs[1])
        bboxes.append([xmin, ymin, xmax, ymax])
    bboxes = np.array(bboxes)
    return bboxes

def draw_masks_to_canvas(img, masks):
    '''
    img: (H, W, 3) 
    masks: (N, H, W) or [N](H, W)
    '''
    import random
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [255, 0, 0]]
    oriImg = img.copy()
    ratio = np.zeros(img.shape, np.float32)
    canvas3c_all = np.zeros(img.shape, np.uint8)
    for mask in masks:
        if np.max(mask)<=0.5:
            continue
        color_id = random.randint(0, len(colors)-1)
        canvas = mask.copy()
        canvas[canvas>0.5] = 1
        canvas[canvas<=0.5] = 0
        canvas3c = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)*np.array(colors[color_id])
        ratio[canvas==1, :] = 0.5
        canvas3c_all[canvas==1, :] = canvas3c[canvas==1, :]
    
    to_plot = img*(1-ratio)+canvas3c_all*ratio
    to_plot = np.uint8(to_plot)
    
    bboxes = masks2bboxes(masks)
    for bbox in bboxes:
        cv2.rectangle(to_plot, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0), 3)
    
    return to_plot

def visualize(canvas_inp, keypoints_inp, group=True):
    '''
    canvas_inp: (H, W, 3) [0, 255]
    keypoints_inp: (N, np, 3), [0, 1]
    return : (N, H, W, 3)
    '''
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    
    canvas = np.float32(canvas_inp)
    H, W, C = canvas.shape
    keypoints = np.int32(keypoints_inp*[W, H, 1.0])
    N, npart, _ = keypoints.shape
    
    if group:
        canvas_t = canvas.copy()
        for keypoint in keypoints:
            for i, part in enumerate(keypoint):
                if part[0]==0 and part[1]==0:
                    continue
                cv2.circle(canvas_t, tuple(part[0:2]), 11, colors[i%len(colors)], thickness=-1)
        to_plot = cv2.addWeighted(canvas, 0.3, canvas_t, 0.7, 0)
        return np.uint8(to_plot[np.newaxis, ...]) # (1, H, W, 3)
    else:
        to_plots = np.zeros((N, H, W, C), dtype = np.float32)
        for ni, keypoint in enumerate(keypoints):
            canvas_t = canvas.copy()
            for i, part in enumerate(keypoint):
                if part[0]==0 and part[1]==0:
                    continue
                cv2.circle(canvas_t, tuple(part[0:2]), 11, colors[i%len(colors)], thickness=-1)
            to_plots[ni] = cv2.addWeighted(canvas, 0.3, canvas_t, 0.7, 0)
        return np.uint8(to_plots) # (N, H, W, 3)
    
    
def aug_matrix(w1, h1, w2, h2, angle_range=(-45, 45), scale_range=(0.5, 1.5), trans_range=(-0.3, 0.3)):
    ''' 
    first Translation, then rotate, final scale.
        [sx, 0, 0]       [cos(theta), -sin(theta), 0]       [1, 0, dx]       [x]
        [0, sy, 0] (dot) [sin(theta),  cos(theta), 0] (dot) [0, 1, dy] (dot) [y]
        [0,  0, 1]       [         0,           0, 1]       [0, 0,  1]       [1]
    '''
    dx = (w2-w1)/2.0
    dy = (h2-h1)/2.0
    matrix_trans = np.array([[1.0, 0, dx],
                             [0, 1.0, dy],
                             [0, 0,   1.0]])

    angle = random.random()*(angle_range[1]-angle_range[0])+angle_range[0]
    scale = random.random()*(scale_range[1]-scale_range[0])+scale_range[0]
    scale *= np.min([float(w2)/w1, float(h2)/h1])
    alpha = scale * math.cos(angle/180.0*math.pi)
    beta = scale * math.sin(angle/180.0*math.pi)

    trans = random.random()*(trans_range[1]-trans_range[0])+trans_range[0]
    centerx = w2/2.0 + w2*trans
    centery = h2/2.0 + h2*trans
    H = np.array([[alpha, beta, (1-alpha)*centerx-beta*centery], 
                  [-beta, alpha, beta*centerx+(1-alpha)*centery],
                  [0,         0,                            1.0]])

    H = H.dot(matrix_trans)[0:2, :]
    return H 

def kpt_affine(kpt, mat):
    kpt = np.array(kpt)
    shape = kpt.shape
    kpt = kpt.reshape(-1, 2)
    return np.dot( np.concatenate((kpt, kpt[:, 0:1]*0+1), axis = 1), mat.T ).reshape(shape)
    
def loadjson(File):
    annos_ins = json.load(open(File))
    annos_img = {}
    for anno in annos_ins:
        image_id = anno['image_id']
        if image_id in annos_img.keys():
            annos_img[image_id].append(anno)
        else:
            annos_img[image_id] = [anno]
    return annos_img


# def putGaussianMaps(entry, center, stride, grid_x, grid_y, sigma, weight=1.0):
#     start = stride / 2.0 - 0.5  # 0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
#     threshold = 4.6025 * sigma ** 2 * 2
#     sqrt_threshold = math.sqrt(threshold)
#     #(start + g_x * stride - center[0]) ** 2 + (start + g_y * stride - center[1]) ** 2 <= threshold ** 2
#     start_y = max(0, int((center[1]-sqrt_threshold-start)/stride))
#     while (start+start_y*stride-center[1]) < -sqrt_threshold:
#         start_y += 1
#     for g_y in range(start_y, grid_y):
#         y = start + g_y * stride
#         sum = (y - center[1]) ** 2
#         th = threshold - sum
#         if th < 0:
#             break
#         sqrt_threshold = math.sqrt(th)
#         start_x = max(0, int((center[0]-sqrt_threshold-start)/stride))
#         while (start+start_x*stride-center[0]) < -sqrt_threshold:
#             start_x += 1
#         for g_x in range(start_x, grid_x):
#             x = start + g_x * stride
#             d2 = sum + (x - center[0]) ** 2
#             if d2 > threshold:
#                 break
#             exponent = d2 / 2.0 / sigma / sigma
#             entry[g_y, g_x] += math.exp(-exponent)*weight
#             if (entry[g_y, g_x] > 1):
#                 entry[g_y, g_x] = 1*weight

                
def putGaussianMaps(entry, center, stride, grid_x, grid_y, sigma, weight=1.0):
    start = stride / 2.0 - 0.5  # 0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
    threshold = 4.6025 * sigma ** 2 * 2
    sqrt_threshold = math.sqrt(threshold)
    #(start + g_x * stride - center[0]) ** 2 + (start + g_y * stride - center[1]) ** 2 <= threshold ** 2
    min_y = max(0, int((center[1] - sqrt_threshold - start) / stride))
    max_y = min(grid_y-1, int((center[1] + sqrt_threshold - start) / stride))
    min_x = max(0, int((center[0] - sqrt_threshold - start) / stride))
    max_x = min(grid_x-1, int((center[0] + sqrt_threshold - start) / stride))
    g_y = np.arange(min_y,max_y+1)[:, None]
    g_x = np.arange(min_x,max_x+1)
    y = start + g_y * stride
    x = start + g_x * stride
    d2 = ((x - center[0]) ** 2 + (y - center[1]) ** 2) / 2 / sigma ** 2
    idx = np.where(d2<4.6025)
    circle = entry[min_y:max_y+1,min_x:max_x+1][idx]
    circle += np.exp(-d2[idx])  ## circle += np.exp(-d2[idx]) ?? 
    circle[circle > 1] = 1
    entry[min_y:max_y + 1, min_x:max_x + 1][idx] = circle * weight

def putVecMaps(entryX, entryY, centerA_ori, centerB_ori, grid_x, grid_y, stride, thre):
    centerA = centerA_ori*(1.0/stride)
    centerB = centerB_ori*(1.0/stride)
    line = centerB - centerA
    min_x = max(int(round(min(centerA[0], centerB[0])-thre)), 0);
    max_x = min(int(round(max(centerA[0], centerB[0])+thre)), grid_x);
    min_y = max(int(round(min(centerA[1], centerB[1])-thre)), 0);
    max_y = min(int(round(max(centerA[1], centerB[1])+thre)), grid_y);
    norm_line = math.sqrt(line[0]*line[0] + line[1]*line[1])
    if norm_line == 0:
        line = [0,0]
    else:
        line = 1.0* line / norm_line
    for g_y in range(min_y, max_y):
        for g_x in range(min_x, max_x):
            vec = [g_x - centerA[0], g_y - centerA[1]]
            dist = abs(vec[0]*line[1] - vec[1]*line[0])
            if dist <= thre:
                entryX[g_y, g_x] = line[0]
                entryY[g_y, g_x] = line[1]

def generate_label(keypoint, crop_size_x, crop_size_y, stride, num_parts, sigma=5, thre=1, weights=None):
    '''
    keypoint: [num_parts*3,]
    '''
    grid_x = int(crop_size_x / stride)
    grid_y = int(crop_size_y / stride)
    assert len(keypoint)==num_parts*3
    if num_parts==18:
        c1 = [2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16]
        c2 = [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
        connection = [[c1[i], c2[i]] for i in range(len(c1))]
    elif num_parts==29:
        c1 = [0, 0, 1, 1, 2, 0, 3, 4, 5, 6, 5, 8, 10, 12, 7, 9,  11, 13, 5, 16, 18, 21, 23, 25, 7, 17, 20, 22, 24, 26, 18, 19] 
        c2 = [1, 2, 2, 3, 4, 6, 5, 7, 6, 7, 8, 10,12, 14, 9, 11, 13, 15, 16,18, 21, 23, 25, 27,17, 20, 22, 24, 26, 28, 19, 20] 
        connection = [[c1[i] + 1, c2[i] + 1] for i in range(len(c1))]
    label = np.zeros((grid_y, grid_x, len(connection)*2+num_parts), dtype = np.float32)
    for i in range(len(connection)):
        if keypoint[3*(connection[i][0]-1)+0] >= crop_size_x or keypoint[3*(connection[i][0]-1)+0] < 0:
            continue
        if keypoint[3*(connection[i][0]-1)+1] >= crop_size_y or keypoint[3*(connection[i][0]-1)+1] < 0:
            continue
        if (keypoint[3*(connection[i][0]-1)+2] > 0 and keypoint[3*(connection[i][1]-1)+2] > 0):
            putVecMaps(label[:,:,2*i], label[:,:,2*i+1], 
                       np.array([keypoint[3*(connection[i][0]-1)], keypoint[3*(connection[i][0]-1)+1]]), 
                       np.array([keypoint[3*(connection[i][1]-1)], keypoint[3*(connection[i][1]-1)+1]]), 
                       grid_x, grid_y, stride, thre)
    for i in range(num_parts):
        if weights is None:
            weight = 1.0
        else:
            weight = weights[i]
        center = [keypoint[i*3], keypoint[i*3+1]]
        if keypoint[i*3+0] >= crop_size_x or keypoint[i*3+0] < 0:
            continue
        if keypoint[i*3+1] >= crop_size_y or keypoint[i*3+1] < 0:
            continue
        if keypoint[i*3+2] > 0:
            putGaussianMaps(label[:,:,len(connection)*2+i], center, stride, grid_x, grid_y, sigma, weight)
    return label

