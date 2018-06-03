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


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                     padding=1, bias=False)
    def forward(self, inputlist):
        print inputlist.shape
        return self.conv(inputlist)
    
import parallel
model = Test()
model = parallel.DataParallel(model, device_ids=[0,1], minibatch=True).cuda()

img = cv2.imread('/home/dalong/testimg.png')
features = np.float32(img.transpose(2, 0, 1)[np.newaxis, :, :, :])
features = to_varabile(features, True, True)

model([features, features])
