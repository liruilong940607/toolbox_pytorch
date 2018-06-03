from __future__ import print_function  

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as transF
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
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
from tqdm import tqdm
from easydict import EasyDict as edict

sys.path.insert(0, '../../')
from nnlib.tools import to_varabile, AverageMeter

parser = argparse.ArgumentParser(description='Training code')
parser.add_argument('--config', default='config.yaml', type=str, help='yaml config file')
args = parser.parse_args()
CONFIG = edict(yaml.load(open(args.config, 'r')))
print ('==> CONFIG is: \n', CONFIG, '\n')

LOGDIR = '%s/%s_%d'%(CONFIG.LOGS.LOG_DIR, CONFIG.NAME, int(time.time()))
SNAPSHOTDIR = '%s/%s_%d'%(CONFIG.LOGS.SNAPSHOT_DIR, CONFIG.NAME, int(time.time()))
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
if not os.path.exists(SNAPSHOTDIR):
    os.makedirs(SNAPSHOTDIR)
    
######################################################################################################################
#                                                    Model
######################################################################################################################

from models.resnetXtFPN import resnet50

######################################################################################################################
#                                                Dataset:Imagenet
######################################################################################################################
def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)

class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (Array Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        height, width = img.shape[0:2]
        for attempt in range(10):
            area = width * height
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= width and h <= img.size[1]:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback
        w = min(width, height)
        i = (height - w) // 2
        j = (height - w) // 2
        return i, j, w, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return transF.resized_crop(img, i, j, h, w, self.size, self.interpolation)



def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

class ImagenetDataset(object):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, istrain, extensions=['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']):
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        self.root = root
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        
        self.istrain = istrain
        print ('==> Load Dataset: \n', {'dataset': root, 'istrain:': istrain, 'len': self.__len__()}, '\n')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image = cv2.imread(path)
        
        if self.istrain:
            image = RandomResizedCrop(CONFIG.NET.INPUT_RES)(image)
            if CONFIG.AUG.FILP_ON:
                if random.random()<0.5:
                    image = image[:,::-1,:]
            
        else:
            image = cv2.resize(image, (CONFIG.EVAL.RESIZE_RES, CONFIG.EVAL.RESIZE_RES), interpolation=cv2.INTER_LINEAR)
            image = transF.center_crop(image, self.size)

        # t.sub_(m).div_(s)
        input = image.astype(np.float32) * CONFIG.DATASET.STD - CONFIG.DATASET.MEAN
        input = input.transpose(2,0,1)
        
        return input, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


######################################################################################################################
#                                                   Training
######################################################################################################################
def train(dataLoader, model, epoch, optimizer, criterion):
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    losses = AverageMeter('losses')
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(dataLoader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, label = data
        input_var = to_varabile(input, requires_grad=True, is_cuda=True)
        label_var = to_varabile(label, requires_grad=False, is_cuda=True)
        
        out_var = model(input_var)
        loss = criterion(out_var, label_var)
        losses.update(loss.data[0], input.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % CONFIG.LOGS.PRINT_FREQ == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})\t'.format(
                   epoch, i, len(dataLoader), batch_time=batch_time, data_time=data_time, 
                   losses=losses ))
        
        if i % CONFIG.LOGS.LOG_FREQ == 0:
            save_image(input_var, os.path.join(LOGDIR, 'epoch%d_%d.jpg'%(epoch, i)), nrow=input_var.size(0), padding=2,
                       normalize=True, range=None, scale_each=True, pad_value=0)
            
        if i % CONFIG.LOGS.SNAPSHOT_FREQ == 0:
            torch.save(model.state_dict(), os.path.join(SNAPSHOTDIR, '%d_%d.pkl'%(epoch,i)))

        
def main():
    dataset_train = ImagenetDataset(root = CONFIG.DATASET.TRAINSETDIR, istrain = True)
    dataset_val = ImagenetDataset(root = CONFIG.DATASET.VALSETDIR, istrain = False)
    
    BATCHSIZE = CONFIG.SOLVER.IMG_PER_GPU * len(CONFIG.SOLVER.GPU_IDS)
    dataLoader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCHSIZE, shuffle=True, num_workers=CONFIG.SOLVER.WORKERS, pin_memory=False)
    dataLoader_val = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    
    model = resnet50(pretrained=False, num_classes=1000).cuda()
    optimizer = torch.optim.Adam(model.parameters(), CONFIG.SOLVER.LR, weight_decay=CONFIG.SOLVER.WEIGHTDECAY)
    criterion = nn.CrossEntropyLoss().cuda()
    
    epoches = 90
    for epoch in range(epoches):
        print ('===========>   [Epoch %d] training    <==========='%epoch)
        train(dataLoader_train, model, epoch, optimizer, criterion)

    
if __name__ == '__main__' :
    main()

