import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torch.autograd import Variable

import sys
import os 
import time
import numpy as np
import cv2
import argparse
import yaml
import json
import random
import copy
from tqdm import tqdm

from nnlib.layers import Conv
from nnlib import parallel
from tools import AverageMeter

parser = argparse.ArgumentParser(description='Training code')
parser.add_argument('--config', default='config_template.yaml', type=str, help='yaml config file')
args = parser.parse_args()
CONFIG = yaml.load(open(args.config, 'r'))

######################################################################################################################
#                                                   Dataset
######################################################################################################################

class MyDataset(object):
    def __init__(self, ImageDir, AnnoDir):
        self.imgIds = range(10)
        pass
        
    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        return self.loadImage(idx)
    
    def loadImage(self, idx):
        image = np.zeros((3, 100, 100), np.float32)
        label = np.ones((100, 100), np.uint8)
        return image, label
    
    def collate_fn(self, batch):
        image_batch = []
        label_batch = []
        for i, data in enumerate(batch):
            image, label = batch
            image_batch.append(image)
            label_batch.append(label)
        image_batch = np.array(image_batch) # (bz, 3, 100, 100)
        label_batch = np.array(label_batch) # (bz, 100, 100)
        return image_batch, label_batch

######################################################################################################################
#                                                   Network
######################################################################################################################

class MyNetwork(nn.Module):
    def __init__(self, istrain=True):
        super(MyNetwork, self).__init__()
        self.conv = Conv(inp_dim=3, out_dim=64, kernel_size=3, stride=1, bias=True, bn=True, relu=True, relu_before_bn=True)
     
    def forward(self, input):
        return self.conv(input)

    def calc_loss(self, pred, gt):
        criterion = nn.CrossEntropyLoss(ignore_index=255) 
        loss = criterion(pred, gt)
        return loss

    def calc_accuracy(self, pred, gt)
        pass
        return 0

######################################################################################################################
#                                                   Training
######################################################################################################################

def train(dataLoader, model, optimizer, epoch, iteration):
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    losses = AverageMeter('losses')
    scores = AverageMeter('accuracy') 
        
    # switch to train mode
    netmodel.train()

    end = time.time()
    for i, data in enumerate(dataLoader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = data
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = model.calc_loss(output, target_var)

        # measure accuracy and record loss
        accuracy = model.calc_accuracy(output, target_var)
        losses.update(loss.data[0], input.size(0))
        scores.update(accuracy[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {scores.val:.3f} ({scores.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, scores=scores))
            

######################################################################################################################
#                                                   Evaluation
######################################################################################################################

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


######################################################################################################################
#                                                      Main
######################################################################################################################

def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    if args.arch=='mobilenetv1':
        import basemodel
        from basemodel.mobilenet_v1_pretrain import get_model
        model = get_model()
    elif args.arch=='resnet101':
        from basemodel.resnetXtFPN import resnet101
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = resnet101(pretrained=True, num_classes=1000)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = resnet101(pretrained=False, num_classes=1000)
    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay) 

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


model = parallel.DataParallel(model, device_ids=[0,1], minibatch=False).cuda()

