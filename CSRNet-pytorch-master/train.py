import sys
import os

import warnings

from model import CSRNet
import make_dataset
from utils import save_checkpoint
from torchsummary import summary

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset
import time

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_csv', metavar='TRAIN',
                    help='path to train csv')
parser.add_argument('test_csv', metavar='TEST',
                    help='path to csv json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu', metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task', metavar='TASK', default=None, type=str,
                    help='task id to use.')


def main():

    global args, best_prec1

    best_prec1 = 1e6

    args = parser.parse_args()
    args.original_lr = 1e-5
    args.lr = 1e-5
    args.batch_size = 1
    args.momentum = 0.95
    args.decay = 5*1e-4
    args.start_epoch = 0
    args.epochs = 100
    args.steps = [-1, 20, 40, 60]
    args.scales = [1, 0.1, 0.1, 0.1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30
    # with open(args.train_json, 'r') as outfile:
    #     train_list = json.load(outfile)
    # with open(args.test_json, 'r') as outfile:
    #     val_list = json.load(outfile)

    csv_train_path = args.train_csv
    csv_test_path = args.test_csv

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # torch.cuda.manual_seed(args.seed)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = CSRNet()

    #summary(model, (3, 256, 256))

    model = model.to(device)

    criterion = nn.MSELoss(size_average=False).to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
    precs = []
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        train(csv_train_path, model, criterion, optimizer, epoch)
        prec1 = validate(csv_test_path, model, criterion)
        precs.append(prec1)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'MAE_history' : precs
        }, is_best, args.task)


def train(csv_path, model, criterion, optimizer, epoch):

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        make_dataset.DensityDataset(csv_path,
                                    shuffle=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    train=True,
                                    seen=model.seen,
                                    batch_size=args.batch_size,
                                    num_workers=args.workers),
        batch_size=args.batch_size)

    # train_loader = torch.utils.data.DataLoader(
    #     make_dataset.CountDataset(csv_path,
    #                                 shuffle=True,
    #                                 transform=transforms.Compose([
    #                                     transforms.ToTensor(),
    #                                 ]),
    #                                 train=True,
    #                                 seen=model.seen,
    #                                 batch_size=args.batch_size,
    #                                 num_workers=args.workers),
    #     batch_size=args.batch_size)


    print('epoch %d, processed %d samples, lr %.10f' %
          (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()

    for i, (img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.to(device)
        img = Variable(img)
        # print(f'Img shape {img.shape}')
        output = model(img)
        # print(f"output dim {output.shape} ")
        target = target.type(torch.FloatTensor).unsqueeze(0).to(device)
        target = Variable(target)

        loss = criterion(output, target)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))


def validate(csv_path, model, criterion):
    print('begin test')

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_loader = torch.utils.data.DataLoader(
        make_dataset.DensityDataset(csv_path,
                                    shuffle=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),  train=False),
        batch_size=args.batch_size)

    # test_loader = torch.utils.data.DataLoader(
    #     make_dataset.CountDataset(csv_path,
    #                                 shuffle=False,
    #                                 transform=transforms.Compose([
    #                                     transforms.ToTensor(),
    #                                 ]),  train=False),
    #     batch_size=args.batch_size)


    model.eval()

    mae = 0

    for i, (img, target) in enumerate(test_loader):
        img = img.to(device)
        img = Variable(img)
        output = model(img)

        mae += abs(output.data.sum() -
                   target.sum().type(torch.FloatTensor).to(device))

    mae = mae/len(test_loader)
    print(' * MAE {mae:.3f} '
          .format(mae=mae))

    return mae


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    args.lr = args.original_lr

    for i in range(len(args.steps)):

        scale = args.scales[i] if i < len(args.scales) else 1

        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
