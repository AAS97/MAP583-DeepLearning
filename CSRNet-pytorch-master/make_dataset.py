import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

import pandas as pd



class DotsDataset:

    '''
    Dots Dataset
    '''
    def __init__(self, csv_file ,gt_downsample=0, shape=None, shuffle = True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        '''
        img_root: the root path of img.
        gt_dmap_root: the root path of ground-truth density-map.
        gt_downsample: default is 0, denote that the output of deep-model is the same size as input image.
        '''
        self.file = pd.read_csv(csv_file)

        self.nSamples=len(self.file['counts'])

        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'

        csv_file = self.file

        cell_img_path = csv_file.iloc[idx, 0]
        dots_img_path = csv_file.iloc[idx, 1]


        cell_img=plt.imread(cell_img_path)
        dots_img = plt.imread(dots_img_path)

        cell_img_tensor = torch.tensor(cell_img, dtype=torch.float)
        dots_img_tensor = torch.tensor(dots_img, dtype=torch.float)

        # print(gt_dmap_tensor.sum())

        if self.transform is not None:
            cell_img_tensor = self.transform(cell_img_tensor)
            dots_img_tensor = self.transform(dots_img_tensor)

        return cell_img_tensor, dots_img_tensor




class DensityDataset:

    '''
    Density Dataset
    '''
    def __init__(self, csv_file ,gt_downsample=0, shape=None, shuffle = True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        '''
        img_root: the root path of img.
        gt_dmap_root: the root path of ground-truth density-map.
        gt_downsample: default is 0, denote that the output of deep-model is the same size as input image.
        '''
        self.file = pd.read_csv(csv_file)

        self.nSamples=len(self.file['counts'])

        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'

        csv_file = self.file

        cell_img_path = csv_file.iloc[idx, 0]
        density_img_path = csv_file.iloc[idx, 2]

        cell_img=plt.imread(cell_img_path)
        density_img = plt.imread(density_img_path)
        
        cell_img_tensor = torch.tensor(cell_img, dtype=torch.float)
        density_img_tensor = torch.tensor(density_img, dtype=torch.float)

        # print(gt_dmap_tensor.sum())

        if self.transform is not None:
            cell_img_tensor = self.transform(cell_img_tensor)
            density_img_tensor = self.transform(density_img_tensor)

        return cell_img_tensor, density_img_tensor

class CountDataset:

    '''
    Count Dataset
    '''
    def __init__(self, csv_file ,gt_downsample=0, shape=None, shuffle = True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        '''
        csv_file: csv file where paths to cell images an the associated number of cells are stored
        '''
        self.file = pd.read_csv(csv_file)

        self.nSamples=len(self.file['counts'])

        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'

        csv_file = self.file

        cell_img_path = csv_file.iloc[idx, 0]
        counts = csv_file.iloc[idx, 3]

        cell_img=plt.imread(cell_img_path)
        
        cell_img_tensor = torch.tensor(cell_img, dtype=torch.float)

        # print(gt_dmap_tensor.sum())

        if self.transform is not None:
            cell_img_tensor = self.transform(cell_img_tensor)

        return cell_img_tensor, counts


