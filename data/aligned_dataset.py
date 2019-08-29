import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from skimage import io
import SimpleITK as sitk
import numpy as np





class AlignedDataset(BaseDataset):   ## train
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))

        assert(opt.resize_or_crop == 'resize_and_crop')



        self.transform = torch.from_numpy
        self.eps = 10**(-12)

    def __getitem__(self, index):

        AB_path = self.AB_paths[index]
        img0 = io.imread(AB_path, plugin = 'simpleitk')



        AB = np.zeros((1,self.opt.fineSize*2,self.opt.fineSize,self.opt.fineSize))
        AB[0,0:self.opt.fineSize,:,:] = img0[0:self.opt.fineSize,:,:]

        AB[0,self.opt.fineSize:self.opt.fineSize*2,:,:] = img0[self.opt.fineSize:(self.opt.fineSize*2),:,:]

        AB = self.transform(AB)

        A = AB[:,:self.opt.fineSize,:,:]
        B = AB[:,self.opt.fineSize:self.opt.fineSize*2,:,:]

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(3) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(3, idx)
            B = B.index_select(3, idx)


        return {'A': A, 'B': B,
            'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'





class TestAlignedDataset(BaseDataset):   
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))

        assert(opt.resize_or_crop == 'resize_and_crop')



        self.transform = torch.from_numpy

    def __getitem__(self, index):

        AB_path = self.AB_paths[index]
        img0 = io.imread(AB_path, plugin = 'simpleitk')
        img0 = img0.reshape((2*self.opt.fineSize,self.opt.fineSize,self.opt.fineSize))
        AB = np.zeros((1,self.opt.fineSize*2,self.opt.fineSize,self.opt.fineSize))
        AB[0,0:self.opt.fineSize,:,:] = img0[0:self.opt.fineSize,:,:]

        AB[0,self.opt.fineSize:self.opt.fineSize*2,:,:] = img0[self.opt.fineSize:(self.opt.fineSize*2),:,:]


        AB = self.transform(AB)



        A = AB[:,:self.opt.fineSize,:,:]
        B = AB[:,self.opt.fineSize:self.opt.fineSize*2,:,:]



        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(3) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(3, idx)
            B = B.index_select(3, idx)


        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'