from __future__ import print_function, division
import os
import sys
import torch
import random
import collections
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import Dataset


class potsdamloader(Dataset):
    '''
    ISPRS dataset reader
    '''
    def __init__(self, root, split='train', crop_size=512, scale=False, rotate=True, HorizontalFlip=True, normalize=False, degree=10):
        super().__init__()
        self.root = root
        self.split = split

        self.image_list = []
        self.label_list = []

        # preprocessing
        self.crop_size = 512
        self.degree = degree
        self.rotate = rotate
        self.HorizontalFlip = HorizontalFlip
        self.scale = scale
        self.normalize = normalize

        # mean, std
        self.image_mean = [0.38083647, 0.33612353, 0.35943412]
        self.image_std = [0.10240941, 0.10278902, 0.10292039]
        # self.ir_mean = [0.38083647]
        # self.ir_std = [0.10240941]
        # self.nDSM_mean = [0.18184235]
        # self.nDSM_std = [0.10249333]

        self.ignore_label = 255

        # ./train
        # ├── IR
        # ├── Label
        # ├── nDSM
        # └── RGB

        for image_fp in os.listdir(os.path.join(self.root, self.split, 'image')):
            # IRRG img path
            image_path = os.path.join(self.root, self.split, 'image', image_fp)
            # Label path
            label_fp = image_fp.replace('IRRG', 'label_noBoundary')
            label_path = os.path.join(self.root, self.split, 'label', label_fp)

            self.image_list.append(image_path)
            self.label_list.append(label_path)
        print('Potadam dataset have {} images.'.format(len(self.image_list)))
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        
        image_path = self.image_list[index]
        label_path = self.label_list[index]
        name = os.path.basename(image_path)

        img = Image.open(image_path).convert('RGB')

        label = Image.open(label_path)

        if self.split == 'train':
            image,  label = self.RandomHorizontalFlip(img, label)
            # image,  label = self.RandomScaleCrop(image, label)
            image,  label = self.RandomGaussianBlur(image, label)
            image,  label = self.Normalize(image, label)
            image,  label = self.toTensor(image, label)
        else:
            image,  label = self.Normalize(img, label)
            image,  label = self.toTensor(image,  label)
        
        label = np.asarray(label, dtype=np.uint8)    
        label = self.encode_segmap(label).astype(np.uint8)  # numpy

        image = torch.cat([image], dim=0)
        label = torch.from_numpy(label).type(torch.LongTensor)
        
        return image, label, name

    def RandomHorizontalFlip(self, image, label):
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return image, label

    def RandomRotate(self, image, label):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        image = image.rotate(rotate_degree, Image.BILINEAR)
        label = label.rotate(rotate_degree, Image.NEAREST)
        return image, label

    def RandomGaussianBlur(self, image, label):
        radius = random.random()
        if radius < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image, label

    def RandomScaleCrop(self, image, label):
        short_size = random.randint(int(self.crop_size * 0.5), int(self.crop_size * 2.0))
        w, h = image.size
        
        oh = short_size
        ow = int(1.0 * w * oh / h)

        image = image.resize((ow, oh), Image.BILINEAR)
        label = label.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size <= self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            
            image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=0)
            label = ImageOps.expand(label, border=(0, 0, padw, padh), fill=0)

        # random crop crop_size
        w, h = image.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)

        image = image.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        label = label.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size)) 
        
        return image, label

    def Normalize(self, image, label, div_std=False):
        image = np.array(image).astype(np.float32)

        image /= 255
        image -= self.image_mean

        if div_std == True:
            image /= self.std

        return image, label

    def toTensor(self, image, label):
        image = np.array(image).astype(np.float32).transpose((2, 0, 1))
        image = torch.from_numpy(image).type(torch.FloatTensor)

        return image, label

    # get_ISPRS and encode_segmap generate label map
    def get_ISPRS(self):
        return np.asarray(
            [
              [255, 255, 255],  # 不透水面
              [  0,   0, 255],  # 建筑物
              [  0, 255, 255],  # 低植被
              [  0, 255,   0],  # 树
              [255, 255,   0],  # 车
              [255,   0,   0],  # Clutter/background
              [  0,   0,   0]   # ignore
            ]
        )
    
    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        
        for ii, label in enumerate(self.get_ISPRS()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        label_mask[label_mask == 6] = 255
        # plt.imshow(label_mask)
        # plt.title('Remote Sense')
        # pylab.show()
        return label_mask


if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import pylab

    def get_Potsdam_label():
        return np.asarray(
                            [
                            [255, 255, 255],  # 不透水面
                            [  0,   0, 255],  # 建筑物
                            [  0, 255, 255],  # 低植被
                            [  0, 255,   0],  # 树
                            [255, 255,   0],  # 车
                            [255,   0,   0],  # Clutter/background
                            [  0,   0,   0]   # ignore
                            ]
                            )

    def decode_segmap(label_mask):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
            the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
            in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        n_classes = 7
        label_colours = get_Potsdam_label()

        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)
        rgb[:, :, 0] = r 
        rgb[:, :, 1] = g 
        rgb[:, :, 2] = b 
        return rgb


    root = '/media/ssd/lsy_data/igarss/data/postdam'

    isprs = potsdamloader(root=root, split='train')
    print(len(isprs))
    dataloader = DataLoader(isprs, batch_size=1, shuffle=True, num_workers=2)
    print(len(dataloader))
    for image, label in dataloader:
        print('images type is {},  labels type is {}'.format(image.type(), label.type()))
        print('images size is {},  labels size is {}'.format(image.size(), label.size()))
        img = image.numpy()[0, 0:3, :, :].transpose((1, 2, 0))
        label = label.numpy().squeeze().astype(np.uint8)
        seg_pred_image = decode_segmap(label)
        # print(np.unique(label))
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(seg_pred_image)
        pylab.show()
        break

# rgb mean tensor([85.7115, 91.6557, 84.8835])
# rgb std tensor([26.2112, 26.2447, 27.1329])
# ir mean tensor([97.1133])
# ir std tensor([26.1144])
# nDSM mean tensor([46.3698])
# nDSM std tensor([26.1358])
