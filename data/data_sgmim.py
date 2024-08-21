# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import math
import random
import numpy as np
import cv2
import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import functional
import torchvision.transforms.functional as f
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDepthDataset(Dataset):
    def __init__(self, image_paths, target_paths, config, train=True):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.input_size = config.DATA.IMG_SIZE

        if config.MODEL.TYPE == 'swin':
            model_patch_size=config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size=config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError

        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )
    def transform(self, image, target):
        #ndarray to img
        image=Image.fromarray(image)
        target=Image.fromarray(target)
        #RandomResizedCrop
        trans1=RandomResizedCrop_pair(self.input_size,  scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.))
        image,target=trans1(image, target)

        #RandomHorizontalFlip
        trans2 = RandomHorizontalFlip_pair(0.5)
        image,target = trans2(image, target)

        #ToTensor
        trans3 = ToTensor_pair()
        image, target = trans3(image, target)
        target = target.unsqueeze(0)

        #Normalize only to image
        trans4 = transforms.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD))
        image = trans4(image)

        return image, target

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        depth = Image.open(self.target_paths[index])
        depth = torch.Tensor(np.array(depth) / 2 ** 16)
        depth = np.array(depth)
        x, y = self.transform(image, depth)
        mask = self.mask_generator()
        return x, y, mask

    def __len__(self):
        return len(self.image_paths)

#reference from https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
class RandomResizedCrop_pair:
    def __init__(self, size,  scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.),interpolation=f.InterpolationMode.BILINEAR):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, image, depth):
        params = transforms.RandomResizedCrop.get_params(image, self.scale,self.ratio)
        image = functional.resized_crop(image, *params, (self.size,self.size), interpolation=self.interpolation)
        depth = functional.resized_crop(depth, *params, (self.size,self.size), interpolation=self.interpolation)
        return image, depth

class RandomHorizontalFlip_pair:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, depth):
        if random.random() < self.flip_prob:
            image = functional.hflip(image)
            depth = functional.hflip(depth)
        return image, depth
class Normalize_pair:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, image, depth):
        image = functional.normalize(image, mean=self.mean, std=self.std)
        return image, depth
class ToTensor_pair:
    def __call__(self, image, depth):
        image = functional.to_tensor(image)
        depth = torch.as_tensor(np.array(depth), dtype=torch.float32)
        return image, depth
class Compose_pair:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, depth):
        for t in self.transforms:
            image, depth = t(image, depth)
        return image, depth


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask


class SGMIMTransform:
    def __init__(self, config):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
 
        if config.MODEL.TYPE == 'swin':
            model_patch_size=config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size=config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError
        
        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return img, mask


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_sgmim(config, logger):
    f = open(config.DATA.FILE_NAME_PATH, 'r')
    lines = f.readlines()
    image_file_paths = list(map(lambda s: config.DATA.DATA_PATH + "/" + s.split(".")[0] + ".JPEG", lines))
    target_file_paths = list(map(lambda s: config.DATA.DEPTH_DATA_PATH + "/" + s.split(".")[0] + ".JPX", lines))
    dataset = ImageDepthDataset(image_file_paths, target_file_paths, config)
    logger.info(f'Build dataset: train images = {len(dataset)}')
    
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    return dataloader