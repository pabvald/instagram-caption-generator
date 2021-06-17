#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import collections

from os.path import join as pjoin
from torch.utils import data
from torchvision import transforms
from PIL import Image 

class instagramDataset(data.Dataset):
    """ Data loader for the Instagram Captions dataset 
    
    A total of 4 splits are provided by this dataset:
        train: 60% of the original dataset
        val: 20% of the original dataset 
        test: 20% of the original dataset
        trainval: 80% of the original dataset
    """

    def __init__(self, root, split="train", is_transform=True, img_size=512):

        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.captions = collections.defaultdict(list)
        self.serial_numbers = collections.defaultdict(list)
        self.files = collections.defaultdict(list)

        for split in ["train", "val", "test", "trainval"]:
            path = pjoin(self.root, split + ".csv")
            dataset = pd.read_csv(path)
            file_list = dataset['Image File'].tolist()
            serial_numbers = dataset['Sr No'].tolist()
            captions = dataset['Caption'].tolist()

            self.files[split] = file_list 
            self.serial_numbers[split] = serial_numbers
            self.captions[split] = captions

        self.img_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.files[self.split])
    
    def __getitem__(self, index):
        caption = self.captions[self.split][index]
        sr_no = self.serial_numbers[self.split][index]
        img_name = self.files[self.split][index]
        img_path = pjoin(self.root, img_name + ".jpg")
        img = Image.open(img_path)

        if self.is_transform:
            img, caption = self.transform(img, caption)

        return img, caption, sr_no

    def get_by_serial(self, serial_number):
        sr_nos = self.serial_numbers[self.split]
        index = sr_nos.index(serial_number)
        return self.__getitem__(index)

    def transform(self, img, caption):
        if self.img_size == ("same", "same"):
            pass 
        else: 
            img = img.resize((self.img_size[0], self.img_size[1]))

        # transform image
        img = self.img_tf(img)

        # transform captions

        return img, caption