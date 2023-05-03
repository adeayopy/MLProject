from __future__ import division, print_function, absolute_import

import os
import re
import pdb
import glob
import pickle

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL.Image as PILI
import numpy as np

from tqdm import tqdm


class EpisodeDataset(data.Dataset):

    def __init__(self, root, phase='train', n_shot=5, n_eval=15, transform=None):
        """Args:
            root (str): path to data
            phase (str): train, val or test
            n_shot (int): how many examples per class for training (k/n_support)
            n_eval (int): how many examples per class for evaluation
                - n_shot + n_eval = batch_size for data.DataLoader of ClassDataset
            transform (torchvision.transforms): data augmentation
        """
        root = os.path.join(root, phase)
        #return list of all folders present. Class names
        self.labels = sorted(os.listdir(root))
        images = [glob.glob(os.path.join(root, label, '*')) for label in self.labels]

        self.episode_loader = [data.DataLoader(
            ClassDataset(images=images[idx], label=idx, transform=transform),
            batch_size=n_shot+n_eval, shuffle=True, num_workers=0) for idx, _ in enumerate(self.labels)]

    def __getitem__(self, idx):
        return next(iter(self.episode_loader[idx]))

    def __len__(self):
        return len(self.labels)


class ClassDataset(data.Dataset):

    def __init__(self, images, label, transform=None):
        """
            images (list of str): each item is a path to an image of the same label
            label (int): the label of all the images
        """
        self.images = images
        self.label = label
        self.transform = transform

    def __getitem__(self, idx):
        image = PILI.open(self.images[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, self.label

    def __len__(self):
        return len(self.images)


class EpisodicSampler(data.Sampler):

    def __init__(self, total_classes, n_class, n_episode):
        self.total_classes = total_classes
        self.n_class = n_class
        self.n_episode = n_episode

    def __iter__(self):
        for i in range(self.n_episode):
            yield torch.randperm(self.total_classes)[:self.n_class]

    def __len__(self):
        return self.n_episode


def prepare_data(data_root, n_shot, n_eval, image_size, n_workers,pin_mem, episode, episode_val, n_class):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #Load data and apply transformation
    train_set = EpisodeDataset(data_root, 'train', n_shot, n_eval,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize]))

    val_set = EpisodeDataset(data_root, 'val', n_shot, n_eval,
        transform=transforms.Compose([
            transforms.Resize(image_size * 8 // 7),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize]))

    test_set = EpisodeDataset(data_root, 'test', n_shot, n_eval,
        transform=transforms.Compose([
            transforms.Resize(image_size * 8 // 7),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize]))

    train_loader = data.DataLoader(train_set, num_workers=n_workers, pin_memory=pin_mem,
        batch_sampler=EpisodicSampler(len(train_set), n_class, episode))

    val_loader = data.DataLoader(val_set, num_workers=2, pin_memory=False,
        batch_sampler=EpisodicSampler(len(val_set), n_class, episode_val))

    test_loader = data.DataLoader(test_set, num_workers=2, pin_memory=False,
        batch_sampler=EpisodicSampler(len(test_set), n_class, episode_val))

    return train_loader, val_loader, test_loader
