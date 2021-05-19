from PIL import Image
from data.transform import xception_data_transforms
from data.utils import *
import random
import torch.utils.data as data
import os
import torch
import matplotlib.pyplot as plt


class ImageLoader(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, root_path, path_list, mode="train", transform=None):
        self.transform = transform
        self.mode = mode
        self.face_detector = FaceDetector()

        _len = {"train": 5000*7, "val": 5000, "test": 5000*2}
        f_len = {"train": 2500*7, "val": 2500, "test": 5000}
        o_len = {"train": 70000, "val": 10000, "test": 20000}
        # _len = {"train": 18000, "val": 2000, "test": 600}

        self.file = []

        for path in path_list:
            data_path = os.path.join(root_path, path, mode)
            print(data_path)
            image_path = os.listdir(data_path)
            random.shuffle(image_path)
            if path == "Original":
                for image in image_path[:_len[mode]]:
                    img = os.path.join(data_path, image)
                    self.file.append([img, 1])
            else:
                for image in image_path[:_len[mode]]:
                    img = os.path.join(data_path, image)
                    self.file.append([img, 0])

    def __len__(self):
        return len(self.file)

    def read_image(self, index, use_transform=xception_data_transforms):
        image = Image.open(self.file[index][0])

        if use_transform:
            image = use_transform[self.mode](image)

        return image

    def __getitem__(self, index):
        X = self.read_image(index, self.transform)  # (input) spatial image
        y = self.file[index][1]

        return X, y


class TestImageLoader(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, root_path, mode, transform=None):
        self.transform = transform
        self.mode = mode
        self.file = []

        path_list = os.listdir(root_path)
        for path in path_list:
            if path.startswith("origin"):
                y = 1
            else:
                y = 0
            self.file.append([os.path.join(root_path, path), y])

    def __len__(self):
        return len(self.file)

    def read_image(self, index):
        image = Image.open(self.file[index][0])

        if self.transform:
            image = self.transform(image)

        return image

    def __getitem__(self, index):
        X = self.read_image(index)  # (input) spatial image
        y = self.file[index][1]
        return X, y


class ImageLoaderGN(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, root_path, path_list, mode="train", transform=None):
        self.transform = transform
        self.mode = mode
        self.face_detector = FaceDetector()

        _len = {"train": 5000*7, "val": 5000, "test": 5000*2}
        o_len = {"train": 70000, "val": 10000, "test": 20000}
        # _len = {"train": 18000, "val": 2000, "test": 100}

        self.file = []

        for path in path_list:
            data_path = os.path.join(root_path, path, mode)
            print(data_path)
            image_path = os.listdir(data_path)
            random.shuffle(image_path)
            if path == "GAN":
                for image in image_path[:_len[mode]]:
                    img = os.path.join(data_path, image)
                    self.file.append([img, 1])
            else:
                for image in image_path[:_len[mode]]:
                    img = os.path.join(data_path, image)
                    self.file.append([img, 0])

    def __len__(self):
        return len(self.file)

    def read_image(self, index, use_transform=xception_data_transforms):
        image = Image.open(self.file[index][0])

        if use_transform:
            image = use_transform[self.mode](image)

        return image

    def __getitem__(self, index):
        X = self.read_image(index, self.transform)  # (input) spatial image
        y = self.file[index][1]

        return X, y


class NewImageLoader(data.Dataset):
    def __init__(self, root_path, path_list, mode="train", transform=None):
        self.transform = transform
        self.mode = mode

        _len = {"train": 5000 * 7, "val": 5000, "test": 5000 * 2}
        # _len = {"train": 10 * 7, "val": 10, "test": 10 * 2}

        self.file = []

        for path in path_list:
            data_path = os.path.join(root_path, path, mode)
            image_path = os.listdir(data_path)
            random.shuffle(image_path)
            for image in image_path[:_len[mode]]:
                img = os.path.join(data_path, image)
                if path == "GAN":
                    self.file.append([img, 1])
                else:
                    self.file.append([img, 0])

    def __len__(self):
        return len(self.file)

    def read_image(self, index, use_transform=None):
        image = Image.open(self.file[index][0])

        if use_transform:
            image = use_transform[self.mode](image)

        return image

    def __getitem__(self, index):
        X = self.read_image(index, self.transform)  # (input) spatial image
        y = self.file[index][1]

        return X, y
