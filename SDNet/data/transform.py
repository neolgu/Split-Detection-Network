from torchvision import transforms
from PIL import ImageFilter
from PIL import Image
import numpy as np
import cv2
from torch.autograd import Variable
import torch


def gaussian_blur(image):
    kernel_size = 11
    img = transforms.ToPILImage()(image).convert("RGB")
    img = np.array(img)
    img = cv2.GaussianBlur(np.array(img), (kernel_size, kernel_size), 1.5)
    img = Image.fromarray(img.astype(np.uint8))
    return transforms.ToTensor()(img)


def gaussian_noise(ins):
    stddev = 5/255.
    noise = Variable(torch.zeros(ins.size()))
    ins = ins + noise.data.normal_(0, stddev)
    return torch.clamp(ins, 0, 1)


xception_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        # transforms.Lambda(gaussian_blur),
        # transforms.Lambda(gaussian_noise),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}

resnet18_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
