import torch.nn as nn
import torch.nn.functional as F
import torch

from model.FF_plus_model import TransferModel


class SDNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SDNet, self).__init__()
        self.num_classes = num_classes
        # Gan / Non-GAN Discriminator
        self.conf = TransferModel(modelchoice='xception', num_out_classes=num_classes)
        # GAN Detector
        self.Gan = TransferModel(modelchoice='xception', num_out_classes=num_classes)
        # Non-GAN Detector
        self.nonGan = TransferModel(modelchoice='xception', num_out_classes=num_classes)

    def forward(self, x):
        y0 = self.conf.forward(x)
        with torch.no_grad():
            y1 = self.nonGan.forward(x)
            y2 = self.Gan.forward(x)

        weight = torch.sigmoid(y0)
        y1 = torch.multiply(weight[:, 0].reshape(-1, 1), y1)
        y2 = torch.multiply(weight[:, 1].reshape(-1, 1), y2)

        return y1 + y2

    def load_subnet(self, gan_path, n_gan_path):
        print("Load GAN Detector. Path={}".format(gan_path))
        self.Gan.load_state_dict(torch.load(gan_path))
        print("Load Non-GAN Detector. Path={}".format(gan_path))
        self.nonGan.load_state_dict(torch.load(n_gan_path))

        # freeze GAN, Non-GAN Detector
        self.Gan.eval()
        self.nonGan.eval()
