import torch
import torch.nn as nn
import torch.utils.data as data
import os
from argparse import ArgumentParser
from train.utils import get_config, get_model_list
from network.FF_plus_model import TransferModel
from data.transform import xception_data_transforms
from data.dataloader import ImageLoader, TestImageLoader
import torch.nn.functional as F
import tqdm
from torchvision.utils import save_image

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/FF_plus.yaml',
                    help="testing configuration")


class Tester(nn.Module):
    def __init__(self, config, model_path=None):
        super(Tester, self).__init__()
        self.config = get_config(config)
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']

    def model_selection(self, model_name, num_classes, dropout=None):
        if model_name == 'xception':
            return TransferModel(modelchoice='xception',
                                 num_out_classes=num_classes)
        elif model_name == 'resnet18':
            return TransferModel(modelchoice='resnet18', dropout=dropout,
                                 num_out_classes=num_classes)
        elif model_name == 'xception_concat':
            return TransferModel(modelchoice='xception_concat',
                                 num_out_classes=num_classes)
        else:
            raise NotImplementedError(model_name)

    def load_net(self, model_path):
        net = self.model_selection(self.config["model_name"], self.config["num_classes"])
        if self.use_cuda:
            net.to(self.device_ids[0])
        net.load_state_dict(torch.load(model_path))
        net = nn.DataParallel(net)
        return net

    def test(self, classification_GN, GAN_model, N_GAN_model, A_model):
        print(torch.cuda.is_available())
        test_data = ImageLoader(self.config["train_data_path"], self.config["dataset_paths"], "test", xception_data_transforms)
        test_dataset_size = len(test_data)

        test_data_loader = data.DataLoader(test_data, batch_size=1, shuffle=True)

        model_GN = self.load_net(classification_GN)
        model_GAN = self.load_net(GAN_model)
        model_NGAN = self.load_net(N_GAN_model)
        model_A = self.load_net(A_model)

        model_GN.eval()
        model_GAN.eval()
        model_NGAN.eval()
        model_A.eval()

        A_corrects = 0
        GN_corrects = 0
        confusion_real_GN = [[0, 0], [0, 0]]
        confusion_real_ALL= [[0, 0], [0, 0]]
        confusion_GAN = [[0, 0], [0, 0]]
        confusion_NGAN = [[0, 0], [0, 0]]
        with torch.no_grad():
            for (images, labels) in tqdm.tqdm(test_data_loader):
            # for (images, labels) in test_data_loader:
                if self.use_cuda:
                    images = images.to(self.device_ids[0])
                    labels = labels.to(self.device_ids[0])
                outputs = model_GN(images)
                GAN_output = model_GAN(images)
                NGAN_output = model_NGAN(images)
                A_output = model_A(images)

                _, A_preds = torch.max(A_output.data, 1)
                _, GN_preds = torch.max(outputs.data, 1)
                _, GAN_pred = torch.max(GAN_output.data, 1)
                _, NGAN_pred = torch.max(NGAN_output.data, 1)

                for i in range(len(labels)):
                    real_class = int(labels[i])
                    GN_class = int(GN_preds[i] > 0.5)
                    A_pred_class = int(A_preds[i] > 0.5)

                    confusion_real_ALL[real_class][A_pred_class] += 1
                    confusion_real_GN[real_class][GN_class] += 1

                    if GN_class == 1:
                        output_class = int(GAN_pred[i] > 0.5)
                        confusion_GAN[real_class][output_class] += 1
                    else:
                        output_class= int(NGAN_pred[i] > 0.5)
                        confusion_NGAN[real_class][output_class] += 1

                # GAN 1, NGAN 0
                """
                confidence = F.softmax(outputs, dim=1).data[0]
                GAN_real_conf = F.softmax(GAN_output, dim=1).data[0][1]
                NGAN_real_conf = F.softmax(NGAN_output, dim=1).data[0][1]

                custom_pred = (confidence[0] * NGAN_real_conf) + (confidence[1] * GAN_real_conf)
                custom_pred_class = int(custom_pred > 0.5)
                """

                A_corrects += torch.sum(A_pred_class == labels.data).to(torch.float32)
                GN_corrects += torch.sum(output_class == labels.data).to(torch.float32)

            acc_A = A_corrects / test_dataset_size
            acc_GN = GN_corrects / test_dataset_size

            print("All data accuracy: {}\nGN data accuracy: {}\n\n".format(acc_A, acc_GN))
            print("GAN data accuracy: {}\nNGAN data accuracy: {}\n\n".format(confusion_GAN, confusion_NGAN))
            print("GN_confusion", confusion_real_GN)
            print("ALL_confusion", confusion_real_ALL)
            print("real, NGAN: {}\nreal, GAN: {}".format(confusion_real_GN[1][0], confusion_real_GN[1][1]))
