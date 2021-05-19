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

    def test(self, GN_model, A_model):
        print(torch.cuda.is_available())
        test_data = ImageLoader(self.config["train_data_path"], self.config["dataset_paths"], "test", xception_data_transforms)
        test_dataset_size = len(test_data)

        test_data_loader = data.DataLoader(test_data, batch_size=1, shuffle=True)

        # model_GN = self.load_net(GN_model)
        model_A = self.load_net(A_model)

        # model_GN.eval()
        model_A.eval()

        A_corrects = 0
        GN_corrects = 0
        # confusion_real_GN = [[0, 0], [0, 0]]
        confusion_real_ALL= [[0, 0], [0, 0]]
        with torch.no_grad():
            for (images, labels) in tqdm.tqdm(test_data_loader):
                if self.use_cuda:
                    images = images.to(self.device_ids[0])
                    labels = labels.to(self.device_ids[0])
                # outputs = model_GN(images)
                A_output = model_A(images)

                _, A_preds = torch.max(A_output.data, 1)
                # _, GN_preds = torch.max(outputs.data, 1)

                for i in range(len(labels)):
                    real_class = int(labels[i])
                    # GN_pred_class = int(GN_preds[i] > 0.5)
                    A_pred_class = int(A_preds[i] > 0.5)

                    confusion_real_ALL[real_class][A_pred_class] += 1
                    # confusion_real_GN[real_class][GN_preds] += 1

                A_corrects += torch.sum(A_pred_class == labels.data).to(torch.float32)
                # GN_corrects += torch.sum(GN_pred_class == labels.data).to(torch.float32)

            acc_A = A_corrects / test_dataset_size
            # acc_GN = GN_corrects / test_dataset_size

            # print("All data accuracy: {}\nGN data accuracy: {}\n\n".format(acc_A, acc_GN))
            print("All data accuracy: {}\n".format(acc_A))
            # print("GN_confusion", confusion_real_GN)
            print("ALL_confusion", confusion_real_ALL)
