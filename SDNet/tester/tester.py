import torch
import torch.nn as nn
import torch.utils.data as data
import os
from argparse import ArgumentParser
from train.utils import get_config, get_model_list
from network.FF_plus_model import TransferModel
from data.transform import xception_data_transforms
from data.dataloader import ImageLoader, TestImageLoader
import tqdm
from torchvision.utils import save_image

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/FF_plus.yaml',
                    help="testing configuration")


class Tester(nn.Module):
    def __init__(self, config, model_path=None):
        super(Tester, self).__init__()
        self.config = get_config(config)
        self.model_path = model_path
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.batch_size = self.config["batch_size"]

        self.net = self.model_selection(self.config["model_name"], self.config["num_classes"])

        if self.use_cuda:
            self.net.to(self.device_ids[0])

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

    def test(self):
        print(torch.cuda.is_available())
        if not os.path.exists(self.config["save_path"]):
            os.mkdir(self.config["save_path"])
        test_data = ImageLoader(self.config["train_data_path"], self.config["dataset_paths"], "test", xception_data_transforms)
        test_dataset_size = len(test_data)

        test_dataloader = data.DataLoader(test_data, batch_size=1, shuffle=True)

        self.net.load_state_dict(torch.load(self.model_path))

        self.net = nn.DataParallel(self.net)

        corrects = 0
        acc = 0.0
        save_path = "./results/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        confusion_matrix = [[0,0], [0,0]]

        self.net.eval()
        with torch.no_grad():
            for (images, labels) in tqdm.tqdm(test_dataloader):
                if self.use_cuda:
                    images = images.to(self.device_ids[0])
                    labels = labels.to(self.device_ids[0])
                outputs = self.net(images).squeeze(1)

                _, preds = torch.max(outputs.data, 1)

                for i in range(len(labels)):
                    # print(labels[i], outputs[i], preds)
                    real_class = int(labels[i])
                    pred_class = int(preds[i] > 0.5)
                    confusion_matrix[real_class][pred_class] += 1

                corrects += torch.sum(preds == labels.data).to(torch.float32)
            acc = corrects / test_dataset_size

            positive = confusion_matrix[0][0] + confusion_matrix[1][1]
            negative = confusion_matrix[1][0] + confusion_matrix[0][1]
            accuracy = positive / (positive + negative)
            print("Fake_Fake: {}\nReal_Real: {}\nReal_Fake: {}\nFake_Real: {}".
                  format(confusion_matrix[0][0], confusion_matrix[1][1], confusion_matrix[1][0], confusion_matrix[0][1]))
            print('Test Acc: {:.4f}'.format(acc))
