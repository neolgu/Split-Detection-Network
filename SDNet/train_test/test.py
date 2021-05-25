import os

import torch
import torch.nn as nn
import torch.utils.data as data
from argparse import ArgumentParser
import tqdm

from train_test.utils import get_config, model_selection
from data.transform import xception_data_transforms, resnet18_data_transforms
from data.dataset import ImageLoader, NewImageLoader

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/config.yaml',
                    help="testing configuration")


class Tester:
    def __init__(self, config):
        self.config = get_config(config)
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.batch_size = self.config["batch_size"]
        self.model_path = self.config["test_path"]

        self.net = model_selection(self.config["model_name"], self.config["num_classes"], resume=True)

        if self.use_cuda:
            self.net.to(self.device_ids[0])

    def test(self):
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])

        test_data = ImageLoader(self.config["data_path"], self.config["dataset_paths"], "test", xception_data_transforms)
        test_dataset_size = len(test_data)
        test_dataloader = data.DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

        print("Test Model Path: ", self.model_path)
        self.net.load_state_dict(torch.load(self.model_path))
        self.net = nn.DataParallel(self.net)

        corrects = 0
        acc = 0.0
        confusion_matrix = [[0, 0], [0, 0]]

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
            print('Test Acc: {:.4f}'.format(acc))

            print("Confusion Matrix\nFake_Fake: {}\nReal_Real: {}\nReal_Fake: {}\nFake_Real: {}".
                  format(confusion_matrix[0][0], confusion_matrix[1][1],
                         confusion_matrix[1][0], confusion_matrix[0][1]))
