import os

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import lr_scheduler
from argparse import ArgumentParser
import tqdm
from train_test.utils import get_config, model_selection
from data.transform import xception_data_transforms, resnet18_data_transforms
from data.dataset import ImageLoader, NewImageLoader

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/config.yaml',
                    help="training configuration")


class Trainer:
    def __init__(self, config):
        self.config = get_config(config)
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.batch_size = self.config['batch_size']
        self.model_path = self.config['resume_path']
        self.resume = self.config['resume']

        self.net = model_selection(self.config["model_name"], self.config["num_classes"],
                                   self.config["gan_path"], self.config["n_gan_path"], resume=self.resume)
        # optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config['lr'],
                                          betas=(self.config['beta1'], self.config['beta2']), eps=1e-08)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        if self.use_cuda:
            self.net.to(self.device_ids[0])

    def train(self):
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])

        if not os.path.exists(self.config["save_path"]):
            os.mkdir(self.config["save_path"])

        train_data = ImageLoader(self.config["data_path"], self.config["dataset_paths"], "train", xception_data_transforms)
        val_data = ImageLoader(self.config["data_path"], self.config["dataset_paths"], "val", xception_data_transforms)

        train_dataset_size = len(train_data)
        val_dataset_size = len(val_data)

        train_dataloader = data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_dataloader = data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

        if self.resume:
            print("Continue Train")
            print("Continue Model : {}".format(self.model_path))
            self.net.load_state_dict(torch.load(self.model_path))

        self.net = nn.DataParallel(self.net)

        iteration = 0
        best_acc = 0.0

        for epoch in range(self.config["epoch"]):
            print('Epoch {}/{}'.format(epoch + 1, self.config["epoch"]))
            print('-' * 10)

            self.net.train()

            train_loss = 0.0
            train_corrects = 0.0

            for (images, labels) in tqdm.tqdm(train_dataloader):
                if self.use_cuda:
                    images = images.to(self.device_ids[0])
                    labels = labels.to(self.device_ids[0])
                self.optimizer.zero_grad()
                outputs = self.net(images).squeeze(1)
                _, preds = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                iter_loss = loss.data.item()
                train_loss += iter_loss
                iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
                train_corrects += iter_corrects
                iteration += 1
                if not (iteration % self.config['print_iter']):
                    iteration = 0
                    print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(
                        iteration, iter_loss / self.batch_size, iter_corrects / self.batch_size))

            epoch_loss = train_loss / train_dataset_size
            epoch_acc = train_corrects / train_dataset_size
            print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            self.scheduler.step()

            torch.save(self.net.module.state_dict(), os.path.join(self.config["save_path"], "{}.tar".format(epoch)))
            print("saved at {}".format(os.path.join(self.config["save_path"], "{}.tar".format(epoch))))

            # validation part

            val_loss = 0.0
            val_corrects = 0.0

            self.net.eval()
            with torch.no_grad():
                for (images, labels) in tqdm.tqdm(val_dataloader):
                    if self.use_cuda:
                        images = images.to(self.device_ids[0])
                        labels = labels.to(self.device_ids[0])
                    outputs = self.net(images).squeeze(1)
                    _, preds = torch.max(outputs.data, 1)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.data.item()
                    val_corrects += torch.sum(preds == labels.data).to(torch.float32)
                epoch_loss = val_loss / val_dataset_size
                epoch_acc = val_corrects / val_dataset_size
                print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = self.net.state_dict()
                f = open(self.config["save_path"] + "\\log.txt", "a")
                f.write('epoch {} val loss: {:.4f} Acc: {:.4f}\n'.format(epoch + 1, epoch_loss, epoch_acc))
                f.close()
        print("Best val Acc: {:.4f}".format(best_acc))
        self.net.load_state_dict(best_model_wts)
        torch.save(self.net.module.state_dict(), os.path.join(self.config["save_path"], "best_acc.tar"))
