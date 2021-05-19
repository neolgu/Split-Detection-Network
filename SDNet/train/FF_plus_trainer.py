import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import lr_scheduler
import os
from argparse import ArgumentParser
from train.utils import get_config, get_model_list
from network.FF_plus_model import TransferModel
from data.transform import xception_data_transforms
from data.dataloader import ImageLoader, ImageLoaderGN
import tqdm

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/FF_plus.yaml',
                    help="training configuration")


class Trainer(nn.Module):
    def __init__(self, config, model_path=None):
        super(Trainer, self).__init__()
        self.config = get_config(config)
        self.model_path = model_path
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.batch_size = self.config["batch_size"]

        self.net = self.model_selection(self.config["model_name"], self.config["num_classes"])

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config['lr'],
                                          betas=(self.config['beta1'], self.config['beta2']), eps=1e-08)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

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

    def inference(self, continue_train=False):
        print(torch.cuda.is_available())
        if not os.path.exists(self.config["save_path"]):
            os.mkdir(self.config["save_path"])
        train_data = ImageLoader(self.config["train_data_path"], self.config["dataset_paths"], "train",
                                 xception_data_transforms)
        val_data = ImageLoader(self.config["train_data_path"], self.config["dataset_paths"], "val",
                               xception_data_transforms)
        # train_data = ImageLoaderGN(self.config["train_data_path"], self.config["dataset_paths"], "train",
        #                          xception_data_transforms)
        # val_data = ImageLoaderGN(self.config["train_data_path"], self.config["dataset_paths"], "val",
        #                        xception_data_transforms)
        train_dataset_size = len(train_data)
        val_dataset_size = len(val_data)

        train_dataloader = data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.config["num_workers"])
        val_dataloader = data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

        if continue_train:
            self.net.load_state_dict(torch.load(self.model_path))

        self.net = nn.DataParallel(self.net)
        best_model_wts = self.net.state_dict()

        best_acc = 0.0
        iteration = 0

        train_log = open(os.path.join(self.config["save_path"], "log.txt"), "w")
        train_log.write(self.config["dataset_name"])
        for epoch in range(self.config["epoch"]):
            print('Epoch {}/{}'.format(epoch + 1, self.config["epoch"]))
            print('-' * 10)
            self.net.train()
            train_loss = 0.0
            train_corrects = 0.0
            val_loss = 0.0
            val_corrects = 0.0

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
                if not (iteration % 5000):
                    print('\niteration {} train loss: {} Acc: {}'.format(iteration, iter_loss / self.batch_size,
                                                                         iter_corrects / self.batch_size))

            epoch_loss = train_loss / train_dataset_size
            epoch_acc = train_corrects / train_dataset_size
            print('\n{} epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

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
                print('epoch val loss: {} Acc: {}'.format(epoch_loss, epoch_acc))
                train_log.write("{} epoch loss: {}, Acc: {}\n\n".format(epoch, epoch_loss, epoch_acc))
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = self.net.state_dict()
            self.scheduler.step()
            torch.save(self.net.module.state_dict(), os.path.join(self.config["save_path"], "{}.tar".format(epoch)))
            print("saved at {}".format(os.path.join(self.config["save_path"], "{}.tar".format(epoch))))

        print("Best val Acc: {:.4f}".format(best_acc))
        self.net.load_state_dict(best_model_wts)
        torch.save(self.net.module.state_dict(), os.path.join(self.config["save_path"], "best_acc.tar"))
        train_log.close()
