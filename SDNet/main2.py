import torch
import torch.nn as nn
import torch.utils.data as data
import os
from data.transform import xception_data_transforms
from data.dataloader import ImageLoader, TestImageLoader
from network.FF_plus_model import TransferModel
from data.transform import xception_data_transforms
import tqdm
from tester.tester import Tester

def model_selection(model_name, num_classes, dropout=None):
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


def test(dt_path, dataset, model_path):
    print(torch.cuda.is_available())
    test_data = ImageLoader(dt_path, dataset, "test", xception_data_transforms)
    test_dataset_size = len(test_data)

    test_dataloader = data.DataLoader(test_data, batch_size=1, shuffle=True)

    net = model_selection("xception", num_classes=2)

    net.load_state_dict(torch.load(model_path))

    net = nn.DataParallel(net)

    corrects = 0
    acc = 0.0

    confusion_matrix = [[0, 0], [0, 0]]

    net.eval()
    with torch.no_grad():
        i = 0
        for (images, labels) in tqdm.tqdm(test_dataloader):
            images = images.to(0)
            labels = labels.to(0)
            outputs = net(images).squeeze(1)

            _, preds = torch.max(outputs.data, 1)

            for i in range(len(labels)):
                real_class = int(labels[i])
                pred_class = int(preds[i] > 0.5)
                confusion_matrix[real_class][pred_class] += 1

            corrects += torch.sum(preds == labels.data).to(torch.float32)
            i += 1
        acc = corrects / test_dataset_size

        positive = confusion_matrix[0][0] + confusion_matrix[1][1]
        negative = confusion_matrix[1][0] + confusion_matrix[0][1]
        accuracy = positive / (positive + negative)
        print("Fake_Fake: {}\nReal_Real: {}\nReal_Fake: {}\nFake_Real: {}".
              format(confusion_matrix[0][0], confusion_matrix[1][1], confusion_matrix[1][0], confusion_matrix[0][1]))
        print('Test Acc: {:.4f}'.format(acc))


if __name__ == '__main__':
    data_path = "/home/yoon/DF_dataset"
    G_model_path = "checkpoint/GAN_train/5.tar"
    N_model_path = "checkpoint/N_GAN_train/7.tar"
    A_model_path = "checkpoint/ALL_train/7.tar"
    dataset_paths_G = {'GAN': 'GAN', 'Original': 'Original'}
    dataset_paths_N = {'N_GAN': 'N_GAN', 'Original': 'Original'}
    dataset_paths_A = {'GAN': 'GAN', 'N_GAN': 'N_GAN', 'Original': 'Original'}

    # test(data_path, dataset_paths_G, G_model_path)
    # test(data_path, dataset_paths_N, G_model_path)
    # test(data_path, dataset_paths_G, N_model_path)
    # test(data_path, dataset_paths_N, N_model_path)
    # test(data_path, dataset_paths_G, A_model_path)
    # test(data_path, dataset_paths_N, A_model_path)

    # test(data_path, dataset_paths_A, A_model_path)
    # test(data_path, dataset_paths_A, A_model_path)
    # test(data_path, dataset_paths, model_path)

    # for i in range(9):
    #     test(data_path, dataset_paths_N, "checkpoint/N_GAN_train/{}.tar".format(i))

    print("JPEG")
    tester = Tester("config/config.yaml", "checkpoint/conf_train/6.tar")
    tester.test()