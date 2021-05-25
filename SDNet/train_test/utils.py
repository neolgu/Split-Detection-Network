import os
import yaml

import torch.nn as nn
from torchvision.models import resnet18
from model.FF_plus_model import TransferModel
from model.SDNet import SDNet


# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


# Get model list for resume
def get_model_list(dirname, key, iteration=0):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    if iteration == 0:
        last_model_name = gen_models[-1]
    else:
        for model_name in gen_models:
            if '{:0>8d}'.format(iteration) in model_name:
                return model_name
        raise ValueError('Not found models with this iteration')
    return last_model_name


def model_selection(model_name, num_classes, gan_path=None, n_gan_path=None, resume=False):
    print("Model: ", model_name)
    if model_name == 'xception':
        return TransferModel(modelchoice='xception', num_out_classes=num_classes)
    elif model_name == 'resnet18':
        model = resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
    elif model_name == 'conf':
        model = SDNet(num_classes=num_classes)
        if not resume:
            model.load_subnet(gan_path=gan_path, n_gan_path=n_gan_path)
        return model
    else:
        raise NotImplementedError(model_name)
