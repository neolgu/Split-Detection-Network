from train_test.train import Trainer
from train_test.test import Tester
from argparse import ArgumentParser
from train_test.utils import get_config

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/config.yaml',
                    help="training configuration")

if __name__ == '__main__':
    config = get_config("config/config.yaml")
    if config['mode'] == "train":
        trainer = Trainer("config/config.yaml")
        trainer.train()
    elif config['mode'] == "test":
        tester = Tester("config/config.yaml")
        tester.test()
    else:
        print("Invalid Mode. Check config file")
