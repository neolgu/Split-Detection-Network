from train.train import Trainer
from tester.tester import Tester

gan_path = 'checkpoint/GAN_train/5.tar'
n_gan_path = 'checkpoint/N_GAN_train/7.tar'

if __name__ == '__main__':
    # trainer = Trainer("config/config.yaml")
    # trainer.train(continue_train=False)
    print("noise")
    tester = Tester("config/config.yaml", "checkpoint/conf_train/6.tar")
    tester.test()
    ###########################################
    # trainer = Trainer("config/config.yaml", gan_path=gan_path, n_gan_path=n_gan_path)
    # trainer.train(continue_train=False)
