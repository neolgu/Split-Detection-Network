from train.FF_plus_trainer import Trainer
# from tester.conf_tester import Tester
from tester.tester import Tester


if __name__ == '__main__':
    # trainer = Trainer("config/GN_model.yaml")
    # trainer.inference()

    # data_path = "/home/yoon/DF_dataset_75"
    G_model_path = "checkpoint/GAN_train/5.tar"
    N_model_path = "checkpoint/N_GAN_train/7.tar"
    # A_model_path = "checkpoint/ALL_train/7.tar"
    A_model_path = "checkpoint/ALL_train/6.tar"
    # GN_model_path = "checkpoint/GN_train/6.tar"
    GN_model_path = "checkpoint/conf/3.tar"
    # 8epoch
    # tester = Tester("config/GN_model.yaml")
    # # tester.test(GN_model_path, A_model_path)
    # for i in range(9):
    #     print("{} test".format(i))
    #     tester.test(GN_model_path, "checkpoint/ALL_train/{}.tar".format(i))
    print("blur")
    tester = Tester("config/config.yaml", "checkpoint/conf_train/6.tar")
    tester.test()
