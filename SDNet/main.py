from train.FF_plus_trainer import Trainer
from tester.tester import Tester

if __name__ == '__main__':
	# trainer = Trainer("config/FF_plus.yaml")
	# trainer.inference()
	# for i in range(9):
	# 	tester = Tester("config/FF_plus.yaml", "checkpoint/GAN_train/{}.tar".format(i))
	# 	tester.test()
	print("raw")
	tester = Tester("config/config.yaml", "checkpoint/GAN_train/8.tar")
	tester.test()
