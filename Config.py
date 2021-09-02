# Usage:	configuration of the whole run
# Date:		AUG 23 2021
# Author:	Jun ZHU
# Email:	Jun__Zhu@outlook.com


import os

from datetime import date


class HyperParams():
	"""hyperparameters of the model"""

	def __init__(self):
		# ["cnn", "resn1", "resn2"]
		self.model = "cnn"
#		self.model = "resn1"
#		self.model = "resn2"
		# exclusive hyperparameters for CNN--------------
		self.dropout = .3
		self.lr = 0.001
		# exclusive hyperparameters for ResNet-----------
		self.num_blocks = 5
		# common (hyper)parameters for CNN & ResNet-------------
		self.epoch = 100
		self.batch_size = 256


class Config():
	"""configuration of the whole run"""

	def __init__(self, IfTransferLearning=False):
		# buffer size and multiprocessing 
		self.queue_size = 50
		self.workers = 8
		self.multiprocessing = True
		# logging level
		self.verbose = 1
		# split ratios of training, validation, test datasets
		self.split_ratio_TVT = (80, 5, 15)
		# fold numbers for categorical events (augmentation)
		self.TypeFold = ({"quake":10, "blast":1} if IfTransferLearning else {"quake":1, "blast":4})
		# sample rate of tpical instruments: 100 Hz for HHZ; 40 Hz for BHZ
		self.sample_rate = 100
		# the onset is randomly chosen within the first 5 seconds
		self.OnsetSlidingRange = 5 # unit: second
		self.CutDuration = 50 # unit: second 
		self.type_to_label = {"quake":0, "blast":1} # add custom label oneday 
		self.label_to_type = {0:"quake", 1:"blast"}
		# the random window onset should be less than MaxCutOnset
		self.MaxCutOnset = self.OnsetSlidingRange * self.sample_rate
		self.newnpts = self.CutDuration * self.sample_rate
		# input shape and output shape
		self.num_channels = 3
		self.num_classes = 2
		self.input_shape = (self.newnpts, 1, self.num_channels)
		self.output_shape = (self.num_classes, 1)


class Dir():
	"""directory and file name"""

	def __init__(self, IfTransferLearning, archit='cnn'):
		self.dataset_dir = os.path.join("dataset")
		# Southern California data to train the raw model
		self.data_California = os.path.join(self.dataset_dir, "California",	"nonfilter_50second")
#		self.data_California = os.path.join(self.dataset_dir, "California",	"nonfilter_50second")
		# Kentucky data for transfer learning
		self.data_Kentucky = os.path.join(self.dataset_dir, "Kentucky",	"nonfilter_50second")
#		self.data_Kentucky = os.path.join(self.dataset_dir, "Kentucky",	"nonfilter_50second")
		self.dir = (self.data_Kentucky if IfTransferLearning else self.data_California)
		self.waveform = os.path.join(self.dir, "waveform")
		self.flist = os.path.join(self.dir, "log_npz.txt")
		self.test_id = os.path.join(self.dir, "test_id.txt")
		self.val_id = os.path.join(self.dir, "val_id.txt")
		self.tr_id = os.path.join(self.dir, "train_id.txt")
		# predict model
#		self.predictmodel = os.path.join("model", "SoCAL", "filter_30second", archit+".h5")
		self.predictmodel = os.path.join("model", "SoCAL", "nonfilter_50second", archit+".h5")
		# log the retrained model
		self.log_dir = os.path.join("log", date.today().strftime("%m%d%y"))
		self.model = os.path.join(self.log_dir, archit+'.h5')
		self.history = os.path.join(self.log_dir, 'history.npz')
		self.plot_history = os.path.join(self.log_dir, 'history.png')
		self.plot_cfm = os.path.join(self.log_dir, 'cfm.png')
		self.plot_roc = os.path.join(self.log_dir, 'roc.png')
		# output the result
		self.output_dir = os.path.join("output")


if __name__ == "__main__":
	IfTransferLearning = False
	print(Dir(IfTransferLearning).__dict__)
	print(Config(IfTransferLearning).__dict__)
	print(HyperParams().__dict__)
