"Modified from Afshine Amidi & Shervine Amidi's blog"
# Usage:  Generator for Keras
# Author: Jun ZHU
# Date:   AUG 23, 2021
# Email:  Jun__Zhu@outlook.com


import os

import numpy as np

from keras.utils import Sequence
from keras.utils import to_categorical as onehot

from Config import Config, HyperParams, Dir


class DataGenerator(Sequence):
	"""Custom data generator for Keras"""

	def __init__(self,
			list_IDs,
			data_dir='..',
			num_classes=2,
			batch_size=128,
			SliceRange=3000,
			MaxCutOnset=500,
			shuffle=True,
			TypeConvertLabel={"quake":0, "blast":1}):
		"""initialize the class
			params:
					1. list_IDs: ID of the file
					2. **kwargs: e.g. prefix: directory of waveform data
		"""

		self.data_dir = data_dir
		self.list_IDs = list_IDs
		self.TypeConvertLabel = TypeConvertLabel
		self.shape = (SliceRange, 1)
		self.MaxCutOnset = MaxCutOnset
		self.SliceRange = SliceRange
		self.batch_size = batch_size
		self.num_channels = 3
		self.num_classes = num_classes
		self.shuffle = shuffle
		self.on_epoch_end() # shuffle if necessary

	def __len__(self):
		"""length of the mini-batch"""

		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		"""get mini-batch data for fit_generator in Keras"""

		indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]
		list_IDs_tmp = [self.list_IDs[k] for k in indexes]
		X, y = self.__data_generation(list_IDs_tmp)
		return X, y

	def on_epoch_end(self):
		"""shuffle at the end of each epoch"""

		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_tmp):
		"""generate data of mini-batch size"""

		X = np.empty((self.batch_size, *self.shape, self.num_channels))
		y = np.empty((self.batch_size), dtype=int)
		for i, ID in enumerate(list_IDs_tmp):
			data = np.load(os.path.join(self.data_dir, str(ID)+'.npz'))
			randomonset = np.random.randint(self.MaxCutOnset)
			# a sample with a random onset
			sample = np.transpose(data['wf'][:, randomonset: randomonset+self.SliceRange])
			# normalization by maximum standard error
			X[i, ] = sample.reshape(*self.shape, self.num_channels) /	np.max(np.std(sample, axis=0))
			y[i] = self.TypeConvertLabel[str(data['label'])]
#			print("shape of input: ", X[i, ].shape, "target: ", y[i], "id: ",
#					data['id'], "evid: ", data['evid'], "event type: ",
#					data['label'], "file path: ",
#					os.path.join(self.data_dir,	str(ID)+'.npz'))
#			print("\n\n", np.max(X[i, ], axis=0))
#			import matplotlib.pyplot as plt
#			fig, ax = plt.subplots(3, 1, sharex=True)
#			ax[0].plot(X[i,:,:,0], color='k', lw=0.2)
#			ax[1].plot(X[i,:,:,1], color='k', lw=0.2)
#			ax[2].plot(X[i,:,:,2], color='k', lw=0.2)
#			plt.suptitle("random onset: %d"%randomonset)
#			plt.show()
#			plt.close()
		return X, onehot(y, num_classes=self.num_classes)


if __name__ == "__main__":
#	IfTransferLearning = True
	IfTransferLearning = False
	conf = Config(IfTransferLearning); hp = HyperParams(); Dir = Dir(IfTransferLearning)
	# Parameters
	params = {
			'data_dir': Dir.waveform,
			'batch_size': hp.batch_size,
#			'batch_size': 10000, # uncomment for test
			'MaxCutOnset': conf.MaxCutOnset,
			'SliceRange': conf.newnpts,
			'num_classes': len(conf.type_to_label),
			'shuffle': False, # False if test
#			'shuffle': True,
			'TypeConvertLabel': conf.type_to_label}
	# Datasets
	import pandas as pd
	df = pd.read_csv(Dir.test_id,  delimiter=" ")
	partition = df['id']
	# Generator
	train_generator = DataGenerator(partition, **params)
	X, Y = train_generator.__getitem__(0)
	print(X.__getitem__(1))
	print(Y.__getitem__(1))
	print(len(train_generator))
