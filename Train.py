# Usage:  train the earthquake discriminator & save the model
# Date:   AUG 31 2021
# Author: Jun ZHU
# Email:  Jun__Zhu@outlook.com


import os
import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import (confusion_matrix, classification_report,
		roc_curve, auc)
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

from Config import Dir, Config, HyperParams
from CNN_Classifier import CNN
from ResNetclassifier import ResNet
from DataGenerator import DataGenerator
from Plot import plot_train_history
from DivideDatasets import divide_datasets


def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode",
						default="retrain",
						type=str,
						help="retrain/pretrain/finetune")
	parser.add_argument("--model",
						default="model/SoCAL/nonfilter_50second/cnn.h5",
						type=str,
						help="model/SoCAL/filter_30second/cnn.h5")
	args = parser.parse_args()
	return args

def train(Dir, conf, hp, ifdivision, args):
	"""train the discriminator && save the model & save the test data
		params:
				1. directory
				2. configuration
				3. hyperparameters
				4. whether the dataset has been divided
		output:
	"""


	# parameters for train generator
	params_generator = {
			'data_dir': Dir.waveform,
			'batch_size': hp.batch_size,
			'MaxCutOnset': conf.MaxCutOnset,
			'SliceRange': conf.newnpts,
			'num_classes': len(conf.type_to_label),
			'shuffle': True,
			'TypeConvertLabel': conf.type_to_label}
	# check whether the data division has been done
	if not ifdivision:
		#--------------------------data division------------------
		print("\n---\nDividing the datasets ... ing\n---\n")
		params_division = {'TVTratio': conf.split_ratio_TVT, 'TypeFold': conf.TypeFold, 'shuffle': True}
		df = pd.read_csv(Dir.flist, delimiter=" ")
		IDs = np.array(df['id'])
		eventID = np.array(df['evid'])
		label = np.array(df['eventtype'])
		trainindex, validindex, testindex = divide_datasets(label, eventID, **params_division)
		print("\n---\nData has been divided into train, validation and test\n---\n")
		#--------------------------save the test dataset----------
		test_path = Dir.test_id
		with open(test_path, "w") as f:
			f.write("id evid eventtype\n")
			f.write('\n'.join(["%d %d %s"%(i,j,k) for i,j,k in zip(IDs[testindex],	eventID[testindex], label[testindex])]))
		#--------------------------data generator-----------------
		train_generator = DataGenerator(IDs[trainindex], **params_generator)
		validation_generator = DataGenerator(IDs[validindex], **params_generator)
		print("#train: %d; #valid: %d; test: %d"%(len(train_generator), len(validation_generator), len(testindex)))
	else:
		df_tr = pd.read_csv(Dir.tr_id, delimiter=" ")
		df_val = pd.read_csv(Dir.val_id, delimiter=" ")
		#--------------------------data generator-----------------
		train_generator = DataGenerator(df_tr['id'], **params_generator)
		validation_generator = DataGenerator(df_val['id'], **params_generator)
		print("train: %d; valid: %d"%(len(train_generator), len(validation_generator)))
	#--------------------------choose a model---------------------
	if hp.model == "cnn":
		binary = CNN(conf.input_shape, hp.dropout, lr=hp.lr) if args.mode=="retrain" else load_model(args.model)
	elif hp.model == "resn1":
		binary = ResNet(conf.input_shape, hp.num_blocks, hp.model)
	elif hp.model == "resn2":
		binary = ResNet(conf.input_shape, hp.num_blocks, hp.model)
	else:
		print("\n\tAlert: please specify a model between {'cnn', 'resn1', 'resn2'}.\n")
		exit(0)
	#--------------------------early stopping & checkpoint--------
	if not os.path.exists(Dir.log_dir):
		os.makedirs(Dir.log_dir)
	EarlyStop = EarlyStopping(monitor='val_loss', patience=5)
	checkpoint = ModelCheckpoint(Dir.model, monitor='val_loss', save_best_only=True)
	#--------------------------fit the model----------------------
	history = binary.fit(
						train_generator,
						epochs=hp.epoch,
						steps_per_epoch=len(train_generator),
						max_queue_size=conf.queue_size,
						validation_data=validation_generator,
						use_multiprocessing=conf.multiprocessing,
						workers=conf.workers,
						callbacks=[EarlyStop, checkpoint])
	#--------------------------log train history------------------
	acc = {'acc': history.history['accuracy'], 'val_acc': history.history['val_accuracy']}
	loss = {'loss': history.history['loss'], 'val_loss': history.history['val_loss']}
	np.savez(Dir.history, acc=acc, loss=loss)
	#--------------------------plot train history-----------------
	acc = history.history['accuracy'], history.history['val_accuracy']
	loss = history.history['loss'], history.history['val_loss']
	plot_train_history(acc, loss, Dir.plot_history)
	return


if __name__ == "__main__":
	args = read_args()
	IfTransferLearning = True if args.mode=="pretrain" else False
	Dir = Dir(IfTransferLearning); conf = Config(IfTransferLearning); hp = HyperParams()
	# check if the data division has been done
	ifdivision = (os.path.exists(Dir.val_id) and os.path.exists(Dir.tr_id))
	train(Dir, conf, hp, ifdivision, args)
