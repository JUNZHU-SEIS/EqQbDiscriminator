# Usage:		Evaluate the earthquake discriminator
# Author:		Jun ZHU
# Date:			AUG 21 2021
# Email:		Jun__Zhu@outlook.com


import os
import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix, classification_report,
		roc_curve, auc)

from keras.utils.np_utils import to_categorical as category
from keras.models import load_model

import matplotlib.pyplot as plt

from Config import Dir, HyperParams, Config
from Plot import plot_confusion_matrix, plot_ROC
from DataGenerator import DataGenerator


def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model",
						default="model/SoCAL/nonfilter_50second/cnn.h5",
						type=str,
						help="model/SoCAL/nonfilter_50second/cnn.h5")
	args = parser.parse_args()
	return args

def evaluate(testgenerator, testlabel, model):
	"""evaluate the earthquake discriminator
		params:
			1. test waveform dataset
			2. test label dataset
			3. loaded model
		return:
			no return value, just print the results to the screen
	"""

	binary = model
	score = binary.evaluate(testgenerator)
							#workers=conf.workers,
							#max_queue_size=conf.queue_size,
							#use_multiprocessing=conf.multiprocessing)
	predict_proba =	binary.predict(testgenerator) # , use_multiprocessing=True)
	predict_class = np.argmax(predict_proba, axis=-1)
	true_class = np.array(testlabel[:len(predict_class)])
	#--------------------------confusion matrix-------------------
	cfm = confusion_matrix(true_class, predict_class)
	print(binary.metrics_names, score)
	print('confusion matrix:\n', cfm)
	report = classification_report(true_class, predict_class, target_names=["quake", "blast"])
	print(report)
	if not os.path.exists(Dir.log_dir):
		os.makedirs(Dir.log_dir)
	plot_confusion_matrix(cfm, Dir.plot_cfm)
	#--------------------------ROC curve--------------------------
	positive_index = 0
	positive_score = predict_proba[:, 0]
	fpr, tpr, _ = roc_curve(true_class, positive_score, pos_label=0)
	AUC = auc(fpr, tpr)
	plot_ROC(fpr=fpr, tpr=tpr, auc=AUC, path=Dir.plot_roc)
	#--------------------------statistics-------------------------
	right = np.argwhere((true_class-predict_class)==0)
	mis = np.argwhere((true_class-predict_class)!=0)
	print("right case:\n", len(right), "\nmisidentified case\n", len(mis))
	return


if __name__ == "__main__":
	args = read_args()
	IfTransferLearning = False
	IfTransferLearning = True
	Dir = Dir(IfTransferLearning); hp = HyperParams()
	conf = Config(IfTransferLearning)
	model = load_model(args.model)
	params = {
		'data_dir': Dir.waveform,
		'batch_size': min(hp.batch_size, 128),
		'MaxCutOnset': conf.MaxCutOnset,
		'SliceRange': conf.newnpts,
		'num_classes': len(conf.type_to_label),
		'shuffle': False, # must be false
		'TypeConvertLabel': conf.type_to_label}
	df = pd.read_csv(Dir.test_id, delimiter=" ")
#	df = pd.read_csv(Dir.val_id, delimiter=" ")
#	df = pd.read_csv(Dir.tr_id, delimiter=" ")
	label = df['eventtype']; IDs = df['id']; evid = df['evid']
	test_generator = DataGenerator(IDs, **params)
	test_label = [conf.type_to_label[x] for x in label]
	evaluate(test_generator, test_label, model)
