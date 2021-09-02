# Usage:	predict the probablity
# Date:		AUG 21 2021
# Author:	Jun ZHU
# Email:	Jun__Zhu@outlook.com


import os

from datetime import date

import numpy as np
import pandas as pd

from keras.models import load_model

from Config import Dir, Config, HyperParams
from DataGenerator import DataGenerator


def Predict(predict_id, predict_generator, model, output_dir):
	# predict the label of the sample
	proba = model.predict(predict_generator)
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	fname = os.path.join(output_dir, "predict.txt")
	text = ["%s %.2f %.2f"%(x, y[0], y[1]) for x,y in zip(predict_id, proba)]
	with open(fname, 'w') as f:
		dic = conf.label_to_type
		f.write('%s %s %s\n'%('id', dic[0], dic[1]))
		f.write('\n'.join(text))
	return proba


if __name__ == "__main__":
	IfTransferLearning = True
	IfTransferLearning = False
	Dir = Dir(IfTransferLearning); conf = Config(IfTransferLearning); hp = HyperParams()
	model = load_model(Dir.predictmodel)
	params = {
		'data_dir': Dir.waveform,
		'batch_size': hp.batch_size,
		'MaxCutOnset': conf.MaxCutOnset,
		'SliceRange': conf.newnpts,
		'num_classes': len(conf.type_to_label),
		'shuffle': False,
		'TypeConvertLabel': conf.type_to_label}
	df = pd.read_csv(Dir.test_id, delimiter=" ")
	label = df['eventtype']; IDs = df['id']; evid = df['evid']
	# mini test
#	num_test = 1000
#	predict_id = IDs[:num_test]
	predict_id = IDs
	predict_generator = DataGenerator(predict_id, **params)
	Predict(predict_id, predict_generator, model, output_dir=Dir.output_dir)
