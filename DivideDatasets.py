# Usage:	Divide the the dataset into three sets according to event ID
# Author:	Jun ZHU
# Date:		AUG 23 2021
# Email:	Jun__Zhu@outlook.com


import os

import numpy as np
import pandas as pd

from Config import Dir, Config


def divide_datasets(label, eventID,
		TVTratio=(80,5,15), TypeFold={'quake':1, 'blast':4},
		shuffle=True):
	"""For each dataset, you should set data agumentation
		params:
				1. label
				2. eventID
				3. **kwargs
		output:
				1. index of train/validation/test dataset
	"""


	label = np.array(label); eventID = np.array(eventID)
	labeltype, counts = np.unique(label, return_counts=True)
	trainindex = np.array([], dtype=np.int32)
	validindex = np.array([], dtype=np.int32)
	testindex = np.array([], dtype=np.int32)
	for x in labeltype:
		x_index = np.where(label==x)[0] # index of a given event type
		evid, counts = np.unique(eventID[x_index], return_counts=True)
		#---------------------------------------------------------------------
		# shuffle the event, avoiding putting temporally-adjacent events in a set
		shufflematrix = np.arange(len(evid)); np.random.shuffle(shufflematrix)
		evid = evid[shufflematrix]; counts = counts[shufflematrix]
		#---------------------------------------------------------------------
		# 2 pointers (div1 & div2) determining where to divide datasets
		divergence = len(x_index) * np.cumsum(TVTratio) / np.sum(TVTratio)
		div1 = np.argmin(np.abs(np.cumsum(counts) - divergence[0]))
		div2 = np.argmin(np.abs(np.cumsum(counts) - divergence[1]))
		# print(x, (div1, div2), x_index, evid, counts)
		train = np.concatenate([np.where(eventID==x)[0] for x in evid[:div1+1]])
		# augmented train dataset
		agm_train = np.hstack([train for x in range(TypeFold[x])])
		# print(x, TypeFold, train, agm_train)
		trainindex = np.concatenate([trainindex, agm_train])
		valid = np.concatenate([np.where(eventID==x)[0] for x in evid[div1+1: div2+1]])
		agm_valid = np.hstack([valid for x in range(TypeFold[x])])
		validindex = np.concatenate([validindex, agm_valid])
		test = np.concatenate([np.where(eventID==x)[0] for x in evid[div2+1:]])
#		agm_test = np.hstack([test for x in range(TypeFold[x])]) # augmented train dataset
		testindex = np.concatenate([testindex, test])
	# shuffle the index (optional)
	if shuffle:
		np.random.shuffle(trainindex)
		np.random.shuffle(validindex)
		np.random.shuffle(testindex)
	return trainindex, validindex, testindex


if __name__ == "__main__":
	# mini test
	params = {'TVTratio': (1, 1, 1), 'TypeFold': {'quake':1, 'blast':1}, 'shuffle': False}
	label = np.array(['blast']*12+['quake']*12)
	eventID = np.array(['0']*2+['1']*2+['2']*3+['3']*1+['4']*2+['5']*2+['6']*4+['7']*1+['8']*3+['9']*4)
#	print(np.unique(label, return_counts=True))
#	print(np.unique(eventID, return_counts=True))
#	tr, val, te = divide_datasets(label, eventID, **params)
#	print(len(tr), len(val), len(te))
#	print(eventID[tr], eventID[val], eventID[te])
#	print(label[tr], label[val], label[te])

	# test on real data
	IfTransferLearning = True
#	IfTransferLearning = False
	Dir = Dir(IfTransferLearning); conf = Config(IfTransferLearning)
	params = {'TVTratio': conf.split_ratio_TVT, 'TypeFold': conf.TypeFold, 'shuffle': True}
	fpath = Dir.flist
	df = pd.read_csv(fpath, delimiter=" ")
	IDs = np.array(df['id'])
	eventID = np.array(df['evid'])
	label = np.array(df['eventtype'])
	print(np.unique(label, return_counts=True))
	print(np.unique(eventID, return_counts=True))
	tr, val, te = divide_datasets(label, eventID, **params)
	print(len(tr), len(val), len(te))
	print(eventID[tr], eventID[val], eventID[te])
	print(label[tr], label[val], label[te])
	# save the results
	test_npzpath = Dir.test_id
	tr_npzpath = Dir.tr_id
	val_npzpath = Dir.val_id
	for x, y in zip([tr,val,te], [tr_npzpath,val_npzpath,test_npzpath]):
		with open(y, "w") as f:
			f.write("id evid eventtype\n")
			f.write('\n'.join(["%d %d %s"%(i,j,k) for i,j,k in zip(IDs[x],
				eventID[x], label[x])]))
