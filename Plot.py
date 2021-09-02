# Usage:	visualization
# Date:		AUG 21 2021
# Author:	Jun ZHU
# Email:	Jun__Zhu@outlook.com


import numpy as np

import matplotlib.pyplot as plt


def plot_train_history(acc, loss, imagepath):
	"""plot the accuracy & loss during the training period"""

	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
	ax1.plot(acc[0])
	ax1.plot(acc[1])
	ax1.grid()
	ax1.set_ylabel('Accuracy')
	ax1.legend(['Training', 'Validation'], loc='best')
	ax2.plot(loss[0])
	ax2.plot(loss[1])
	ax2.grid()
	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('Loss')
	ax2.legend(['Training', 'Validation'], loc='best')
	ax2.set_xticks(np.arange(len(acc[0]) // 10 + 1) * 10)
	ax2.set_xticklabels(np.arange(len(acc[0]) // 10 + 1) *10)
	plt.suptitle("Train history")
	plt.savefig(imagepath, dpi=500)
	plt.close()
	return


def plot_confusion_matrix(cfm, path):
	fig, ax = plt.subplots()
	im = ax.pcolormesh(cfm, cmap="binary")
	ticks = ["quake", "blast"]
	ax.invert_yaxis()
	ax.text(0.5, 0.5, cfm[0][0], color="white")
	ax.text(1.5, 0.5, cfm[0][1], color="black")
	ax.text(0.5, 1.5, cfm[1][0], color="black")
	ax.text(1.5, 1.5, cfm[1][1], color="white")
	ax.set_xlabel("Predicted")
	ax.set_xticks((0.5, 1.5))
	ax.set_xticklabels(ticks)
	ax.xaxis.set_label_position("top")
	ax.xaxis.set_ticks_position("top")
	ax.set_ylabel("True")
	ax.set_yticks((0.5, 1.5))
	ax.set_yticklabels(ticks)
#	fig.colorbar(im, ax=ax, orientation='horizontal', label='#ticks')
	plt.savefig(path, dpi=500)
	plt.close()
	return


def plot_ROC(fpr=[0,0.1,0.2,0.3,0.4,0.5], tpr=[0,0.2,0.3,0.3,0.3,0.5], auc=1, path="./"):
	plt.plot(fpr, tpr, lw=4, color="#836FE8", label="ROC curve")
	plt.fill_between(fpr, tpr, y2=0, color="#FAAB96", label="AUC=%.2f"%auc)
	plt.plot((0,1), (0,1), "--k", label="TPR=FPR")
	plt.xlim([0,1])
	plt.ylim([0,1])
	plt.title("Receiver Operating Characteristics")
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.legend(loc="lower right")
	plt.savefig(path, dpi=500)
	plt.show()
	plt.close()
	return


if __name__=="__main__":
	from Config import Dir
	Dir = Dir()
	cfm = [[1,2],[3,4]]
	plot_ROC(path=Dir.plot_roc)
