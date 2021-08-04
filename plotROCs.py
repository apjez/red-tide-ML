import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from os import listdir
from os.path import isfile, join

filename_roc_curve_info = 'roc_curve_info'

files = [f for f in listdir(filename_roc_curve_info) if isfile(join(filename_roc_curve_info, f))]

files.sort()

file_splits = []
for file in files:
	file_splits.append(file.split('_')[0])
split_uniques = np.unique(file_splits)

plotColors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plotNumber = 0

last_unique = 'None'

plt.figure(dpi=500)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], 'k')

for file in files:
	# Check if file name is a new type
	split = file.split('_')[0]
	if(split != last_unique and last_unique != 'None'):
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.xlim(-0.05, 1.05)
		plt.ylim(-0.05, 1.05)
		plt.title('Model Performance')
		plt.legend(loc='lower right')
		plt.savefig(last_unique+'.png', bbox_inches='tight')

		plt.figure(dpi=500)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], 'k')
		plotNumber = 0
	last_unique = split

	fpr_and_tprs = np.load(filename_roc_curve_info+'/'+file)
	if(len(fpr_and_tprs)>2):
		fpr = fpr_and_tprs[:, 0]

		tpr_means = np.zeros(fpr_and_tprs.shape[0])
		tpr_stds = np.zeros(fpr_and_tprs.shape[0])
		for i in range(fpr_and_tprs.shape[0]):
			tpr_means[i] = np.mean(fpr_and_tprs[i, 1:])
			tpr_stds[i] = np.std(fpr_and_tprs[i, 1:])
		# Insert values to make sure plots start at (0, 0)
		fpr = np.insert(fpr, 0, 0)
		tpr_means = np.insert(tpr_means, 0, 0)
		tpr_stds = np.insert(tpr_stds, 0, 0)
		# Insert values to make sure plots end at (1, 1)
		fpr = np.append(fpr, 1)
		tpr_means = np.append(tpr_means, 1)
		tpr_stds = np.append(tpr_stds, 1)
		plt.plot(fpr, tpr_means, label=file, color=plotColors[plotNumber], zorder=0)
		x_values = np.concatenate((fpr, np.flip(fpr)))
		y_values = np.concatenate((tpr_means+tpr_stds, np.flip(tpr_means-tpr_stds)))
		plt.fill(x_values, y_values, alpha=0.3, color=plotColors[plotNumber], zorder=0)
		plotNumber += 1
	else:
		fpr = fpr_and_tprs[0]
		tpr = fpr_and_tprs[1]
		plt.scatter(fpr, tpr, label=file, color=plotColors[plotNumber], zorder=10)
		plotNumber += 1

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.title('Model Performance')
plt.legend(loc='lower right')
plt.savefig(last_unique+'.png', bbox_inches='tight')