import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

filename_roc_curve_info = 'roc_curve_info'

files = [f for f in listdir(filename_roc_curve_info) if isfile(join(filename_roc_curve_info, f))]

files.sort()

file_splits = []
for file in files:
	file_splits.append(file.split('_')[0])
split_uniques = np.unique(file_splits)

last_unique = 'None'

plt.figure(dpi=500)

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
		plt.savefig(last_unique+'.png')

		plt.figure(dpi=500)
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
		plt.plot(fpr, tpr_means, label=file)
		x_values = np.concatenate((fpr, np.flip(fpr)))
		y_values = np.concatenate((tpr_means+tpr_stds, np.flip(tpr_means-tpr_stds)))
		plt.fill(x_values, y_values, alpha=0.3)
	else:
		fpr = fpr_and_tprs[0]
		tpr = fpr_and_tprs[1]
		plt.scatter(fpr, tpr, label=file)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.title('Model Performance')
plt.legend(loc='lower right')
plt.savefig(last_unique+'.png')