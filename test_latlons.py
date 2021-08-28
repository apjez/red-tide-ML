import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

file_folder = 'latlon_stats/'
features = ['angstrom', 'chlor_a', 'chl_ocx', 'Kd_490', 'nflh', 'poc']
vmaxes = [1.5, 80, 80, 6, 0.75, 2000]

for i in range(len(features)):
	feature_n = np.load(file_folder+features[i]+'_n.npy')
	feature_x = np.load(file_folder+features[i]+'_x.npy')
	feature_x2 = np.load(file_folder+features[i]+'_x2.npy')

	feature_mean = np.zeros_like(feature_n)
	feature_std = np.zeros_like(feature_n)

	for j in range(feature_mean.shape[0]):
		for k in range(feature_mean.shape[1]):
			# Avoid dividing by 0
			if(feature_n[j, k] > 0):
				feature_mean[j, k] = feature_x[j, k] / feature_n[j, k]
				feature_std[j, k] = np.sqrt((feature_x2[j, k]/feature_n[j, k]) - (feature_mean[j, k]*feature_mean[j, k]))
			else:
				feature_mean[j, k] = -1
				feature_std[j, k] = -1

	plt.figure(dpi=500)
	plt.imshow(feature_mean, origin='lower', vmin=-1, vmax=vmaxes[i])
	plt.title(features[i]+' Mean')
	plt.axis('off')
	#plt.colorbar()
	plt.savefig(file_folder+features[i]+'_mean.png', bbox_inches='tight')

	plt.figure(dpi=500)
	plt.imshow(feature_std, origin='lower')
	plt.title(features[i]+' Std')
	plt.axis('off')
	#plt.colorbar()
	plt.savefig(file_folder+features[i]+'_std.png', bbox_inches='tight')

	np.save(file_folder+features[i]+'_mean.npy', feature_mean)
	np.save(file_folder+features[i]+'_std.npy', feature_std)