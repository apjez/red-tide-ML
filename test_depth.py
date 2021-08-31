import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

load_folder = 'depth_stats'
filenames = ['angstrom_sums.npy', 'chlor_a_sums.npy', 'chl_ocx_sums.npy', 'Kd_490_sums.npy', 'poc_sums.npy', 'nflh_sums.npy']

for filename in filenames:
	sums = np.load(load_folder+'/'+filename, allow_pickle='TRUE').item()

	keys = []
	means = []
	stds = []
	ns = []

	for key in sums.keys():
		if(key[-1] == 'n'):
			key_strip = key[:-2]
			keys.append(int(key_strip))
			means.append(sums[key_strip+'_x']/sums[key_strip+'_n'])
			stds.append(math.sqrt((sums[key_strip+'_x2']/sums[key_strip+'_n'])-(means[-1]**2)))
			ns.append(sums[key_strip+'_n'])

	keys = np.array(keys)
	means = np.array(means)
	stds = np.array(stds)
	ns = np.array(ns)

	sort_inds = np.argsort(keys)

	keys = keys[sort_inds]
	means = means[sort_inds]
	stds = stds[sort_inds]

	fill_xs = np.concatenate((keys, np.flip(keys)))
	fill_ys = np.concatenate((means+stds, np.flip(means-stds)))

	plt.figure(dpi=500)
	plt.plot(keys, means, 'b')
	plt.fill(fill_xs, fill_ys, alpha=0.3, facecolor='b')
	plt.xlim(-100, 2)
	#plt.ylim(0, 6)
	plt.xlabel('Bedrock Depth')
	plt.title(filename)
	plt.savefig('depth_stat_plots/'+filename[:-4]+'.png')

	np.save('depth_stats/'+filename[:-4]+'_keys.npy', keys)
	np.save('depth_stats/'+filename[:-4]+'_means.npy', means)
	np.save('depth_stats/'+filename[:-4]+'_stds.npy', stds)