import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage

load_folder = 'depth_stats'
filenames = ['angstrom_fullstate_sums.npy', 'chlor_a_fullstate_sums.npy', 'chl_ocx_fullstate_sums.npy', 'Kd_490_fullstate_sums.npy', 'poc_fullstate_sums.npy', 'nflh_fullstate_sums.npy']
plot_titles = ['Angstrom', 'Chlorophyll-a OCI', 'Chlorophyll-a OCX', 'Diffuse Attenuation at 490 nm', 'Particulate Organic Carbon', 'Normalized Fluorescence Line Height']
plot_y_lim_mins = [0.9, -1, -1, 0, 25, 0.05]
plot_y_lim_maxs = [1.25, 6, 6, 0.35, 350, 0.18]
plotNumber = 0

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

	# Filter means and stds to be smoother
	means = ndimage.median_filter(means, size=5)
	stds = ndimage.median_filter(stds, size=5)
	stds = stds/5

	fill_xs = np.concatenate((keys, np.flip(keys)))
	fill_ys = np.concatenate((means+stds, np.flip(means-stds)))

	plt.figure(dpi=500)
	plt.plot(keys, means, 'b')
	plt.fill(fill_xs, fill_ys, alpha=0.3, facecolor='b')
	plt.xlim(-50, 0)
	plt.ylim(plot_y_lim_mins[plotNumber], plot_y_lim_maxs[plotNumber])
	plt.xlabel('Bedrock Depth')
	plt.title(plot_titles[plotNumber])
	plt.savefig('depth_stat_plots/'+filename[:-4]+'.png', bbox_inches='tight')

	np.save('depth_stats/'+filename[:-4]+'_fullstate_keys.npy', keys)
	np.save('depth_stats/'+filename[:-4]+'_fullstate_means.npy', means)
	np.save('depth_stats/'+filename[:-4]+'_fullstate_stds.npy', stds)

	plotNumber = plotNumber + 1