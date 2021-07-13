import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

paired_df = pd.read_pickle('paired_dataset.pkl')

features_to_use=['chlor_a', 'bedrock']
paired_df = paired_df[features_to_use]

#Remove samples with NaN values
paired_df = paired_df.dropna()

chlor_a = paired_df['chlor_a'].to_numpy().copy()
bedrock = paired_df['bedrock'].to_numpy().copy()

bedrock_uniques = np.unique(bedrock)

# Find stats of chlor_a per depth
chlor_a_means = np.zeros_like(bedrock_uniques)
chlor_a_means = chlor_a_means.astype(float)
chlor_a_stds = np.zeros_like(bedrock_uniques)
chlor_a_stds = chlor_a_stds.astype(float)

for i in range(len(bedrock_uniques)):
	inds = np.where(bedrock == bedrock_uniques[i])[0]
	chlor_a_means[i] = np.mean(chlor_a[inds])
	chlor_a_stds[i] = np.std(chlor_a[inds])

fill_xs = np.concatenate((bedrock_uniques, np.flip(bedrock_uniques)))
fill_ys = np.concatenate((chlor_a_means+chlor_a_stds, np.flip(chlor_a_means-chlor_a_stds)))

chlor_a_sums = np.load('chlor_a_sums.npy', allow_pickle='TRUE').item()

all_chlor_a_keys = []
all_chlor_a_means = []
all_chlor_a_stds = []
all_chlor_a_ns = []

for key in chlor_a_sums.keys():
	if(key[-1] == 'n'):
		key_strip = key[:-2]
		all_chlor_a_keys.append(int(key_strip))
		all_chlor_a_means.append(chlor_a_sums[key_strip+'_x']/chlor_a_sums[key_strip+'_n'])
		all_chlor_a_stds.append(math.sqrt((chlor_a_sums[key_strip+'_x2']/chlor_a_sums[key_strip+'_n'])-(all_chlor_a_means[-1]**2)))
		all_chlor_a_ns.append(chlor_a_sums[key_strip+'_n'])

all_chlor_a_keys = np.array(all_chlor_a_keys)
all_chlor_a_means = np.array(all_chlor_a_means)
all_chlor_a_stds = np.array(all_chlor_a_stds)
all_chlor_a_ns = np.array(all_chlor_a_ns)

sort_inds = np.argsort(all_chlor_a_keys)

all_chlor_a_keys = all_chlor_a_keys[sort_inds]
all_chlor_a_means = all_chlor_a_means[sort_inds]
all_chlor_a_stds = all_chlor_a_stds[sort_inds]

all_fill_xs = np.concatenate((all_chlor_a_keys, np.flip(all_chlor_a_keys)))
all_fill_ys = np.concatenate((all_chlor_a_means+all_chlor_a_stds, np.flip(all_chlor_a_means-all_chlor_a_stds)))

plt.figure(dpi=500)
plt.plot(bedrock_uniques, chlor_a_means, 'b')
plt.fill(fill_xs, fill_ys, alpha=0.3, facecolor='b')
plt.plot(all_chlor_a_keys, all_chlor_a_means, 'r')
plt.fill(all_fill_xs, all_fill_ys, alpha=0.3, facecolor='r')
plt.xlim(-100, 2)
#plt.ylim(0, 6)
plt.xlabel('Bedrock Depth')
plt.ylabel('Estimated Chlorophyll-a')
plt.title('Depth Dependence of Chlorophyll-a')
plt.savefig('chlor_a_per_depth.png')

filenames = ['angstrom_sums.npy', 'chlor_a_sums.npy', 'chl_ocx_sums.npy', 'Kd_490_sums.npy', 'poc_sums.npy', 'nflh_sums.npy']

for filename in filenames:
	sums = np.load(filename, allow_pickle='TRUE').item()

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
	plt.savefig(filename[:-4]+'.png')

	np.save('depth_stats/'+filename[:-4]+'_keys.npy', keys)
	np.save('depth_stats/'+filename[:-4]+'_means.npy', means)
	np.save('depth_stats/'+filename[:-4]+'_stds.npy', stds)