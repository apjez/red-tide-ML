import pandas as pd
import numpy as np
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

plt.figure(dpi=500)
plt.plot(bedrock_uniques, chlor_a_means)
plt.fill(fill_xs, fill_ys, alpha=0.3)
plt.xlim(-100, 2)
plt.xlabel('Bedrock Depth')
plt.ylabel('Estimated Chlorophyll-a')
plt.title('Depth Dependence of Chlorophyll-a')
plt.savefig('chlor_a_per_depth.png')