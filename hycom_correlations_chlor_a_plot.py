import numpy as np
import matplotlib.pyplot as plt

chlor_a_pairs = np.load("/run/media/rfick/UF10/HYCOM/chlor_a_pairs.npy")

unique_times = np.unique(chlor_a_pairs[:,0])

chlor_a_means = np.zeros_like(unique_times)

for i in range(len(unique_times)):
	time_inds = np.where(chlor_a_pairs == unique_times[i])[0]
	chlor_a_means[i] = np.mean(chlor_a_pairs[time_inds, 1])

plt.figure(dpi=500)
plt.scatter(unique_times, chlor_a_means)
plt.ylim(0, 5)
plt.savefig('test.png')