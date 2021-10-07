import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

paired_df = pd.read_pickle('paired_dataset.pkl')
red_tide = paired_df['Red Tide Concentration'].to_numpy().copy()
dates = paired_df['Sample Date'].to_numpy().copy()

months = np.zeros_like(red_tide)
red_tide_classes = np.zeros_like(red_tide)

for i in range(len(months)):
	months[i] = int(str(dates[i])[5:7])

for i in range(len(red_tide_classes)):
	if(red_tide[i] > 100000):
		red_tide_classes[i] = 1

neg_sums = np.zeros((12, 1))
pos_sums = np.zeros((12, 1))

for i in range(len(red_tide)):
	if(red_tide_classes[i] == 1):
		pos_sums[months[i]-1] += 1
	else:
		neg_sums[months[i]-1] += 1

plt.figure(dpi=500)
plt.bar(np.arange(1, 13), np.squeeze(pos_sums))
plt.xticks(np.arange(1, 13), month_labels, rotation='vertical')
plt.ylim(0, 1500)
plt.title('Samples Over 100,000 cells/L')
plt.savefig('pos_samples.png')

plt.figure(dpi=500)
plt.bar(np.arange(1, 13), np.squeeze(neg_sums))
plt.xticks(np.arange(1, 13), month_labels, rotation='vertical')
plt.ylim(0, 1500)
plt.title('Samples Under 100,000 cells/L')
plt.savefig('neg_samples.png')

plt.figure(dpi=500)
plt.subplot(2, 1, 1)
plt.bar(np.arange(1, 13), np.squeeze(pos_sums))
plt.xticks(np.arange(1, 13), [], rotation='vertical')
plt.ylim(0, 1500)
plt.title('Samples Over 100,000 cells/L')
plt.subplot(2, 1, 2)
plt.bar(np.arange(1, 13), np.squeeze(neg_sums))
plt.xticks(np.arange(1, 13), month_labels, rotation='vertical')
plt.ylim(0, 1500)
plt.title('Samples Under 100,000 cells/L')
plt.savefig('sample_distribution.png')

print(np.sum(neg_sums))
print(np.sum(pos_sums))

plt.figure(dpi=500)
plt.bar(np.arange(1, 13), np.squeeze(neg_sums), color='#0021A5', label='Samples Under 100,000 cells/L')
plt.bar(np.arange(1, 13), np.squeeze(pos_sums), color='#FA4616', label='Samples Over 100,000 cells/L')
plt.xticks(np.arange(1, 13), month_labels, rotation='vertical')
plt.legend()
plt.title('Paired Data Distribution')
plt.savefig('sample_distribution2.png')