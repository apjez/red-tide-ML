import numpy as np
import pandas as pd
import datetime as dt
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from utils import *

file_path = 'PinellasMonroeCoKareniabrevis 2010-2020.06.12.xlsx'
num_classes = 2

df = pd.read_excel(file_path, engine='openpyxl')
df_dates = df['Sample Date']
df_lats = df['Latitude'].to_numpy()
df_lons = df['Longitude'].to_numpy()
df_concs = df['Karenia brevis abundance (cells/L)'].to_numpy()

df_conc_classes = np.zeros_like(df_concs)
for i in range(len(df_conc_classes)):
	if(df_concs[i] < 100000):
		df_conc_classes[i] = 0
	else:
		df_conc_classes[i] = 1

#Balance classes by number of samples
values, counts = np.unique(df_conc_classes, return_counts=True)
values = values[0:num_classes]
counts = counts[0:num_classes]
pointsPerClass = np.min(counts)
reducedInds = np.array([])
for i in range(num_classes):
	if(i==0):
		class_inds = np.where(df_concs < 100000)[0]
	else:
		class_inds = np.where(df_concs >= 100000)[0]
	reducedInds = np.append(reducedInds, class_inds[np.random.choice(class_inds.shape[0], pointsPerClass)])

reducedInds = reducedInds.astype(int)
df_dates = df_dates[reducedInds]
df_dates = df_dates.reset_index()
df_lats = df_lats[reducedInds]
df_lons = df_lons[reducedInds]
df_concs = df_concs[reducedInds]
df_classes = df_conc_classes[reducedInds]

betas = np.arange(0.05, 5, 0.05)

knn_accs = []
nn_accs = []
iteration = 0
for beta in betas:
	if(iteration%10 == 0):
		print('Iteration #{}'.format(iteration))
	iteration = iteration+1
	dataDates = df_dates['Sample Date'].copy()
	knn_classes = np.zeros((dataDates.shape[0]))
	nn_classes = np.zeros((dataDates.shape[0]))
	mask_size = np.zeros(len(dataDates))
	#Do some nearest neighbor thing with the last week's samples
	for i in range(len(dataDates)):
		searchdate = dataDates[i]
		weekbefore = searchdate - dt.timedelta(days=3)
		twoweeksbefore = searchdate - dt.timedelta(days=24)
		mask = (df_dates['Sample Date'] > twoweeksbefore) & (df_dates['Sample Date'] <= weekbefore)
		week_prior_inds = df_dates[mask].index.values

		if(week_prior_inds.size):
			mask_size[i] = len(week_prior_inds)

			physicalDistance = 100*np.sqrt((df_lats[week_prior_inds]-df_lats[i])**2 + (df_lons[week_prior_inds]-df_lons[i])**2)
			daysBack = (searchdate - df_dates['Sample Date'][week_prior_inds]).astype('timedelta64[D]').values
			totalDistance = physicalDistance + beta*daysBack
			inverseDistance = 1/totalDistance
			NN_weights = inverseDistance/np.sum(inverseDistance)
			closestClasses = df_classes[week_prior_inds]
			negativeInds = np.where(closestClasses==0)[0]
			positiveInds = np.where(closestClasses==1)[0]

			idx = find_nearest_latlon(df_lats[week_prior_inds], df_lons[week_prior_inds], df_lats[i], df_lons[i])

			closestConc = df_concs[week_prior_inds][idx]
			if(closestConc < 100000):
				nn_classes[i] = 0
			else:
				nn_classes[i] = 1
			if(np.sum(NN_weights[negativeInds]) > np.sum(NN_weights[positiveInds])):
				knn_classes[i] = 0
			else:
				knn_classes[i] = 1
		else:
			nn_classes[i] = 0
			knn_classes[i] = 0

	knn_accs.append(metrics.accuracy_score(df_classes, knn_classes))
	nn_accs.append(metrics.accuracy_score(df_classes, nn_classes))

plt.figure(dpi=500)
plt.plot(betas, knn_accs, label='KNN')
plt.plot(betas, nn_accs, label='NN')
plt.xlabel('Beta')
plt.ylabel('Classification Accuracy')
plt.legend()
plt.title('Distance Function Analysis')
plt.savefig('dist_func2new.png')