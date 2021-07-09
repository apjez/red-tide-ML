import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from random import sample
from model import *
from dataset import *
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

numEpochs = 5000
learning_rate = 0.0001
mb_size = 100
num_classes = 2
num_models = 10
loss = nn.BCELoss()

paired_df = pd.read_pickle('paired_dataset.pkl')

#features_to_use=['Sample Date', 'Latitude', 'aot_869', 'angstrom', 'Rrs_412', 'Rrs_443', 'Rrs_469', 'Rrs_488',\
#	'Rrs_531', 'Rrs_547', 'Rrs_555', 'Rrs_645',\
#	'Rrs_667', 'Rrs_678', 'chlor_a', 'chl_ocx', 'Kd_490', 'poc', 'par', 'ipar', 'nflh', 'Red Tide Concentration']
features_to_use=['Sample Date', 'Latitude', 'angstrom', 'chlor_a', 'chl_ocx', 'Kd_490', 'poc', 'nflh', 'bedrock', 'Red Tide Concentration']

paired_df = paired_df[features_to_use]

#Remove samples with NaN values
paired_df = paired_df.dropna()

red_tide = paired_df['Red Tide Concentration'].to_numpy().copy()

dates = paired_df['Sample Date'].to_numpy().copy()

latitudes = paired_df['Latitude'].to_numpy().copy()

features = paired_df[features_to_use[2:-1]]

concentrations = red_tide
classes = np.zeros((concentrations.shape[0], 1))

for i in range(len(classes)):
	if(concentrations[i] < 100000):
		classes[i] = 0
	else:
		classes[i] = 1

for model_number in range(num_models):

	#Balance classes by number of samples
	values, counts = np.unique(classes, return_counts=True)
	pointsPerClass = np.min(counts)
	reducedInds = np.array([])
	for i in range(num_classes):
		class_inds = np.where(classes == i)[0]
		reducedInds = np.append(reducedInds, class_inds[np.random.choice(class_inds.shape[0], pointsPerClass)])

	reducedInds = reducedInds.astype(int)

	classes = classes[reducedInds]

	featuresTensor = torch.tensor(features.values)

	reducedFeaturesTensor = featuresTensor[reducedInds, :]

	dates = dates[reducedInds]
	latitudes = latitudes[reducedInds]

	trainInds = sample(range(reducedFeaturesTensor.shape[0]), int(0.8*reducedFeaturesTensor.shape[0]))
	testInds = list(set(range(reducedFeaturesTensor.shape[0]))-set(trainInds))
	#trainInds = np.where(dates < np.datetime64('2018-01-01'))[0]
	#testInds = np.where(dates >= np.datetime64('2018-01-01'))[0]
	#trainInds = np.logical_or(latitudes >= 27, latitudes < 26.5)
	#testInds = np.logical_and(latitudes < 27, latitudes >= 26.5)

	trainSet = reducedFeaturesTensor[trainInds, :].float().cuda()
	testSet = reducedFeaturesTensor[testInds, :].float().cuda()

	trainClasses = classes[trainInds]

	trainClasses = trainClasses.astype(int)

	trainTargets = np.zeros((trainClasses.shape[0], num_classes))
	for i in range(len(trainClasses)):
		trainTargets[i, trainClasses[i]] = 1

	testClasses = classes[testInds]

	testClasses = testClasses.astype(int)

	testTargets = np.zeros((testClasses.shape[0], num_classes))
	for i in range(len(testClasses)):
		testTargets[i, testClasses[i]] = 1

	trainTargets = torch.Tensor(trainTargets).float().cuda()
	testTargets = torch.Tensor(testTargets).float().cuda()

	trainDataset = RedTideDataset(trainSet, trainTargets)
	trainDataloader = DataLoader(trainDataset, batch_size=mb_size, shuffle=True)

	predictor = Predictor(trainSet.shape[1], num_classes).cuda()
	optimizer = optim.Adam(predictor.parameters(), lr=learning_rate)

	losses = np.zeros((numEpochs, 1))

	for i in range(numEpochs):
		epochLoss = 0
		for mini_batch_data, mini_batch_labels in trainDataloader:
			optimizer.zero_grad()
			output = predictor(mini_batch_data)
			miniBatchLoss = loss(output, mini_batch_labels)
			miniBatchLoss.backward()
			epochLoss += miniBatchLoss.item()
			optimizer.step()
		if(i%10==0):
			print('Epoch: {}, Loss: {}'.format(i, epochLoss))
		losses[i] = epochLoss

	plt.figure()
	plt.plot(losses)
	plt.savefig('losses.png')

	testOutput = predictor(testSet).cpu().detach().numpy()
	testOutput = np.argmax(testOutput, axis=1)
	print(confusion_matrix(testClasses, testOutput))

	torch.save(predictor.state_dict(), 'saved_model_info/predictor{}.pt'.format(model_number))
	np.save('saved_model_info/reducedInds{}.npy'.format(model_number), reducedInds)
	np.save('saved_model_info/testInds{}.npy'.format(model_number), testInds)