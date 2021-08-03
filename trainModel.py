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
from utils import *
from convertFeaturesByDepth import *
from convertFeaturesByPosition import *
import datetime as dt
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix
import json
from configparser import ConfigParser
import matplotlib.pyplot as plt

configfilename = 'date_train_test_depth_norm_w_nn'

config = ConfigParser()
config.read('configfiles/'+configfilename+'.ini')

numEpochs = config.getint('main', 'numEpochs')
learning_rate = config.getfloat('main', 'learning_rate')
mb_size = config.getint('main', 'mb_size')
num_classes = config.getint('main', 'num_classes')
randomseeds = json.loads(config.get('main', 'randomseeds'))
normalization = config.getint('main', 'normalization')
traintest_split = config.getint('main', 'traintest_split')
use_nn_feature = config.getboolean('main', 'use_nn_feature')

loss = nn.BCELoss()

paired_df = pd.read_pickle('paired_dataset.pkl')

#features_to_use=['Sample Date', 'Latitude', 'aot_869', 'angstrom', 'Rrs_412', 'Rrs_443', 'Rrs_469', 'Rrs_488',\
#	'Rrs_531', 'Rrs_547', 'Rrs_555', 'Rrs_645',\
#	'Rrs_667', 'Rrs_678', 'chlor_a', 'chl_ocx', 'Kd_490', 'poc', 'par', 'ipar', 'nflh', 'Red Tide Concentration']
features_to_use=['Sample Date', 'Latitude', 'Longitude', 'angstrom', 'chlor_a', 'chl_ocx', 'Kd_490', 'poc', 'nflh', 'bedrock', 'Red Tide Concentration', 'Rrs_443', 'Rrs_555']

paired_df = paired_df[features_to_use]

#Remove samples with NaN values
paired_df = paired_df.dropna()

red_tide = paired_df['Red Tide Concentration'].to_numpy().copy()
dates = paired_df['Sample Date'].to_numpy().copy()
latitudes = paired_df['Latitude'].to_numpy().copy()
longitudes = paired_df['Longitude'].to_numpy().copy()

features = paired_df[features_to_use[1:-3]]
features = np.array(features.values)

if(normalization == 0):
	features = features[:, 2:-1]
elif(normalization == 1):
	features = convertFeaturesByDepth(features[:, 2:], features_to_use[3:-4])
elif(normalization == 2):
	features = convertFeaturesByPosition(features[:, :-1], features_to_use[3:-4])

concentrations = red_tide
classes = np.zeros((concentrations.shape[0], 1))

for i in range(len(classes)):
	if(concentrations[i] < 100000):
		classes[i] = 0
	else:
		classes[i] = 1




if(use_nn_feature == True):
	file_path = 'PinellasMonroeCoKareniabrevis 2010-2020.06.12.xlsx'

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

	dataDates = pd.DatetimeIndex(dates)
	nn_classes = np.zeros((features.shape[0]))
	#Do some nearest neighbor thing with the last week's samples
	for i in range(len(dataDates)):
		searchdate = dataDates[i]
		weekbefore = searchdate - dt.timedelta(days=7)
		twoweeksbefore = searchdate - dt.timedelta(days=14)
		mask = (df_dates['Sample Date'] > twoweeksbefore) & (df_dates['Sample Date'] <= weekbefore)
		week_prior_inds = df_dates[mask].index.values

		if(week_prior_inds.size):
			idx = find_nearest_latlon(df_lats[week_prior_inds], df_lons[week_prior_inds], latitudes[i], longitudes[i])

			closestConc = df_concs[week_prior_inds][idx]
			if(closestConc < 100000):
				nn_classes[i] = 0
			else:
				nn_classes[i] = 1
		else:
			nn_classes[i] = 0

	ensure_folder('saved_model_info/'+configfilename)
	np.save('saved_model_info/'+configfilename+'/nn_classes.npy', nn_classes)
	features = np.concatenate((features, np.expand_dims(nn_classes, axis=1)), axis=1)





for model_number in range(len(randomseeds)):
	# Set up random seeds for reproducability
	torch.manual_seed(randomseeds[model_number])
	np.random.seed(randomseeds[model_number])

	#Balance classes by number of samples
	values, counts = np.unique(classes, return_counts=True)
	values = values[0:num_classes]
	counts = counts[0:num_classes]
	pointsPerClass = np.min(counts)
	reducedInds = np.array([])
	for i in range(num_classes):
		class_inds = np.where(classes == i)[0]
		reducedInds = np.append(reducedInds, class_inds[np.random.choice(class_inds.shape[0], pointsPerClass)])

	reducedInds = reducedInds.astype(int)

	usedClasses = classes[reducedInds]

	featuresTensor = torch.tensor(features)

	reducedFeaturesTensor = featuresTensor[reducedInds, :]

	usedDates = dates[reducedInds]
	usedLatitudes = latitudes[reducedInds]

	if(traintest_split == 0):
		trainInds = sample(range(reducedFeaturesTensor.shape[0]), int(0.8*reducedFeaturesTensor.shape[0]))
		testInds = list(set(range(reducedFeaturesTensor.shape[0]))-set(trainInds))
	elif(traintest_split == 1):
		trainInds = np.where(usedDates < np.datetime64('2018-01-01'))[0]
		testInds = np.where(usedDates >= np.datetime64('2018-01-01'))[0]
	elif(traintest_split == 2):
		trainInds = np.logical_or(usedLatitudes >= 27, usedLatitudes < 26.5)
		testInds = np.logical_and(usedLatitudes < 27, usedLatitudes >= 26.5)

	trainSet = reducedFeaturesTensor[trainInds, :].float().cuda()
	testSet = reducedFeaturesTensor[testInds, :].float().cuda()

	trainClasses = usedClasses[trainInds]

	trainClasses = trainClasses.astype(int)

	trainTargets = np.zeros((trainClasses.shape[0], num_classes))
	for i in range(len(trainClasses)):
		trainTargets[i, trainClasses[i]] = 1

	testClasses = usedClasses[testInds]

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

	ensure_folder('saved_model_info/'+configfilename)

	torch.save(predictor.state_dict(), 'saved_model_info/'+configfilename+'/predictor{}.pt'.format(model_number))
	np.save('saved_model_info/'+configfilename+'/reducedInds{}.npy'.format(model_number), reducedInds)
	np.save('saved_model_info/'+configfilename+'/testInds{}.npy'.format(model_number), testInds)