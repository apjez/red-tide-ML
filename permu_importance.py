import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import datetime as dt
from random import sample
from model import *
from dataset import *
from utils import *
from convertFeaturesByDepth import *
from convertFeaturesByPosition import *
from SotoEtAlDetector import *
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix
import json
from configparser import ConfigParser
import matplotlib.pyplot as plt

configfilename = 'random_train_test'

config = ConfigParser()
config.read('configfiles/'+configfilename+'.ini')

numEpochs = config.getint('main', 'numEpochs')
learning_rate = config.getfloat('main', 'learning_rate')
mb_size = config.getint('main', 'mb_size')
num_classes = config.getint('main', 'num_classes')
randomseeds = json.loads(config.get('main', 'randomseeds'))
depth_normalize = config.getboolean('main', 'depth_normalize')
traintest_split = config.getint('main', 'traintest_split')

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

features_lin_lee = paired_df[['chl_ocx', 'nflh', 'Rrs_443', 'Rrs_555']].to_numpy().copy()

features = paired_df[features_to_use[1:-3]]
features_used = features_to_use[3:-4]

features = np.array(features.values)
if(depth_normalize):
	features = convertFeaturesByDepth(features[:, 2:], features_to_use[3:-4])
else:
	features = convertFeaturesByPosition(features[:, :-1], features_to_use[3:-4])

concentrations = red_tide
classes = np.zeros((concentrations.shape[0], 1))

for i in range(len(classes)):
	if(concentrations[i] < 100000):
		classes[i] = 0
	else:
		classes[i] = 1

accs = np.zeros((len(randomseeds), 1))
confusonMatrixSum = np.zeros((num_classes, num_classes))
accsLinLee = np.zeros((len(randomseeds), 1))
permu_accs = np.zeros((len(randomseeds), len(features_used)))
accsNN = np.zeros((len(randomseeds), 1))

for model_number in range(len(randomseeds)):

	reducedInds = np.load('saved_model_info/'+configfilename+'/reducedInds{}.npy'.format(model_number))

	usedClasses = classes[reducedInds]

	featuresTensor = torch.tensor(features)
	reducedFeaturesTensor = featuresTensor[reducedInds, :]

	reducedFeaturesLinLee = features_lin_lee[reducedInds, :]

	reducedDates = dates[reducedInds]
	reducedLatitudes = latitudes[reducedInds]
	reducedLongitudes = longitudes[reducedInds]

	testInds = np.load('saved_model_info/'+configfilename+'/testInds{}.npy'.format(model_number))

	reducedDates = reducedDates[testInds]
	reducedLatitudes = reducedLatitudes[testInds]
	reducedLongitudes = reducedLongitudes[testInds]

	testSet = reducedFeaturesTensor[testInds, :].float().cuda()

	testSetLinLee = reducedFeaturesLinLee[testInds, :].astype(float)

	outputLinLee = SotoEtAlDetector(testSetLinLee)

	testClasses = usedClasses[testInds]

	testClasses = testClasses.astype(int)

	predictor = Predictor(testSet.shape[1], num_classes).cuda()
	predictor.load_state_dict(torch.load('saved_model_info/'+configfilename+'/predictor{}.pt'.format(model_number)))
	predictor.eval()

	output = predictor(testSet)

	output = output.detach().cpu().numpy()

	output = np.argmax(output, axis=1)

	reducedDates = pd.DatetimeIndex(reducedDates)
	nn_preds = np.zeros_like(output)
	#Do some nearest neighbor thing with the last week's samples
	for i in range(len(reducedDates)):
		searchdate = reducedDates[i]
		weekbefore = searchdate - dt.timedelta(days=7)
		twoweeksbefore = searchdate - dt.timedelta(days=14)
		mask = (df_dates['Sample Date'] > twoweeksbefore) & (df_dates['Sample Date'] <= weekbefore)
		week_prior_inds = df_dates[mask].index.values

		if(week_prior_inds.size):
			idx = find_nearest_latlon(df_lats[week_prior_inds], df_lons[week_prior_inds], reducedLatitudes[i], reducedLongitudes[i])

			closestConc = df_concs[week_prior_inds][idx]
			if(closestConc < 100000):
				nn_preds[i] = 0
			else:
				nn_preds[i] = 1
		else:
			nn_preds[i] = 0

	accs[model_number] = accuracy_score(testClasses, output)
	confusonMatrixSum += confusion_matrix(testClasses, output)
	accsLinLee[model_number] = accuracy_score(testClasses, outputLinLee)
	accsNN[model_number] = accuracy_score(testClasses, nn_preds)

	feature_permu = np.random.permutation(testSet.shape[0])
	for i in range(testSet.shape[1]):
		# permute feature i
		testSetClone = testSet.clone()
		testSetClone[:, i] = testSetClone[feature_permu, i]

		output = predictor(testSetClone)

		output = output.detach().cpu().numpy()

		output = np.argmax(output, axis=1)
		
		acc = accuracy_score(testClasses, output)

		permu_accs[model_number, i] = accs[model_number] - acc

feature_importance = np.zeros(testSet.shape[1])
for i in range(testSet.shape[1]):
	feature_importance[i] = np.mean(permu_accs[:, i])

print('Average Accuracy Soto et al: {}'.format(np.mean(accsLinLee)))
print('Average Accuracy NN: {}'.format(np.mean(accsNN)))
print('Average Accuracy: {}'.format(np.mean(accs)))
print(confusonMatrixSum)

inds = np.argsort(feature_importance)
inds = np.flip(inds)
for i in range(testSet.shape[1]):
	print('{}: {}'.format(features_used[inds[i]], feature_importance[inds[i]]))