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
from convertROC import *
from convertFeaturesByDepth import *
from convertFeaturesByPosition import *
from detectors.SotoEtAlDetector import *
from detectors.AminEtAlDetector import *
from detectors.StumpfEtAlDetector import *
from detectors.CannizzaroEtAlDetector import *
from detectors.ShehhiEtAlDetector import *
from detectors.SS488Detector import *
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
import json
from configparser import ConfigParser
import matplotlib.pyplot as plt

configfilename = 'date_train_test_depth_norm_w_knn'

config = ConfigParser()
config.read('configfiles/'+configfilename+'.ini')

numEpochs = config.getint('main', 'numEpochs')
learning_rate = config.getfloat('main', 'learning_rate')
mb_size = config.getint('main', 'mb_size')
num_classes = config.getint('main', 'num_classes')
randomseeds = json.loads(config.get('main', 'randomseeds'))
normalization = config.getint('main', 'normalization')
traintest_split = config.getint('main', 'traintest_split')
use_nn_feature = config.getint('main', 'use_nn_feature')
balance_train = config.getint('main', 'balance_train')

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


if(balance_train == 0):
	##### Don't balance data by classes
	reducedInds = np.array(range(len(df_concs)))



reducedInds = reducedInds.astype(int)
df_dates = df_dates[reducedInds]
df_dates = df_dates.reset_index()
df_lats = df_lats[reducedInds]
df_lons = df_lons[reducedInds]
df_concs = df_concs[reducedInds]
df_concs_log = np.log10(df_concs)/np.max(np.log10(df_concs))
df_concs_log[np.isinf(df_concs_log)] = 0
df_classes = df_conc_classes[reducedInds]

paired_df = pd.read_pickle('paired_dataset.pkl')

#features_to_use=['Sample Date', 'Latitude', 'aot_869', 'angstrom', 'Rrs_412', 'Rrs_443', 'Rrs_469', 'Rrs_488',\
#	'Rrs_531', 'Rrs_547', 'Rrs_555', 'Rrs_645',\
#	'Rrs_667', 'Rrs_678', 'chlor_a', 'chl_ocx', 'Kd_490', 'poc', 'par', 'ipar', 'nflh', 'Red Tide Concentration']
#features_to_use=['Sample Date', 'Latitude', 'Longitude', 'aot_869', 'par', 'ipar', 'angstrom', 'chlor_a', 'chl_ocx', 'Kd_490', 'poc', 'nflh', 'bedrock', 'Red Tide Concentration', 'Rrs_443', 'Rrs_555', 'Rrs_667', 'Rrs_678']
features_to_use=['Sample Date', 'Latitude', 'Longitude', 'angstrom', 'chlor_a', 'chl_ocx', 'Kd_490', 'poc', 'nflh', 'bedrock', 'Red Tide Concentration', 'Rrs_443', 'Rrs_555', 'Rrs_667', 'Rrs_678', 'Rrs_469', 'Rrs_488', 'Rrs_531']

paired_df = paired_df[features_to_use]

#Remove samples with NaN values
paired_df = paired_df.dropna()

red_tide = paired_df['Red Tide Concentration'].to_numpy().copy()

dates = paired_df['Sample Date'].to_numpy().copy()

latitudes = paired_df['Latitude'].to_numpy().copy()
longitudes = paired_df['Longitude'].to_numpy().copy()

features_lin_lee = paired_df[['chl_ocx', 'nflh', 'Rrs_443', 'Rrs_555']].to_numpy().copy()
features_amin = paired_df[['Rrs_667', 'Rrs_678']].to_numpy().copy()
features_stumpf = paired_df[['chlor_a']].to_numpy().copy()
features_shehhi = paired_df[['nflh']].to_numpy().copy()
features_SS488 = paired_df[['Rrs_469', 'Rrs_488', 'Rrs_531']].to_numpy().copy()
#features_cannizzaro = paired_df[['chl_ocx', 'Rrs_443', 'Rrs_555']].to_numpy().copy()

features = paired_df[features_to_use[1:-8]]
features_used = features_to_use[3:-9]

features = np.array(features.values)
if(normalization == 0):
	features = features[:, 2:-1]
elif(normalization == 1):
	features = convertFeaturesByDepth(features[:, 2:], features_to_use[3:-9])
elif(normalization == 2):
	features = convertFeaturesByPosition(features[:, :-1], features_to_use[3:-9])

if(use_nn_feature == 1):
	nn_classes = np.load('saved_model_info/'+configfilename+'/nn_classes.npy')
	features = np.concatenate((features, np.expand_dims(nn_classes, axis=1)), axis=1)
	features_used.append('nearest_ground_truth')
if(use_nn_feature == 2):
	knn_concs = np.load('saved_model_info/'+configfilename+'/knn_concs.npy')
	features = np.concatenate((features, np.expand_dims(knn_concs, axis=1)), axis=1)
	features_used.append('weighted_knn_conc')

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

refFpr = []
tprs = []
refFprNN = []
tprsNN = []
refFprKNN = []
tprsKNN = []
fprsSoto = []
tprsSoto = []
fprsAmin = []
tprsAmin = []
refFprStumpf = []
tprsStumpf = []
refFprShehhi = []
tprsShehhi = []
refFprSS488 = []
tprsSS488 = []
#fprsCannizzaro = []
#tprsCannizzaro = []

beta = 1

for model_number in range(len(randomseeds)):

	reducedInds = np.load('saved_model_info/'+configfilename+'/reducedInds{}.npy'.format(model_number))

	usedClasses = classes[reducedInds]

	featuresTensor = torch.tensor(features)
	reducedFeaturesTensor = featuresTensor[reducedInds, :]

	reducedFeaturesLinLee = features_lin_lee[reducedInds, :]
	reducedFeaturesAmin = features_amin[reducedInds, :]
	reducedFeaturesStumpf = features_stumpf[reducedInds, :]
	reducedFeaturesShehhi = features_shehhi[reducedInds, :]
	reducedFeaturesSS488 = features_SS488[reducedInds, :]
	#reducedFeaturesCannizzaro = features_cannizzaro[reducedInds, :]

	reducedDates = dates[reducedInds]
	reducedLatitudes = latitudes[reducedInds]
	reducedLongitudes = longitudes[reducedInds]

	testInds = np.load('saved_model_info/'+configfilename+'/testInds{}.npy'.format(model_number))

	reducedDates = reducedDates[testInds]
	reducedLatitudes = reducedLatitudes[testInds]
	reducedLongitudes = reducedLongitudes[testInds]

	testSet = reducedFeaturesTensor[testInds, :].float().cuda()

	testSetLinLee = reducedFeaturesLinLee[testInds, :].astype(float)
	testSetAmin = reducedFeaturesAmin[testInds, :].astype(float)
	testSetStumpf = reducedFeaturesStumpf[testInds, :].astype(float)
	testSetShehhi = reducedFeaturesShehhi[testInds, :].astype(float)
	testSetSS488 = reducedFeaturesSS488[testInds, :].astype(float)
	#testSetCannizzaro = reducedFeaturesCannizzaro[testInds, :].astype(float)

	outputLinLee = SotoEtAlDetector(testSetLinLee)
	outputAmin = AminEtAlDetector(testSetAmin)
	outputStumpf = StumpfEtAlDetector(testSetStumpf)
	outputShehhi = ShehhiEtAlDetector(testSetShehhi)
	outputSS488 = SS488Detector(testSetSS488)
	#outputCannizzaro = CannizzaroEtAlDetector(testSetCannizzaro)

	testClasses = usedClasses[testInds]

	testClasses = testClasses.astype(int)

	predictor = Predictor(testSet.shape[1], num_classes).cuda()
	predictor.load_state_dict(torch.load('saved_model_info/'+configfilename+'/predictor{}.pt'.format(model_number)))
	predictor.eval()

	output = predictor(testSet)

	output = output.detach().cpu().numpy()

	reducedDates = pd.DatetimeIndex(reducedDates)
	nn_preds = np.zeros((output.shape[0]))
	knn_preds = np.zeros((output.shape[0]))
	knn_concs = np.zeros((output.shape[0]))
	nn_concs = np.zeros((output.shape[0]))
	#Do some nearest neighbor thing with the last week's samples
	for i in range(len(reducedDates)):
		searchdate = reducedDates[i]
		weekbefore = searchdate - dt.timedelta(days=3)
		twoweeksbefore = searchdate - dt.timedelta(days=10)
		mask = (df_dates['Sample Date'] > twoweeksbefore) & (df_dates['Sample Date'] <= weekbefore)
		week_prior_inds = df_dates[mask].index.values

		if(week_prior_inds.size):
			physicalDistance = 100*np.sqrt((df_lats[week_prior_inds]-reducedLatitudes[i])**2 + (df_lons[week_prior_inds]-reducedLongitudes[i])**2)
			daysBack = (searchdate - df_dates['Sample Date'][week_prior_inds]).astype('timedelta64[D]').values
			totalDistance = physicalDistance + beta*daysBack
			inverseDistance = 1/totalDistance
			NN_weights = inverseDistance/np.sum(inverseDistance)
			closestClasses = df_classes[week_prior_inds]
			negativeInds = np.where(closestClasses==0)[0]
			positiveInds = np.where(closestClasses==1)[0]

			idx = find_nearest_latlon(df_lats[week_prior_inds], df_lons[week_prior_inds], reducedLatitudes[i], reducedLongitudes[i])

			closestConc = df_concs[week_prior_inds][idx]
			nn_concs[i] = closestConc
			knn_concs[i] = np.sum(NN_weights*df_concs_log[week_prior_inds])
			if(closestConc < 100000):
				nn_preds[i] = 0
			else:
				nn_preds[i] = 1
			if(np.sum(NN_weights[negativeInds]) > np.sum(NN_weights[positiveInds])):
				knn_preds[i] = 0
			else:
				knn_preds[i] = 1
		else:
			nn_concs[i] = 0
			nn_preds[i] = 0
			knn_preds[i] = 0

	accs[model_number] = accuracy_score(testClasses, np.argmax(output, axis=1))
	confusonMatrixSum += confusion_matrix(testClasses, np.argmax(output, axis=1))
	accsLinLee[model_number] = accuracy_score(testClasses, outputLinLee)
	accsNN[model_number] = accuracy_score(testClasses, nn_preds)

	false_positives = 0
	true_positives = 0
	total_negatives = 0
	total_positives = 0

	for i in range(len(testClasses)):
		if(testClasses[i] == 0):
			if(outputLinLee[i] != 0):
				false_positives += 1
			total_negatives += 1
		else:
			if(outputLinLee[i] == 1):
				true_positives += 1
			total_positives += 1

	fpr = false_positives/total_negatives
	tpr = true_positives/total_positives
	fprsSoto.append(fpr)
	tprsSoto.append(tpr)

	false_positives = 0
	true_positives = 0
	total_negatives = 0
	total_positives = 0

	for i in range(len(testClasses)):
		if(testClasses[i] == 0):
			if(outputAmin[i] != 0):
				false_positives += 1
			total_negatives += 1
		else:
			if(outputAmin[i] == 1):
				true_positives += 1
			total_positives += 1

	fpr = false_positives/total_negatives
	tpr = true_positives/total_positives
	fprsAmin.append(fpr)
	tprsAmin.append(tpr)

	#false_positives = 0
	#true_positives = 0
	#total_negatives = 0
	#total_positives = 0

	#for i in range(len(testClasses)):
	#	if(testClasses[i] == 0):
	#		if(outputCannizzaro[i] != 0):
	#			false_positives += 1
	#		total_negatives += 1
	#	else:
	#		if(outputCannizzaro[i] == 1):
	#			true_positives += 1
	#		total_positives += 1

	#fpr = false_positives/total_negatives
	#tpr = true_positives/total_positives
	#fprsCannizzaro.append(fpr)
	#tprsCannizzaro.append(tpr)

	fpr, tpr, thresholds = roc_curve(testClasses, output[:, 1])
	if(model_number == 0):
		refFpr = fpr
		tprs = tpr
		tprs = np.expand_dims(tprs, axis=1)
	else:
		refTpr = convertROC(fpr, tpr, refFpr)
		refTpr = np.expand_dims(refTpr, axis=1)
		tprs = np.concatenate((tprs, refTpr), axis=1)

	fpr, tpr, thresholds = roc_curve(testClasses, nn_concs)
	if(model_number == 0):
		refFprNN = fpr
		tprsNN = tpr
		tprsNN = np.expand_dims(tprsNN, axis=1)
	else:
		refTprNN = convertROC(fpr, tpr, refFprNN)
		refTprNN = np.expand_dims(refTprNN, axis=1)
		tprsNN = np.concatenate((tprsNN, refTprNN), axis=1)

	fpr, tpr, thresholds = roc_curve(testClasses, knn_concs)
	if(model_number == 0):
		refFprKNN = fpr
		tprsKNN = tpr
		tprsKNN = np.expand_dims(tprsKNN, axis=1)
	else:
		refTprKNN = convertROC(fpr, tpr, refFprKNN)
		refTprKNN = np.expand_dims(refTprKNN, axis=1)
		tprsKNN = np.concatenate((tprsKNN, refTprKNN), axis=1)

	fpr, tpr, thresholds = roc_curve(testClasses, outputStumpf)
	if(model_number == 0):
		refFprStumpf = fpr
		tprsStumpf = tpr
		tprsStumpf = np.expand_dims(tprsStumpf, axis=1)
	else:
		refTprStumpf = convertROC(fpr, tpr, refFprStumpf)
		refTprStumpf = np.expand_dims(refTprStumpf, axis=1)
		tprsStumpf = np.concatenate((tprsStumpf, refTprStumpf), axis=1)

	fpr, tpr, thresholds = roc_curve(testClasses, outputShehhi)
	if(model_number == 0):
		refFprShehhi = fpr
		tprsShehhi = tpr
		tprsShehhi = np.expand_dims(tprsShehhi, axis=1)
	else:
		refTprShehhi = convertROC(fpr, tpr, refFprShehhi)
		refTprShehhi = np.expand_dims(refTprShehhi, axis=1)
		tprsShehhi = np.concatenate((tprsShehhi, refTprShehhi), axis=1)

	fpr, tpr, thresholds = roc_curve(testClasses, outputSS488)
	if(model_number == 0):
		refFprSS488 = fpr
		tprsSS488 = tpr
		tprsSS488 = np.expand_dims(tprsSS488, axis=1)
	else:
		refTprSS488 = convertROC(fpr, tpr, refFprSS488)
		refTprSS488 = np.expand_dims(refTprSS488, axis=1)
		tprsSS488 = np.concatenate((tprsSS488, refTprSS488), axis=1)

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

filename_roc_curve_info = 'roc_curve_info'

refFpr = np.expand_dims(refFpr, axis=1)
fpr_and_tprs = np.concatenate((refFpr, tprs), axis=1)

np.save(filename_roc_curve_info+'/'+configfilename+'.npy', fpr_and_tprs)

refFprNN = np.expand_dims(refFprNN, axis=1)
fpr_and_tprsNN = np.concatenate((refFprNN, tprsNN), axis=1)

np.save(filename_roc_curve_info+'/'+configfilename.split('_')[0]+'_NN.npy', fpr_and_tprsNN)

refFprKNN = np.expand_dims(refFprKNN, axis=1)
fpr_and_tprsKNN = np.concatenate((refFprKNN, tprsKNN), axis=1)

np.save(filename_roc_curve_info+'/'+configfilename.split('_')[0]+'_KNN.npy', fpr_and_tprsKNN)

refFprStumpf = np.expand_dims(refFprStumpf, axis=1)
fpr_and_tprsStumpf = np.concatenate((refFprStumpf, tprsStumpf), axis=1)

np.save(filename_roc_curve_info+'/'+configfilename.split('_')[0]+'_Stumpf.npy', fpr_and_tprsStumpf)

refFprShehhi = np.expand_dims(refFprShehhi, axis=1)
fpr_and_tprsShehhi = np.concatenate((refFprShehhi, tprsShehhi), axis=1)

np.save(filename_roc_curve_info+'/'+configfilename.split('_')[0]+'_Shehhi.npy', fpr_and_tprsShehhi)

refFprSS488 = np.expand_dims(refFprSS488, axis=1)
fpr_and_tprsSS488 = np.concatenate((refFprSS488, tprsSS488), axis=1)

np.save(filename_roc_curve_info+'/'+configfilename.split('_')[0]+'_SS488.npy', fpr_and_tprsSS488)

fpr_and_tprsSoto = np.zeros(2)
fpr_and_tprsSoto[0] = np.mean(fprsSoto)
fpr_and_tprsSoto[1] = np.mean(tprsSoto)

np.save(filename_roc_curve_info+'/'+configfilename.split('_')[0]+'_Soto.npy', fpr_and_tprsSoto)

fpr_and_tprsAmin = np.zeros(2)
fpr_and_tprsAmin[0] = np.mean(fprsAmin)
fpr_and_tprsAmin[1] = np.mean(tprsAmin)

np.save(filename_roc_curve_info+'/'+configfilename.split('_')[0]+'_Amin.npy', fpr_and_tprsAmin)

#fpr_and_tprsCannizzaro = np.zeros(2)
#fpr_and_tprsCannizzaro[0] = np.mean(fprsCannizzaro)
#fpr_and_tprsCannizzaro[1] = np.mean(tprsCannizzaro)

#np.save(filename_roc_curve_info+'/'+configfilename.split('_')[0]+'_Cannizzaro.npy', fpr_and_tprsCannizzaro)