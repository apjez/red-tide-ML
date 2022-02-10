import numpy as np
import sys
import pandas as pd
import xarray as xr
import os
import torch
import netCDF4
import json
import matplotlib.pyplot as plt
import datetime as dt
import time
from configparser import ConfigParser
from convertFeaturesByDepth import *
from findMatrixCoordsBedrock import *
from model import *
from utils import *

testDate = pd.Timestamp(year=2016, month=10, day=10, hour=0)

step_size = 0.015
florida_x = np.arange(-92, -75, step_size)
florida_y = np.arange(20, 35, step_size)
# 'angstrom', 'chlor_a', 'chl_ocx', 'Kd_490', 'poc', 'nflh', count
florida_stats = np.zeros((florida_x.shape[0], florida_y.shape[0], 7))
florida_lats = np.tile(florida_y, len(florida_x))
florida_lons = np.repeat(florida_x, len(florida_y))

data_folder = '/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/requested_files'
data_list = os.listdir(data_folder)

reduced_file_list = []
for i in range(len(data_list)):
	year = int(data_list[i][1:5])
	if(abs(testDate.year - year) < 1.5):
		reduced_file_list.append(data_list[i])

for i in range(len(reduced_file_list)):
	file_id = reduced_file_list[i]
	file_path = data_folder + '/' + file_id

	fh = netCDF4.Dataset(file_path, mode='r')
	collectionDate = fh.time_coverage_start[0:10]
	collectionTimeStamp = pd.Timestamp(int(collectionDate[0:4]), int(collectionDate[5:7]), int(collectionDate[8:10]), 0)

	if((testDate - collectionTimeStamp).days <= 8 and (testDate - collectionTimeStamp).days >= 0):
		dataset = xr.open_dataset(file_path, 'geophysical_data')
		nav_dataset = xr.open_dataset(file_path, 'navigation_data')
		latitude = nav_dataset['latitude']
		longitude = nav_dataset['longitude']
		latarr = np.array(latitude).flatten()
		longarr = np.array(longitude).flatten()

		angstrom = np.array(dataset['angstrom']).flatten()
		chlor_a = np.array(dataset['chlor_a']).flatten()
		chl_ocx = np.array(dataset['chl_ocx']).flatten()
		Kd_490 = np.array(dataset['Kd_490']).flatten()
		poc = np.array(dataset['poc']).flatten()
		nflh = np.array(dataset['nflh']).flatten()

		avail_inds = ~np.isnan(angstrom)
		x_ind = find_nearest_batch(florida_x, longarr[avail_inds])
		y_ind = find_nearest_batch(florida_y, latarr[avail_inds])
		florida_stats[x_ind, y_ind, 0] += angstrom[avail_inds]

		avail_inds = ~np.isnan(chlor_a)
		x_ind = find_nearest_batch(florida_x, longarr[avail_inds])
		y_ind = find_nearest_batch(florida_y, latarr[avail_inds])
		florida_stats[x_ind, y_ind, 1] += chlor_a[avail_inds]

		avail_inds = ~np.isnan(chl_ocx)
		x_ind = find_nearest_batch(florida_x, longarr[avail_inds])
		y_ind = find_nearest_batch(florida_y, latarr[avail_inds])
		florida_stats[x_ind, y_ind, 2] += chl_ocx[avail_inds]

		avail_inds = ~np.isnan(Kd_490)
		x_ind = find_nearest_batch(florida_x, longarr[avail_inds])
		y_ind = find_nearest_batch(florida_y, latarr[avail_inds])
		florida_stats[x_ind, y_ind, 3] += Kd_490[avail_inds]

		avail_inds = ~np.isnan(poc)
		x_ind = find_nearest_batch(florida_x, longarr[avail_inds])
		y_ind = find_nearest_batch(florida_y, latarr[avail_inds])
		florida_stats[x_ind, y_ind, 4] += poc[avail_inds]

		avail_inds = ~np.isnan(nflh)
		x_ind = find_nearest_batch(florida_x, longarr[avail_inds])
		y_ind = find_nearest_batch(florida_y, latarr[avail_inds])
		florida_stats[x_ind, y_ind, 5] += nflh[avail_inds]

		florida_stats[x_ind, y_ind, 6] += 1
	
features = np.zeros((florida_stats.shape[0], florida_stats.shape[1], 7))
for i in range(florida_stats.shape[0]):
	for j in range(florida_stats.shape[1]):
		if(florida_stats[i, j, 6] == 0):
			features[i, j, :] = -1
		else:
			features[i, j, 0] = florida_stats[i, j, 0]/florida_stats[i, j, 6]
			features[i, j, 1] = florida_stats[i, j, 1]/florida_stats[i, j, 6]
			features[i, j, 2] = florida_stats[i, j, 2]/florida_stats[i, j, 6]
			features[i, j, 3] = florida_stats[i, j, 3]/florida_stats[i, j, 6]
			features[i, j, 4] = florida_stats[i, j, 4]/florida_stats[i, j, 6]
			features[i, j, 5] = florida_stats[i, j, 5]/florida_stats[i, j, 6]

bedrock_x = np.load('florida_x.npy')
bedrock_y = np.load('florida_y.npy')
bedrock_z = np.load('florida_z.npy')

#Reduce data to only Southwest Florida
original_size = features.shape
florida_lats = np.reshape(florida_lats, (features.shape[0], features.shape[1]), order='C')
florida_lons = np.reshape(florida_lons, (features.shape[0], features.shape[1]), order='C')
florida_lats = florida_lats[520:780, 300:580]
florida_lons = florida_lons[520:780, 300:580]
florida_lats = np.reshape(florida_lats, (florida_lats.shape[0]*florida_lats.shape[1]))
florida_lons = np.reshape(florida_lons, (florida_lons.shape[0]*florida_lons.shape[1]))
features = features[520:780, 300:580, :]
original_size = features.shape


features = np.reshape(features, (features.shape[0]*features.shape[1], 7))

orig_indices_bedrock = findMatrixCoordsBedrock(bedrock_y, bedrock_x, florida_lats, florida_lons)

for i in range(len(orig_indices_bedrock)):
	features[i, 6] = bedrock_z[orig_indices_bedrock[i][0]][orig_indices_bedrock[i][1]]

land_mask = features[:,6] > 0
land_mask = np.reshape(land_mask, (original_size[0], original_size[1]))

features_to_use = ['angstrom', 'chlor_a', 'chl_ocx', 'Kd_490', 'poc', 'nflh']
features = convertFeaturesByDepth(features, features_to_use)
featureTensor = torch.tensor(features).float().cuda()
features = np.reshape(features, (original_size[0], original_size[1], 6), order='C')

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

# 0 = No nn features, 1 = nn, 2 = weighted knn
if(use_nn_feature == 1 or use_nn_feature == 2):
	file_path = 'PinellasMonroeCoKareniabrevis 2010-2020.06.12.xlsx'

	df = pd.read_excel(file_path, engine='openpyxl')
	df_dates = df['Sample Date']
	df_lats = df['Latitude'].to_numpy()
	df_lons = df['Longitude'].to_numpy()
	df_concs = df['Karenia brevis abundance (cells/L)'].to_numpy()

	df_concs_log = np.log10(df_concs)/np.max(np.log10(df_concs))
	df_concs_log[np.isinf(df_concs_log)] = 0

	knn_features = np.zeros((featureTensor.shape[0], 1))

	searchdate = testDate
	weekbefore = searchdate - dt.timedelta(days=3)
	twoweeksbefore = searchdate - dt.timedelta(days=10)
	mask = (df_dates > twoweeksbefore) & (df_dates <= weekbefore)
	week_prior_inds = df_dates[mask].index.values

	beta = 1

	if(week_prior_inds.size):
		#Do some nearest neighbor thing with the last week's samples
		for i in range(len(knn_features)):
			physicalDistance = 100*np.sqrt((df_lats[week_prior_inds]-florida_lats[i])**2 + (df_lons[week_prior_inds]-florida_lons[i])**2)
			daysBack = (searchdate - df_dates[week_prior_inds]).astype('timedelta64[D]').values
			totalDistance = physicalDistance + beta*daysBack
			inverseDistance = 1/totalDistance
			NN_weights = inverseDistance/np.sum(inverseDistance)

			knn_features[i] = np.sum(NN_weights*df_concs_log[week_prior_inds])

	knn_features_map = np.reshape(knn_features, (original_size[0], original_size[1]), order='C')

	knn_features_map[land_mask] = -1

	plt.figure(dpi=500)
	plt.imshow(knn_features_map.T)
	plt.colorbar()
	plt.title('Red Tide Prediction {}/{}/{}'.format(testDate.month, testDate.day, testDate.year))
	plt.gca().invert_yaxis()
	plt.savefig('red_tide_knn{}_{}_{}.png'.format(testDate.month, testDate.day, testDate.year), bbox_inches='tight')

	knn_features_map.fill(0)
	knn_features_map[land_mask] = -1

	knn_features_map = knn_features_map[125:160, 120:150]

	plt.figure(dpi=500)
	plt.imshow(knn_features_map.T)
	plt.clim(-1, 1)
	plt.title('Red Tide KNN {}/{}/{}'.format(testDate.month, testDate.day, testDate.year))
	plt.gca().invert_yaxis()
	fig = plt.gcf()
	ax = fig.gca()

	florida_lats = np.reshape(florida_lats, (features.shape[0], features.shape[1]), order='C')
	florida_lons = np.reshape(florida_lons, (features.shape[0], features.shape[1]), order='C')
	florida_lats = florida_lats[125:160, 120:150]
	florida_lons = florida_lons[125:160, 120:150]
	florida_lats = florida_lats[0, :]
	florida_lons = florida_lons[:, 0]

	for i in range(week_prior_inds.size):
		lat_ind = find_nearest(florida_lats, df_lats[week_prior_inds[i]])
		lon_ind = find_nearest(florida_lons, df_lons[week_prior_inds[i]])
		if(min(florida_lats) < df_lats[week_prior_inds[i]] and max(florida_lats) > df_lats[week_prior_inds[i]] \
		   and min(florida_lons) < df_lons[week_prior_inds[i]] and max(florida_lons) > df_lons[week_prior_inds[i]]):
			daysback = (testDate - df_dates[week_prior_inds[i]]).days
			opacity = ((7-daysback)+1)/8
			if(df_concs[week_prior_inds[i]] < 1000):
				color = (0.5, 0.5, 0.5, opacity)
			elif(df_concs[week_prior_inds[i]] > 1000 and df_concs[week_prior_inds[i]] < 10000):
				color = (1, 1, 1, opacity)
			elif(df_concs[week_prior_inds[i]] > 10000 and df_concs[week_prior_inds[i]] < 100000):
				color = (1, 1, 0, opacity)
			elif(df_concs[week_prior_inds[i]] > 100000 and df_concs[week_prior_inds[i]] < 1000000):
				color = (1, 0.65, 0, opacity)
			else:
				color = (1, 0, 0, opacity)

			radius = 1
			circle = plt.Circle((lon_ind, lat_ind), radius, color=color)
			ax.add_patch(circle)
	rectangle = plt.Rectangle((24, 4), 2, 2, color=(0.5, 0.5, 0.5, 1))
	ax.add_patch(rectangle)

	plt.savefig('red_tide_knn2{}_{}_{}.png'.format(testDate.month, testDate.day, testDate.year), bbox_inches='tight')

	featureTensor = torch.cat((featureTensor, torch.tensor(knn_features).float().cuda()), dim=1)

red_tide_output_sum = np.zeros((original_size[0], original_size[1]))

for model_number in range(len(randomseeds)):
	predictor = Predictor(featureTensor.shape[1], num_classes).cuda()
	predictor.load_state_dict(torch.load('saved_model_info/'+configfilename+'/predictor{}.pt'.format(model_number)))
	predictor.eval()

	output = predictor(featureTensor)

	red_tide = output[:, 1].detach().cpu().numpy()
	red_tide = np.reshape(red_tide, (original_size[0], original_size[1]), order='C')

	red_tide[land_mask] = -1

	red_tide_output_sum = red_tide_output_sum + red_tide

red_tide_output = red_tide_output_sum/len(randomseeds)

plt.figure(dpi=500)
plt.imshow(red_tide_output.T)
plt.colorbar()
plt.title('Red Tide Prediction {}/{}/{}'.format(testDate.month, testDate.day, testDate.year))
plt.gca().invert_yaxis()
plt.savefig('red_tide_combined{}_{}_{}.png'.format(testDate.month, testDate.day, testDate.year), bbox_inches='tight')