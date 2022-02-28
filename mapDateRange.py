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
from scipy import ndimage
from configparser import ConfigParser
from convertFeaturesByDepth import *
from findMatrixCoordsBedrock import *
from model import *
from utils import *

startDate = pd.Timestamp(year=2018, month=8, day=1, hour=0)
endDate = pd.Timestamp(year=2018, month=9, day=1, hour=0)

dates = []

testDate = startDate
while(testDate < endDate):
	dates.append(testDate)
	testDate = testDate + pd.Timedelta(days=1)

np.save('map_images/dates.npy', dates)

step_size = 0.015
florida_x = np.arange(-92, -75, step_size)
florida_y = np.arange(20, 35, step_size)
# 'par', count, 'Kd_490', count, 'chlor_a', count, 'Rrs_443', count, 'Rrs_469', count, 'Rrs_488', count, 'nflh', count
florida_stats = np.zeros((florida_x.shape[0], florida_y.shape[0], 14, len(dates)))
florida_lats = np.tile(florida_y, len(florida_x))
florida_lons = np.repeat(florida_x, len(florida_y))

data_folder = '/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/requested_files'
data_list = os.listdir(data_folder)

reduced_file_list = []
for i in range(len(data_list)):
	year = int(data_list[i][1:5])
	if(abs(startDate.year - year) < 1.5 or abs(endDate.year - year) < 1.5):
		reduced_file_list.append(data_list[i])

for i in range(len(reduced_file_list)):
	print('Processing files: {}/{}'.format(i, len(reduced_file_list)))

	file_id = reduced_file_list[i]
	file_path = data_folder + '/' + file_id

	fh = netCDF4.Dataset(file_path, mode='r')
	collectionDate = fh.time_coverage_start[0:10]
	collectionTimeStamp = pd.Timestamp(int(collectionDate[0:4]), int(collectionDate[5:7]), int(collectionDate[8:10]), 0)

	days_ahead = [(date_temp - collectionTimeStamp).days for date_temp in dates]

	if(any(day_ahead >= 0 and day_ahead <= 14 for day_ahead in days_ahead)):
		dataset = xr.open_dataset(file_path, 'geophysical_data')
		nav_dataset = xr.open_dataset(file_path, 'navigation_data')
		latitude = nav_dataset['latitude']
		longitude = nav_dataset['longitude']
		latarr = np.array(latitude).flatten()
		longarr = np.array(longitude).flatten()

		par = np.array(dataset['par']).flatten()
		Kd_490 = np.array(dataset['Kd_490']).flatten()
		chlor_a = np.array(dataset['chlor_a']).flatten()
		Rrs_443 = np.array(dataset['Rrs_443']).flatten()
		Rrs_469 = np.array(dataset['Rrs_469']).flatten()
		Rrs_488 = np.array(dataset['Rrs_488']).flatten()
		nflh = np.array(dataset['nflh']).flatten()

		day_inds = [day_ahead >= 0 and day_ahead <= 14 for day_ahead in days_ahead]
		day_inds = np.where(np.array(day_inds) == True)[0]

		avail_inds = ~np.isnan(par)
		x_ind = find_nearest_batch(florida_x, longarr[avail_inds])
		y_ind = find_nearest_batch(florida_y, latarr[avail_inds])
		florida_stats[np.tile(x_ind, day_inds.shape[0]), np.tile(y_ind, day_inds.shape[0]), 0, np.repeat(day_inds, x_ind.shape[0])] += np.tile(par[avail_inds], day_inds.shape[0])

		florida_stats[np.tile(x_ind, day_inds.shape[0]), np.tile(y_ind, day_inds.shape[0]), 1, np.repeat(day_inds, x_ind.shape[0])] += 1

		avail_inds = ~np.isnan(Kd_490)
		x_ind = find_nearest_batch(florida_x, longarr[avail_inds])
		y_ind = find_nearest_batch(florida_y, latarr[avail_inds])
		florida_stats[np.tile(x_ind, day_inds.shape[0]), np.tile(y_ind, day_inds.shape[0]), 2, np.repeat(day_inds, x_ind.shape[0])] += np.tile(Kd_490[avail_inds], day_inds.shape[0])

		florida_stats[np.tile(x_ind, day_inds.shape[0]), np.tile(y_ind, day_inds.shape[0]), 3, np.repeat(day_inds, x_ind.shape[0])] += 1

		avail_inds = ~np.isnan(chlor_a)
		x_ind = find_nearest_batch(florida_x, longarr[avail_inds])
		y_ind = find_nearest_batch(florida_y, latarr[avail_inds])
		florida_stats[np.tile(x_ind, day_inds.shape[0]), np.tile(y_ind, day_inds.shape[0]), 4, np.repeat(day_inds, x_ind.shape[0])] += np.tile(chlor_a[avail_inds], day_inds.shape[0])

		florida_stats[np.tile(x_ind, day_inds.shape[0]), np.tile(y_ind, day_inds.shape[0]), 5, np.repeat(day_inds, x_ind.shape[0])] += 1

		avail_inds = ~np.isnan(Rrs_443)
		x_ind = find_nearest_batch(florida_x, longarr[avail_inds])
		y_ind = find_nearest_batch(florida_y, latarr[avail_inds])
		florida_stats[np.tile(x_ind, day_inds.shape[0]), np.tile(y_ind, day_inds.shape[0]), 6, np.repeat(day_inds, x_ind.shape[0])] += np.tile(Rrs_443[avail_inds], day_inds.shape[0])

		florida_stats[np.tile(x_ind, day_inds.shape[0]), np.tile(y_ind, day_inds.shape[0]), 7, np.repeat(day_inds, x_ind.shape[0])] += 1

		avail_inds = ~np.isnan(Rrs_469)
		x_ind = find_nearest_batch(florida_x, longarr[avail_inds])
		y_ind = find_nearest_batch(florida_y, latarr[avail_inds])
		florida_stats[np.tile(x_ind, day_inds.shape[0]), np.tile(y_ind, day_inds.shape[0]), 8, np.repeat(day_inds, x_ind.shape[0])] += np.tile(Rrs_469[avail_inds], day_inds.shape[0])

		florida_stats[np.tile(x_ind, day_inds.shape[0]), np.tile(y_ind, day_inds.shape[0]), 9, np.repeat(day_inds, x_ind.shape[0])] += 1

		avail_inds = ~np.isnan(Rrs_488)
		x_ind = find_nearest_batch(florida_x, longarr[avail_inds])
		y_ind = find_nearest_batch(florida_y, latarr[avail_inds])
		florida_stats[np.tile(x_ind, day_inds.shape[0]), np.tile(y_ind, day_inds.shape[0]), 10, np.repeat(day_inds, x_ind.shape[0])] += np.tile(Rrs_488[avail_inds], day_inds.shape[0])

		florida_stats[np.tile(x_ind, day_inds.shape[0]), np.tile(y_ind, day_inds.shape[0]), 11, np.repeat(day_inds, x_ind.shape[0])] += 1

		avail_inds = ~np.isnan(nflh)
		x_ind = find_nearest_batch(florida_x, longarr[avail_inds])
		y_ind = find_nearest_batch(florida_y, latarr[avail_inds])
		florida_stats[np.tile(x_ind, day_inds.shape[0]), np.tile(y_ind, day_inds.shape[0]), 12, np.repeat(day_inds, x_ind.shape[0])] += np.tile(nflh[avail_inds], day_inds.shape[0])

		florida_stats[np.tile(x_ind, day_inds.shape[0]), np.tile(y_ind, day_inds.shape[0]), 13, np.repeat(day_inds, x_ind.shape[0])] += 1

features = np.zeros((florida_stats.shape[0], florida_stats.shape[1], 8, len(dates)))
for i in range(florida_stats.shape[0]):
	for j in range(florida_stats.shape[1]):
		for k in range(len(dates)):
			if(florida_stats[i, j, 1, k] == 0):
				features[i, j, 0, k] = -1
			else:
				features[i, j, 0, k] = florida_stats[i, j, 0, k]/florida_stats[i, j, 1, k]

			if(florida_stats[i, j, 3, k] == 0):
				features[i, j, 1, k] = -1
			else:
				features[i, j, 1, k] = florida_stats[i, j, 2, k]/florida_stats[i, j, 3, k]

			if(florida_stats[i, j, 5, k] == 0):
				features[i, j, 2, k] = -1
			else:
				features[i, j, 2, k] = florida_stats[i, j, 4, k]/florida_stats[i, j, 5, k]

			if(florida_stats[i, j, 7, k] == 0):
				features[i, j, 3, k] = -1
			else:
				features[i, j, 3, k] = florida_stats[i, j, 6, k]/florida_stats[i, j, 7, k]

			if(florida_stats[i, j, 9, k] == 0):
				features[i, j, 4, k] = -1
			else:
				features[i, j, 4, k] = florida_stats[i, j, 8, k]/florida_stats[i, j, 9, k]

			if(florida_stats[i, j, 11, k] == 0):
				features[i, j, 5, k] = -1
			else:
				features[i, j, 5, k] = florida_stats[i, j, 10, k]/florida_stats[i, j, 11, k]

			if(florida_stats[i, j, 13, k] == 0):
				features[i, j, 6, k] = -1
			else:
				features[i, j, 6, k] = florida_stats[i, j, 12, k]/florida_stats[i, j, 13, k]



###########
#Try some average filtering to smooth out noise
#for feature_i in range(features.shape[2]):
#	for date_j in range(features.shape[3]):
#		features[:, :, feature_i, date_j] = ndimage.uniform_filter(np.squeeze(features[:, :, feature_i, date_j]), size=3)
###########


bedrock_x = np.load('florida_x.npy')
bedrock_y = np.load('florida_y.npy')
bedrock_z = np.load('florida_z.npy')

#Reduce data to only Southwest Florida
original_size = features.shape
florida_lats = np.reshape(florida_lats, (features.shape[0], features.shape[1]), order='C')
florida_lons = np.reshape(florida_lons, (features.shape[0], features.shape[1]), order='C')
florida_lats = florida_lats[520:780, 300:580]
florida_lons = florida_lons[520:780, 300:580]

np.save('map_images/latitudes.npy', florida_lats)
np.save('map_images/longitudes.npy', florida_lons)

florida_lats = np.reshape(florida_lats, (florida_lats.shape[0]*florida_lats.shape[1]))
florida_lons = np.reshape(florida_lons, (florida_lons.shape[0]*florida_lons.shape[1]))
features = features[520:780, 300:580, :, :]
original_size = features.shape

features = np.reshape(features, (features.shape[0]*features.shape[1], 8, len(dates)))

orig_indices_bedrock = findMatrixCoordsBedrock(bedrock_y, bedrock_x, florida_lats, florida_lons)

for i in range(len(orig_indices_bedrock)):
	for j in range(len(dates)):
		features[i, 7, j] = bedrock_z[orig_indices_bedrock[i][0]][orig_indices_bedrock[i][1]]

land_mask = features[:,7,0] > 0
land_mask = np.reshape(land_mask, (original_size[0], original_size[1]))

day_counter = 0
for day in dates:
	print('Processing days: {}/{}'.format(day_counter, len(dates)))

	day_features = np.squeeze(features[:, :, day_counter])

	features_to_use = ['par', 'Kd_490', 'chlor_a', 'Rrs_443', 'Rrs_469', 'Rrs_488', 'nflh']
	day_features = convertFeaturesByDepth(day_features, features_to_use)

	featureTensor = torch.tensor(day_features).float().cuda()
	#day_features = np.reshape(day_features, (original_size[0], original_size[1], 6), order='C')

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
	if(use_nn_feature == 1 or use_nn_feature == 2 or use_nn_feature == 3):
		file_path = 'PinellasMonroeCoKareniabrevis 2010-2020.06.12.xlsx'

		df = pd.read_excel(file_path, engine='openpyxl')
		df_dates = df['Sample Date']
		df_lats = df['Latitude'].to_numpy()
		df_lons = df['Longitude'].to_numpy()
		df_concs = df['Karenia brevis abundance (cells/L)'].to_numpy()

		df_concs_log = np.log10(df_concs)/np.max(np.log10(df_concs))
		df_concs_log[np.isinf(df_concs_log)] = 0

		knn_features = np.zeros((featureTensor.shape[0], 1))

		searchdate = day
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

		featureTensorOriginal = featureTensor.clone()
		featureTensor = torch.cat((featureTensor, torch.tensor(knn_features).float().cuda()), dim=1)

	red_tide_output_sum_original = np.zeros((original_size[0], original_size[1]))
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

		if(use_nn_feature == 3):
			originalPredictor = Predictor(featureTensorOriginal.shape[1], num_classes).cuda()
			originalPredictor.load_state_dict(torch.load('saved_model_info/'+configfilename+'/originalPredictor{}.pt'.format(model_number)))
			originalPredictor.eval()

			output = originalPredictor(featureTensorOriginal)

			red_tide = output[:, 1].detach().cpu().numpy()
			red_tide = np.reshape(red_tide, (original_size[0], original_size[1]), order='C')

			red_tide[land_mask] = -1

			red_tide_output_sum_original = red_tide_output_sum_original + red_tide

	#red_tide_output_original = red_tide_output_sum_original/len(randomseeds)
	red_tide_output = red_tide_output_sum/len(randomseeds)

	#np.save('map_images/red_tide_output_original{}.png'.format(day_counter), red_tide_output_original)
	np.save('map_images/red_tide_output{}.npy'.format(day_counter), red_tide_output)

	#plt.figure(dpi=500)
	#plt.imshow(red_tide_output_original.T)
	#plt.clim(-1, 1)
	#plt.colorbar()
	#plt.title('Red Tide Prediction Original {}/{}/{}'.format(day.month, day.day, day.year))
	#plt.gca().invert_yaxis()
	#plt.savefig('map_images/red_tide_original_combined{}.png'.format(str(day_counter).zfill(5)), bbox_inches='tight')

	plt.figure(dpi=500)
	plt.imshow(red_tide_output.T)
	plt.clim(-1, 1)
	plt.colorbar()
	plt.title('Red Tide Prediction {}/{}/{}'.format(day.month, day.day, day.year))
	plt.gca().invert_yaxis()
	plt.savefig('map_images/red_tide_image{}.png'.format(str(day_counter).zfill(5)), bbox_inches='tight')

	day_counter = day_counter + 1