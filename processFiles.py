import xarray as xr
import netCDF4
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial
from findMatrixCoords import *
from utils import *

file_path = 'PinellasMonroeCoKareniabrevis 2010-2020.06.12.xlsx'

df = pd.read_excel(file_path)
df_dates = df['Sample Date'].tolist()
df_lats = df['Latitude'].tolist()
df_lons = df['Longitude'].tolist()

data_folder = 'modis-data'
data_list = os.listdir(data_folder)

totalSamples = 0
foundSamples = 0

for i in range(len(data_list)):
	file_id = data_list[i]
	file_path = data_folder + '/' + file_id

	fh = netCDF4.Dataset(file_path, mode='r')
	collectionDate = fh.time_coverage_start[0:10]
	collectionTimeStamp = pd.Timestamp(int(collectionDate[0:4]), int(collectionDate[5:7]), int(collectionDate[8:10]), 0)

	print('Processing ' + collectionDate + '...')

	insitu_samples_inds = [j for j in range(len(df_dates)) if df_dates[j] == collectionTimeStamp]
	#insitu_samples_inds = [j for j in range(len(df_dates)) if df_dates[j] == collectionTimeStamp or df_dates[j] == collectionTimeStamp + pd.DateOffset(-1) or df_dates[j] == collectionTimeStamp + pd.DateOffset(1)]

	sample_lats = [df_lats[j] for j in insitu_samples_inds]
	sample_lons = [df_lons[j] for j in insitu_samples_inds]

	orig_indices = findMatrixCoords(file_path, sample_lats, sample_lons)

	dataset = xr.open_dataset(file_path, 'geophysical_data')

	RRS_443 = dataset['Rrs_443']

	for j in range(len(orig_indices)):
		totalSamples = totalSamples + 1
		#if(~np.isnan(RRS_443[orig_indices[j][0], orig_indices[j][1]])):
		#	foundSamples = foundSamples + 1

		if(orig_indices[j][0]-1 >= 0 and orig_indices[j][1]-1 >= 0):
			if(~np.isnan(RRS_443[orig_indices[j][0]-1, orig_indices[j][1]-1])):
				foundSamples = foundSamples + 1
		if(orig_indices[j][0]-1 >= 0):
			if(~np.isnan(RRS_443[orig_indices[j][0]-1, orig_indices[j][1]])):
				foundSamples = foundSamples + 1
		if(orig_indices[j][0]-1 >= 0 and orig_indices[j][1]+1 < RRS_443.shape[1]):
			if(~np.isnan(RRS_443[orig_indices[j][0]-1, orig_indices[j][1]+1])):
				foundSamples = foundSamples + 1
		if(orig_indices[j][1]-1 >= 0):
			if(~np.isnan(RRS_443[orig_indices[j][0], orig_indices[j][1]-1])):
				foundSamples = foundSamples + 1
		if(~np.isnan(RRS_443[orig_indices[j][0], orig_indices[j][1]])):
			foundSamples = foundSamples + 1
		if(orig_indices[j][1]+1 < RRS_443.shape[1]):
			if(~np.isnan(RRS_443[orig_indices[j][0], orig_indices[j][1]+1])):
				foundSamples = foundSamples + 1
		if(orig_indices[j][0]+1 < RRS_443.shape[0] and orig_indices[j][1]-1 >= 0):
			if(~np.isnan(RRS_443[orig_indices[j][0]+1, orig_indices[j][1]-1])):
				foundSamples = foundSamples + 1
		if(orig_indices[j][0]+1 < RRS_443.shape[0]):
			if(~np.isnan(RRS_443[orig_indices[j][0]+1, orig_indices[j][1]])):
				foundSamples = foundSamples + 1
		if(orig_indices[j][0]+1 < RRS_443.shape[0] and orig_indices[j][1]+1 < RRS_443.shape[1]):
			if(~np.isnan(RRS_443[orig_indices[j][0]+1, orig_indices[j][1]+1])):
				foundSamples = foundSamples + 1

print('Total samples: {}'.format(totalSamples))
print('Found samples: {}'.format(foundSamples))