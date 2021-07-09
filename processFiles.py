import xarray as xr
import netCDF4
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial
from findMatrixCoords import *
from findMatrixCoordsBedrock import *
from utils import *

file_path = 'PinellasMonroeCoKareniabrevis 2010-2020.06.12.xlsx'

df = pd.read_excel(file_path, engine='openpyxl')
df_dates = df['Sample Date'].tolist()
df_depths = df['Sample Depth (m)'].tolist()
df_lats = df['Latitude'].tolist()
df_lons = df['Longitude'].tolist()
df_concs = df['Karenia brevis abundance (cells/L)']

bedrock = netCDF4.Dataset('ETOPO1_Bed_g_gmt4.grd')
bedrock_x = bedrock['x'][:]
bedrock_y = bedrock['y'][:]
bedrock_z = bedrock['z'][:]

data_folder = '/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/requested_files'
data_list = os.listdir(data_folder)

totalSamples = 0
foundSamples = 0

paired_dataset = []

for i in range(len(data_list)):
	file_id = data_list[i]
	file_path = data_folder + '/' + file_id

	fh = netCDF4.Dataset(file_path, mode='r')
	collectionDate = fh.time_coverage_start[0:10]
	collectionTimeStamp = pd.Timestamp(int(collectionDate[0:4]), int(collectionDate[5:7]), int(collectionDate[8:10]), 0)

	if(i % 100 == 0):
		print('Processing file #{}'.format(i))

	insitu_samples_inds = [j for j in range(len(df_dates)) if df_dates[j] == collectionTimeStamp]
	#insitu_samples_inds = [j for j in range(len(df_dates)) if df_dates[j] == collectionTimeStamp or df_dates[j] == collectionTimeStamp + pd.DateOffset(-1) or df_dates[j] == collectionTimeStamp + pd.DateOffset(1)]

	sample_lats = [df_lats[j] for j in insitu_samples_inds]
	sample_lons = [df_lons[j] for j in insitu_samples_inds]

	orig_indices = findMatrixCoords(file_path, sample_lats, sample_lons)

	orig_indices_bedrock = findMatrixCoordsBedrock(bedrock_y, bedrock_x, sample_lats, sample_lons)

	dataset = xr.open_dataset(file_path, 'geophysical_data')

	RRS_443 = dataset['Rrs_443']

	for j in range(len(orig_indices)):
		totalSamples = totalSamples + 1
		if(orig_indices[j][0] != -1 and orig_indices[j][1] != -1 and ~np.isnan(RRS_443[orig_indices[j][0], orig_indices[j][1]])):
			foundSamples = foundSamples + 1
			sample_info = [df_dates[insitu_samples_inds[j]], df_depths[insitu_samples_inds[j]], df_lats[insitu_samples_inds[j]],\
			df_lons[insitu_samples_inds[j]], df_concs[insitu_samples_inds[j]], dataset['aot_869'][orig_indices[j][0], orig_indices[j][1]].values.item(),\
			dataset['angstrom'][orig_indices[j][0], orig_indices[j][1]].values.item(), dataset['Rrs_412'][orig_indices[j][0], orig_indices[j][1]].values.item(),\
			dataset['Rrs_443'][orig_indices[j][0], orig_indices[j][1]].values.item(), dataset['Rrs_469'][orig_indices[j][0], orig_indices[j][1]].values.item(),\
			dataset['Rrs_488'][orig_indices[j][0], orig_indices[j][1]].values.item(), dataset['Rrs_531'][orig_indices[j][0], orig_indices[j][1]].values.item(),\
			dataset['Rrs_547'][orig_indices[j][0], orig_indices[j][1]].values.item(), dataset['Rrs_555'][orig_indices[j][0], orig_indices[j][1]].values.item(),\
			dataset['Rrs_645'][orig_indices[j][0], orig_indices[j][1]].values.item(), dataset['Rrs_667'][orig_indices[j][0], orig_indices[j][1]].values.item(),\
			dataset['Rrs_678'][orig_indices[j][0], orig_indices[j][1]].values.item(), dataset['chlor_a'][orig_indices[j][0], orig_indices[j][1]].values.item(),\
			dataset['chl_ocx'][orig_indices[j][0], orig_indices[j][1]].values.item(), dataset['Kd_490'][orig_indices[j][0], orig_indices[j][1]].values.item(),\
			dataset['pic'][orig_indices[j][0], orig_indices[j][1]].values.item(), dataset['poc'][orig_indices[j][0], orig_indices[j][1]].values.item(),\
			dataset['ipar'][orig_indices[j][0], orig_indices[j][1]].values.item(), dataset['nflh'][orig_indices[j][0], orig_indices[j][1]].values.item(),\
			dataset['par'][orig_indices[j][0], orig_indices[j][1]].values.item(), bedrock_z[orig_indices_bedrock[j][0]][orig_indices_bedrock[j][1]]]
			paired_dataset.append(sample_info)

		#if(orig_indices[j][0]-1 >= 0 and orig_indices[j][1]-1 >= 0):
		#	if(~np.isnan(RRS_443[orig_indices[j][0]-1, orig_indices[j][1]-1])):
		#		foundSamples = foundSamples + 1
		#if(orig_indices[j][0]-1 >= 0):
		#	if(~np.isnan(RRS_443[orig_indices[j][0]-1, orig_indices[j][1]])):
		#		foundSamples = foundSamples + 1
		#if(orig_indices[j][0]-1 >= 0 and orig_indices[j][1]+1 < RRS_443.shape[1]):
		#	if(~np.isnan(RRS_443[orig_indices[j][0]-1, orig_indices[j][1]+1])):
		#		foundSamples = foundSamples + 1
		#if(orig_indices[j][1]-1 >= 0):
		#	if(~np.isnan(RRS_443[orig_indices[j][0], orig_indices[j][1]-1])):
		#		foundSamples = foundSamples + 1
		#if(~np.isnan(RRS_443[orig_indices[j][0], orig_indices[j][1]])):
		#	foundSamples = foundSamples + 1
		#if(orig_indices[j][1]+1 < RRS_443.shape[1]):
		#	if(~np.isnan(RRS_443[orig_indices[j][0], orig_indices[j][1]+1])):
		#		foundSamples = foundSamples + 1
		#if(orig_indices[j][0]+1 < RRS_443.shape[0] and orig_indices[j][1]-1 >= 0):
		#	if(~np.isnan(RRS_443[orig_indices[j][0]+1, orig_indices[j][1]-1])):
		#		foundSamples = foundSamples + 1
		#if(orig_indices[j][0]+1 < RRS_443.shape[0]):
		#	if(~np.isnan(RRS_443[orig_indices[j][0]+1, orig_indices[j][1]])):
		#		foundSamples = foundSamples + 1
		#if(orig_indices[j][0]+1 < RRS_443.shape[0] and orig_indices[j][1]+1 < RRS_443.shape[1]):
		#	if(~np.isnan(RRS_443[orig_indices[j][0]+1, orig_indices[j][1]+1])):
		#		foundSamples = foundSamples + 1

paired_df = pd.DataFrame(paired_dataset, columns=['Sample Date', 'Sample Depth', 'Latitude', 'Longitude', 'Red Tide Concentration',\
	'aot_869', 'angstrom', 'Rrs_412', 'Rrs_443', 'Rrs_469', 'Rrs_488', 'Rrs_531', 'Rrs_547', 'Rrs_555', 'Rrs_645',\
	'Rrs_667', 'Rrs_678', 'chlor_a', 'chl_ocx', 'Kd_490', 'pic', 'poc', 'ipar', 'nflh', 'par', 'bedrock'])
paired_df.to_pickle('paired_dataset.pkl')