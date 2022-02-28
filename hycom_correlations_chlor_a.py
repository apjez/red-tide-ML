import datetime as dt
import pandas as pd
import numpy as np
import xarray as xr
import os
import netCDF4

def find_nearest(lat_array, lat_value, lon_array, lon_value):
	lat_array = np.asarray(lat_array)
	lon_array = np.asarray(lon_array)
	idx = (np.abs(lon_array - lon_value) + np.abs(lat_array - lat_value)).argmin()
	return idx

startDate = pd.Timestamp(year=2018, month=6, day=1, hour=0)
endDate = pd.Timestamp(year=2018, month=7, day=1, hour=0)

data_folder = '/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/requested_files'
data_list = os.listdir(data_folder)

reduced_file_list = []
for i in range(len(data_list)):
	year = int(data_list[i][1:5])
	if(abs(startDate.year - year) < 1.5 or abs(endDate.year - year) < 1.5):
		reduced_file_list.append(data_list[i])

file_path_list = []
date_list = []

for i in range(len(reduced_file_list)):
	print('Processing files: {}/{}'.format(i, len(reduced_file_list)))

	file_id = reduced_file_list[i]
	file_path = data_folder + '/' + file_id

	fh = netCDF4.Dataset(file_path, mode='r')
	collectionDate = fh.time_coverage_start[0:10]
	collectionTimeStamp = pd.Timestamp(int(collectionDate[0:4]), int(collectionDate[5:7]), int(collectionDate[8:10]), 0)

	if(collectionTimeStamp >= startDate and collectionTimeStamp < endDate):
		file_path_list.append(file_path)
		date_list.append(collectionTimeStamp)

li=[]
for i in range(len(date_list)):
	li.append([date_list[i],i])
li.sort()
sort_index = []
for x in li:
	sort_index.append(x[1])

date_list_sorted = [date_list[i] for i in sort_index]
file_path_list_sorted = [file_path_list[i] for i in sort_index]

lat_min = 25.5
lat_max = 28.5
lon_min = -84.6
lon_max = -81

num_valid_pixels = 0

pixel_chlor_a = np.load('/run/media/rfick/UF10/HYCOM/pixel_chlor_a.npy')
data_netcdf4 = netCDF4.Dataset('/run/media/rfick/UF10/HYCOM/hycom_correlations.nc', mode='r')

data_xarray = xr.open_dataset(xr.backends.NetCDF4DataStore(data_netcdf4))

outputdt = dt.timedelta(hours=1)
timerange = np.arange(np.nanmin(data_xarray['time'].values),
                      np.nanmax(data_xarray['time'].values)+np.timedelta64(outputdt), 
                      outputdt)  # timerange in nanoseconds

chlor_a_pairs = []

for i in range(len(file_path_list_sorted)):
	print('Processing files: {}/{}'.format(i, len(file_path_list_sorted)))

	nav_dataset = xr.open_dataset(file_path_list_sorted[i], 'navigation_data')

	latitude = nav_dataset['latitude']
	longitude = nav_dataset['longitude']
	latarr = np.array(latitude).flatten()
	longarr = np.array(longitude).flatten()

	inds_in_area = np.where((latarr > lat_min) & (latarr < lat_max) & (longarr > lon_min) & (longarr < lon_max))[0]

	if(inds_in_area.size != 0):
		fh = netCDF4.Dataset(file_path_list_sorted[i], mode='r')
		collectionDateTime = fh.time_coverage_start
		collectionDateTime = np.datetime64(collectionDateTime[0:19])

		timedifference = (timerange - collectionDateTime)/np.timedelta64(1, 's')
		closestTime = np.argmin(np.abs(timedifference))

		time_id = np.where(data_xarray['time'] == timerange[closestTime]) # Indices of the data where time = 0
		lon_values = data_xarray['lon'].values[time_id]
		lat_values = data_xarray['lat'].values[time_id]
		chlor_a_values = pixel_chlor_a[time_id[0]]

		latarr = latarr[inds_in_area]
		longarr = longarr[inds_in_area]

		dataset = xr.open_dataset(file_path_list_sorted[i], 'geophysical_data')
		chlor_a = np.array(dataset['chlor_a']).flatten()
		chlor_a = chlor_a[inds_in_area]

		for j in range(len(lon_values)):
			idx = find_nearest(latarr, lat_values[j], longarr, lon_values[j])
			
			chlor_a_new = chlor_a[idx]
			if(time_id[1][j] > 1):
				if(np.isnan(chlor_a_new) == False):
					chlor_a_pairs.append((time_id[1][j], np.abs(chlor_a_new-chlor_a_values[j])))
					#print('{} {}'.format(time_id[1][j], np.abs(chlor_a_new-chlor_a_values[j])))

np.save("/run/media/rfick/UF10/HYCOM/chlor_a_pairs.npy", chlor_a_pairs)