import netCDF4
import os
import pandas as pd
import xarray as xr
import time
import matplotlib.pyplot as plt
from findMatrixCoordsBedrock import *

def maskPoints(A_x, A_y, B_x, B_y, X, Y):
	return (B_x - A_x)*(Y - A_y) - (B_y - A_y)*(X - A_x)

florida_x = np.load('florida_x.npy')
florida_y = np.load('florida_y.npy')
florida_z = np.load('florida_z.npy')

data_folder = '/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/requested_files'
data_list = os.listdir(data_folder)

angstrom_sums = {}
chlor_a_sums = {}
chl_ocx_sums = {}
Kd_490_sums = {}
poc_sums = {}
nflh_sums = {}

for i in range(len(data_list)):
	file_id = data_list[i]
	file_path = data_folder + '/' + file_id

	nav_dataset = xr.open_dataset(file_path, 'navigation_data')

	latitude = nav_dataset['latitude']
	longitude = nav_dataset['longitude']

	latarr = np.array(latitude).flatten()
	longarr = np.array(longitude).flatten()

	# Limit data used to only data from the west coast of Florida
	values = maskPoints(-88.002*np.ones_like(longarr), 30.759*np.ones_like(latarr), -80.788*np.ones_like(longarr), 29.628*np.ones_like(latarr), longarr, latarr)
	values2 = maskPoints(-82.949*np.ones_like(longarr), 30.262*np.ones_like(latarr), -80.349*np.ones_like(longarr), 24.833*np.ones_like(latarr), longarr, latarr)
	inds = np.logical_and(values < 0, values2 < 0)
	reducedLatArr = latarr[inds]
	reducedLongArr = longarr[inds]
	reducedLatArr = np.expand_dims(reducedLatArr, axis=1)
	reducedLongArr = np.expand_dims(reducedLongArr, axis=1)

	fh = netCDF4.Dataset(file_path, mode='r')
	collectionDate = fh.time_coverage_start[0:10]
	collectionTimeStamp = pd.Timestamp(int(collectionDate[0:4]), int(collectionDate[5:7]), int(collectionDate[8:10]), 0)

	if(i % 100 == 0):
		print('Processing file #{}'.format(i))

	start = time.time()
	orig_indices = findMatrixCoordsBedrock(florida_y, florida_x, reducedLatArr, reducedLongArr)
	end = time.time()

	dataset = xr.open_dataset(file_path, 'geophysical_data')
	angstrom = dataset['angstrom']
	angstrom = np.array(angstrom).flatten()
	reducedAngstrom = angstrom[inds]
	chlor_a = dataset['chlor_a']
	chlor_a = np.array(chlor_a).flatten()
	reducedChlorA = chlor_a[inds]
	chl_ocx = dataset['chl_ocx']
	chl_ocx = np.array(chl_ocx).flatten()
	reducedChlOcx = chl_ocx[inds]
	Kd_490 = dataset['Kd_490']
	Kd_490 = np.array(Kd_490).flatten()
	reducedKd490 = Kd_490[inds]
	poc = dataset['poc']
	poc = np.array(poc).flatten()
	reducedPoc = poc[inds]
	nflh = dataset['nflh']
	nflh = np.array(nflh).flatten()
	reducedNflh = nflh[inds]

	for j in range(len(orig_indices)):
		if(orig_indices[j][0] != -1 and orig_indices[j][1] != -1):
			depth = florida_z[orig_indices[j][0]][orig_indices[j][0]]
			if(~np.isnan(reducedAngstrom[j])):
				if('{}_n'.format(depth) not in angstrom_sums):
					angstrom_sums['{}_n'.format(depth)] = 1
					angstrom_sums['{}_x'.format(depth)] = reducedAngstrom[j]
					angstrom_sums['{}_x2'.format(depth)] = reducedAngstrom[j]*reducedAngstrom[j]
				else:
					angstrom_sums['{}_n'.format(depth)] += 1
					angstrom_sums['{}_x'.format(depth)] += reducedAngstrom[j]
					angstrom_sums['{}_x2'.format(depth)] += reducedAngstrom[j]*reducedAngstrom[j]
			if(~np.isnan(reducedChlorA[j])):
				if('{}_n'.format(depth) not in chlor_a_sums):
					chlor_a_sums['{}_n'.format(depth)] = 1
					chlor_a_sums['{}_x'.format(depth)] = reducedChlorA[j]
					chlor_a_sums['{}_x2'.format(depth)] = reducedChlorA[j]*reducedChlorA[j]
				else:
					chlor_a_sums['{}_n'.format(depth)] += 1
					chlor_a_sums['{}_x'.format(depth)] += reducedChlorA[j]
					chlor_a_sums['{}_x2'.format(depth)] += reducedChlorA[j]*reducedChlorA[j]
			if(~np.isnan(reducedChlOcx[j])):
				if('{}_n'.format(depth) not in chl_ocx_sums):
					chl_ocx_sums['{}_n'.format(depth)] = 1
					chl_ocx_sums['{}_x'.format(depth)] = reducedChlOcx[j]
					chl_ocx_sums['{}_x2'.format(depth)] = reducedChlOcx[j]*reducedChlOcx[j]
				else:
					chl_ocx_sums['{}_n'.format(depth)] += 1
					chl_ocx_sums['{}_x'.format(depth)] += reducedChlOcx[j]
					chl_ocx_sums['{}_x2'.format(depth)] += reducedChlOcx[j]*reducedChlOcx[j]
			if(~np.isnan(reducedKd490[j])):
				if('{}_n'.format(depth) not in Kd_490_sums):
					Kd_490_sums['{}_n'.format(depth)] = 1
					Kd_490_sums['{}_x'.format(depth)] = reducedKd490[j]
					Kd_490_sums['{}_x2'.format(depth)] = reducedKd490[j]*reducedKd490[j]
				else:
					Kd_490_sums['{}_n'.format(depth)] += 1
					Kd_490_sums['{}_x'.format(depth)] += reducedKd490[j]
					Kd_490_sums['{}_x2'.format(depth)] += reducedKd490[j]*reducedKd490[j]
			if(~np.isnan(reducedPoc[j])):
				if('{}_n'.format(depth) not in poc_sums):
					poc_sums['{}_n'.format(depth)] = 1
					poc_sums['{}_x'.format(depth)] = reducedPoc[j]
					poc_sums['{}_x2'.format(depth)] = reducedPoc[j]*reducedPoc[j]
				else:
					poc_sums['{}_n'.format(depth)] += 1
					poc_sums['{}_x'.format(depth)] += reducedPoc[j]
					poc_sums['{}_x2'.format(depth)] += reducedPoc[j]*reducedPoc[j]
			if(~np.isnan(reducedNflh[j])):
				if('{}_n'.format(depth) not in nflh_sums):
					nflh_sums['{}_n'.format(depth)] = 1
					nflh_sums['{}_x'.format(depth)] = reducedNflh[j]
					nflh_sums['{}_x2'.format(depth)] = reducedNflh[j]*reducedNflh[j]
				else:
					nflh_sums['{}_n'.format(depth)] += 1
					nflh_sums['{}_x'.format(depth)] += reducedNflh[j]
					nflh_sums['{}_x2'.format(depth)] += reducedNflh[j]*reducedNflh[j]

np.save('angstrom_sums.npy', angstrom_sums)
np.save('chlor_a_sums.npy', chlor_a_sums)
np.save('chl_ocx_sums.npy', chl_ocx_sums)
np.save('Kd_490_sums.npy', Kd_490_sums)
np.save('poc_sums.npy', poc_sums)
np.save('nflh_sums.npy', nflh_sums)