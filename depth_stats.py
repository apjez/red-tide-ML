import netCDF4
import os
import pandas as pd
import xarray as xr
import time
from findMatrixCoordsBedrock import *

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
	latarr = np.expand_dims(latarr, axis=1)
	longarr = np.expand_dims(longarr, axis=1)

	fh = netCDF4.Dataset(file_path, mode='r')
	collectionDate = fh.time_coverage_start[0:10]
	collectionTimeStamp = pd.Timestamp(int(collectionDate[0:4]), int(collectionDate[5:7]), int(collectionDate[8:10]), 0)

	if(i % 100 == 0):
		print('Processing file #{}'.format(i))

	start = time.time()
	orig_indices = findMatrixCoordsBedrock(florida_y, florida_x, latarr, longarr)
	end = time.time()

	dataset = xr.open_dataset(file_path, 'geophysical_data')
	angstrom = dataset['angstrom']
	angstrom = np.array(angstrom).flatten()
	chlor_a = dataset['chlor_a']
	chlor_a = np.array(chlor_a).flatten()
	chl_ocx = dataset['chl_ocx']
	chl_ocx = np.array(chl_ocx).flatten()
	Kd_490 = dataset['Kd_490']
	Kd_490 = np.array(Kd_490).flatten()
	poc = dataset['poc']
	poc = np.array(poc).flatten()
	nflh = dataset['nflh']
	nflh = np.array(nflh).flatten()

	for j in range(len(orig_indices)):
		if(orig_indices[j][0] != -1 and orig_indices[j][1] != -1):
			depth = florida_z[orig_indices[j][0]][orig_indices[j][0]]
			if(~np.isnan(angstrom[j])):
				if('{}_n'.format(depth) not in angstrom_sums):
					angstrom_sums['{}_n'.format(depth)] = 1
					angstrom_sums['{}_x'.format(depth)] = angstrom[j]
					angstrom_sums['{}_x2'.format(depth)] = angstrom[j]*angstrom[j]
				else:
					angstrom_sums['{}_n'.format(depth)] += 1
					angstrom_sums['{}_x'.format(depth)] += angstrom[j]
					angstrom_sums['{}_x2'.format(depth)] += angstrom[j]*angstrom[j]
			if(~np.isnan(chlor_a[j])):
				if('{}_n'.format(depth) not in chlor_a_sums):
					chlor_a_sums['{}_n'.format(depth)] = 1
					chlor_a_sums['{}_x'.format(depth)] = chlor_a[j]
					chlor_a_sums['{}_x2'.format(depth)] = chlor_a[j]*chlor_a[j]
				else:
					chlor_a_sums['{}_n'.format(depth)] += 1
					chlor_a_sums['{}_x'.format(depth)] += chlor_a[j]
					chlor_a_sums['{}_x2'.format(depth)] += chlor_a[j]*chlor_a[j]
			if(~np.isnan(chl_ocx[j])):
				if('{}_n'.format(depth) not in chl_ocx_sums):
					chl_ocx_sums['{}_n'.format(depth)] = 1
					chl_ocx_sums['{}_x'.format(depth)] = chl_ocx[j]
					chl_ocx_sums['{}_x2'.format(depth)] = chl_ocx[j]*chl_ocx[j]
				else:
					chl_ocx_sums['{}_n'.format(depth)] += 1
					chl_ocx_sums['{}_x'.format(depth)] += chl_ocx[j]
					chl_ocx_sums['{}_x2'.format(depth)] += chl_ocx[j]*chl_ocx[j]
			if(~np.isnan(Kd_490[j])):
				if('{}_n'.format(depth) not in Kd_490_sums):
					Kd_490_sums['{}_n'.format(depth)] = 1
					Kd_490_sums['{}_x'.format(depth)] = Kd_490[j]
					Kd_490_sums['{}_x2'.format(depth)] = Kd_490[j]*Kd_490[j]
				else:
					Kd_490_sums['{}_n'.format(depth)] += 1
					Kd_490_sums['{}_x'.format(depth)] += Kd_490[j]
					Kd_490_sums['{}_x2'.format(depth)] += Kd_490[j]*Kd_490[j]
			if(~np.isnan(poc[j])):
				if('{}_n'.format(depth) not in poc_sums):
					poc_sums['{}_n'.format(depth)] = 1
					poc_sums['{}_x'.format(depth)] = poc[j]
					poc_sums['{}_x2'.format(depth)] = poc[j]*poc[j]
				else:
					poc_sums['{}_n'.format(depth)] += 1
					poc_sums['{}_x'.format(depth)] += poc[j]
					poc_sums['{}_x2'.format(depth)] += poc[j]*poc[j]
			if(~np.isnan(nflh[j])):
				if('{}_n'.format(depth) not in nflh_sums):
					nflh_sums['{}_n'.format(depth)] = 1
					nflh_sums['{}_x'.format(depth)] = nflh[j]
					nflh_sums['{}_x2'.format(depth)] = nflh[j]*nflh[j]
				else:
					nflh_sums['{}_n'.format(depth)] += 1
					nflh_sums['{}_x'.format(depth)] += nflh[j]
					nflh_sums['{}_x2'.format(depth)] += nflh[j]*nflh[j]

np.save('angstrom_sums.npy', angstrom_sums)
np.save('chlor_a_sums.npy', chlor_a_sums)
np.save('chl_ocx_sums.npy', chl_ocx_sums)
np.save('Kd_490_sums.npy', Kd_490_sums)
np.save('poc_sums.npy', poc_sums)
np.save('nflh_sums.npy', nflh_sums)