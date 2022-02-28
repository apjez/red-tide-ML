import netCDF4
import os
import pandas as pd
import xarray as xr
import time
import matplotlib.pyplot as plt
from findMatrixCoordsBedrock import *
from MODIS_valid_range import *

def maskPoints(A_x, A_y, B_x, B_y, X, Y):
	return (B_x - A_x)*(Y - A_y) - (B_y - A_y)*(X - A_x)

florida_x = np.load('florida_x.npy')
florida_y = np.load('florida_y.npy')
florida_z = np.load('florida_z.npy')

data_folder = '/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/requested_files'
data_list = os.listdir(data_folder)
save_folder = 'depth_stats'

angstrom_sums = {}
chlor_a_sums = {}
chl_ocx_sums = {}
Kd_490_sums = {}
poc_sums = {}
nflh_sums = {}
par_sums = {}
Rrs_443_sums = {}
Rrs_469_sums = {}
Rrs_488_sums = {}

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
		print('Processing file #{}/{}'.format(i, len(data_list)))

	orig_indices = findMatrixCoordsBedrock(florida_y, florida_x, reducedLatArr, reducedLongArr)

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

	par = dataset['par']
	par = np.array(par).flatten()
	reducedPar = par[inds]
	Rrs_443 = dataset['Rrs_443']
	Rrs_443 = np.array(Rrs_443).flatten()
	reducedRrs_443 = Rrs_443[inds]
	Rrs_469 = dataset['Rrs_469']
	Rrs_469 = np.array(Rrs_469).flatten()
	reducedRrs_469 = Rrs_469[inds]
	Rrs_488 = dataset['Rrs_488']
	Rrs_488 = np.array(Rrs_488).flatten()
	reducedRrs_488 = Rrs_488[inds]

	for j in range(len(orig_indices)):
		if(orig_indices[j][0] != -1 and orig_indices[j][1] != -1):
			depth = florida_z[orig_indices[j][0]][orig_indices[j][0]]
			if(~np.isnan(reducedAngstrom[j]) and MODIS_valid_range(reducedAngstrom[j], 'angstrom')):
				if('{}_n'.format(depth) not in angstrom_sums):
					angstrom_sums['{}_n'.format(depth)] = 1
					angstrom_sums['{}_x'.format(depth)] = reducedAngstrom[j]
					angstrom_sums['{}_x2'.format(depth)] = reducedAngstrom[j]*reducedAngstrom[j]
				else:
					angstrom_sums['{}_n'.format(depth)] += 1
					angstrom_sums['{}_x'.format(depth)] += reducedAngstrom[j]
					angstrom_sums['{}_x2'.format(depth)] += reducedAngstrom[j]*reducedAngstrom[j]
			if(~np.isnan(reducedChlorA[j]) and MODIS_valid_range(reducedChlorA[j], 'chlor_a')):
				if('{}_n'.format(depth) not in chlor_a_sums):
					chlor_a_sums['{}_n'.format(depth)] = 1
					chlor_a_sums['{}_x'.format(depth)] = reducedChlorA[j]
					chlor_a_sums['{}_x2'.format(depth)] = reducedChlorA[j]*reducedChlorA[j]
				else:
					chlor_a_sums['{}_n'.format(depth)] += 1
					chlor_a_sums['{}_x'.format(depth)] += reducedChlorA[j]
					chlor_a_sums['{}_x2'.format(depth)] += reducedChlorA[j]*reducedChlorA[j]
			if(~np.isnan(reducedChlOcx[j]) and MODIS_valid_range(reducedChlOcx[j], 'chl_ocx')):
				if('{}_n'.format(depth) not in chl_ocx_sums):
					chl_ocx_sums['{}_n'.format(depth)] = 1
					chl_ocx_sums['{}_x'.format(depth)] = reducedChlOcx[j]
					chl_ocx_sums['{}_x2'.format(depth)] = reducedChlOcx[j]*reducedChlOcx[j]
				else:
					chl_ocx_sums['{}_n'.format(depth)] += 1
					chl_ocx_sums['{}_x'.format(depth)] += reducedChlOcx[j]
					chl_ocx_sums['{}_x2'.format(depth)] += reducedChlOcx[j]*reducedChlOcx[j]
			if(~np.isnan(reducedKd490[j]) and MODIS_valid_range(reducedKd490[j], 'Kd_490')):
				if('{}_n'.format(depth) not in Kd_490_sums):
					Kd_490_sums['{}_n'.format(depth)] = 1
					Kd_490_sums['{}_x'.format(depth)] = reducedKd490[j]
					Kd_490_sums['{}_x2'.format(depth)] = reducedKd490[j]*reducedKd490[j]
				else:
					Kd_490_sums['{}_n'.format(depth)] += 1
					Kd_490_sums['{}_x'.format(depth)] += reducedKd490[j]
					Kd_490_sums['{}_x2'.format(depth)] += reducedKd490[j]*reducedKd490[j]
			if(~np.isnan(reducedPoc[j]) and MODIS_valid_range(reducedPoc[j], 'poc')):
				if('{}_n'.format(depth) not in poc_sums):
					poc_sums['{}_n'.format(depth)] = 1
					poc_sums['{}_x'.format(depth)] = reducedPoc[j]
					poc_sums['{}_x2'.format(depth)] = reducedPoc[j]*reducedPoc[j]
				else:
					poc_sums['{}_n'.format(depth)] += 1
					poc_sums['{}_x'.format(depth)] += reducedPoc[j]
					poc_sums['{}_x2'.format(depth)] += reducedPoc[j]*reducedPoc[j]
			if(~np.isnan(reducedNflh[j]) and MODIS_valid_range(reducedNflh[j], 'nflh')):
				if('{}_n'.format(depth) not in nflh_sums):
					nflh_sums['{}_n'.format(depth)] = 1
					nflh_sums['{}_x'.format(depth)] = reducedNflh[j]
					nflh_sums['{}_x2'.format(depth)] = reducedNflh[j]*reducedNflh[j]
				else:
					nflh_sums['{}_n'.format(depth)] += 1
					nflh_sums['{}_x'.format(depth)] += reducedNflh[j]
					nflh_sums['{}_x2'.format(depth)] += reducedNflh[j]*reducedNflh[j]

			if(~np.isnan(reducedPar[j]) and MODIS_valid_range(reducedPar[j], 'par')):
				if('{}_n'.format(depth) not in par_sums):
					par_sums['{}_n'.format(depth)] = 1
					par_sums['{}_x'.format(depth)] = reducedPar[j]
					par_sums['{}_x2'.format(depth)] = reducedPar[j]*reducedPar[j]
				else:
					par_sums['{}_n'.format(depth)] += 1
					par_sums['{}_x'.format(depth)] += reducedPar[j]
					par_sums['{}_x2'.format(depth)] += reducedPar[j]*reducedPar[j]
			if(~np.isnan(reducedRrs_443[j]) and MODIS_valid_range(reducedRrs_443[j], 'Rrs_443')):
				if('{}_n'.format(depth) not in Rrs_443_sums):
					Rrs_443_sums['{}_n'.format(depth)] = 1
					Rrs_443_sums['{}_x'.format(depth)] = reducedRrs_443[j]
					Rrs_443_sums['{}_x2'.format(depth)] = reducedRrs_443[j]*reducedRrs_443[j]
				else:
					Rrs_443_sums['{}_n'.format(depth)] += 1
					Rrs_443_sums['{}_x'.format(depth)] += reducedRrs_443[j]
					Rrs_443_sums['{}_x2'.format(depth)] += reducedRrs_443[j]*reducedRrs_443[j]
			if(~np.isnan(reducedRrs_469[j]) and MODIS_valid_range(reducedRrs_469[j], 'Rrs_469')):
				if('{}_n'.format(depth) not in Rrs_469_sums):
					Rrs_469_sums['{}_n'.format(depth)] = 1
					Rrs_469_sums['{}_x'.format(depth)] = reducedRrs_469[j]
					Rrs_469_sums['{}_x2'.format(depth)] = reducedRrs_469[j]*reducedRrs_469[j]
				else:
					Rrs_469_sums['{}_n'.format(depth)] += 1
					Rrs_469_sums['{}_x'.format(depth)] += reducedRrs_469[j]
					Rrs_469_sums['{}_x2'.format(depth)] += reducedRrs_469[j]*reducedRrs_469[j]
			if(~np.isnan(reducedRrs_488[j]) and MODIS_valid_range(reducedRrs_488[j], 'Rrs_488')):
				if('{}_n'.format(depth) not in Rrs_488_sums):
					Rrs_488_sums['{}_n'.format(depth)] = 1
					Rrs_488_sums['{}_x'.format(depth)] = reducedRrs_488[j]
					Rrs_488_sums['{}_x2'.format(depth)] = reducedRrs_488[j]*reducedRrs_488[j]
				else:
					Rrs_488_sums['{}_n'.format(depth)] += 1
					Rrs_488_sums['{}_x'.format(depth)] += reducedRrs_488[j]
					Rrs_488_sums['{}_x2'.format(depth)] += reducedRrs_488[j]*reducedRrs_488[j]

np.save(save_folder+'/angstrom_sums.npy', angstrom_sums)
np.save(save_folder+'/chlor_a_sums.npy', chlor_a_sums)
np.save(save_folder+'/chl_ocx_sums.npy', chl_ocx_sums)
np.save(save_folder+'/Kd_490_sums.npy', Kd_490_sums)
np.save(save_folder+'/poc_sums.npy', poc_sums)
np.save(save_folder+'/nflh_sums.npy', nflh_sums)

np.save(save_folder+'/par_sums.npy', par_sums)
np.save(save_folder+'/Rrs_443_sums.npy', Rrs_443_sums)
np.save(save_folder+'/Rrs_469_sums.npy', Rrs_469_sums)
np.save(save_folder+'/Rrs_488_sums.npy', Rrs_488_sums)