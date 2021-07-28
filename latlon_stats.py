import netCDF4
import os
import pandas as pd
import xarray as xr
import time
from findMatrixCoordsBedrock import *

def ensure_dir(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)

florida_x = np.load('florida_x.npy')
florida_y = np.load('florida_y.npy')
florida_z = np.load('florida_z.npy')

data_folder = '/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/requested_files'
data_list = os.listdir(data_folder)

angstrom_n = np.zeros((florida_y.shape[0], florida_x.shape[0]))
angstrom_x = np.zeros((florida_y.shape[0], florida_x.shape[0]))
angstrom_x2 = np.zeros((florida_y.shape[0], florida_x.shape[0]))
chlor_a_n = np.zeros((florida_y.shape[0], florida_x.shape[0]))
chlor_a_x = np.zeros((florida_y.shape[0], florida_x.shape[0]))
chlor_a_x2 = np.zeros((florida_y.shape[0], florida_x.shape[0]))
chl_ocx_n = np.zeros((florida_y.shape[0], florida_x.shape[0]))
chl_ocx_x = np.zeros((florida_y.shape[0], florida_x.shape[0]))
chl_ocx_x2 = np.zeros((florida_y.shape[0], florida_x.shape[0]))
Kd_490_n = np.zeros((florida_y.shape[0], florida_x.shape[0]))
Kd_490_x = np.zeros((florida_y.shape[0], florida_x.shape[0]))
Kd_490_x2 = np.zeros((florida_y.shape[0], florida_x.shape[0]))
poc_n = np.zeros((florida_y.shape[0], florida_x.shape[0]))
poc_x = np.zeros((florida_y.shape[0], florida_x.shape[0]))
poc_x2 = np.zeros((florida_y.shape[0], florida_x.shape[0]))
nflh_n = np.zeros((florida_y.shape[0], florida_x.shape[0]))
nflh_x = np.zeros((florida_y.shape[0], florida_x.shape[0]))
nflh_x2 = np.zeros((florida_y.shape[0], florida_x.shape[0]))

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
				angstrom_n[orig_indices[j][0], orig_indices[j][1]] += 1
				angstrom_x[orig_indices[j][0], orig_indices[j][1]] += angstrom[j]
				angstrom_x2[orig_indices[j][0], orig_indices[j][1]] += angstrom[j]*angstrom[j]
			if(~np.isnan(chlor_a[j])):
				chlor_a_n[orig_indices[j][0], orig_indices[j][1]] += 1
				chlor_a_x[orig_indices[j][0], orig_indices[j][1]] += chlor_a[j]
				chlor_a_x2[orig_indices[j][0], orig_indices[j][1]] += chlor_a[j]*chlor_a[j]
			if(~np.isnan(chl_ocx[j])):
				chl_ocx_n[orig_indices[j][0], orig_indices[j][1]] += 1
				chl_ocx_x[orig_indices[j][0], orig_indices[j][1]] += chl_ocx[j]
				chl_ocx_x2[orig_indices[j][0], orig_indices[j][1]] += chl_ocx[j]*chl_ocx[j]
			if(~np.isnan(Kd_490[j])):
				Kd_490_n[orig_indices[j][0], orig_indices[j][1]] += 1
				Kd_490_x[orig_indices[j][0], orig_indices[j][1]] += Kd_490[j]
				Kd_490_x2[orig_indices[j][0], orig_indices[j][1]] += Kd_490[j]*Kd_490[j]
			if(~np.isnan(poc[j])):
				poc_n[orig_indices[j][0], orig_indices[j][1]] += 1
				poc_x[orig_indices[j][0], orig_indices[j][1]] += poc[j]
				poc_x2[orig_indices[j][0], orig_indices[j][1]] += poc[j]*poc[j]
			if(~np.isnan(nflh[j])):
				nflh_n[orig_indices[j][0], orig_indices[j][1]] += 1
				nflh_x[orig_indices[j][0], orig_indices[j][1]] += nflh[j]
				nflh_x2[orig_indices[j][0], orig_indices[j][1]] += nflh[j]*nflh[j]

save_dir = 'latlon_stats/'
ensure_dir(save_dir)

np.save(save_dir+'angstrom_n.npy', angstrom_n)
np.save(save_dir+'angstrom_x.npy', angstrom_x)
np.save(save_dir+'angstrom_x2.npy', angstrom_x2)
np.save(save_dir+'chlor_a_n.npy', chlor_a_n)
np.save(save_dir+'chlor_a_x.npy', chlor_a_x)
np.save(save_dir+'chlor_a_x2.npy', chlor_a_x2)
np.save(save_dir+'chl_ocx_n.npy', chl_ocx_n)
np.save(save_dir+'chl_ocx_x.npy', chl_ocx_x)
np.save(save_dir+'chl_ocx_x2.npy', chl_ocx_x2)
np.save(save_dir+'Kd_490_n.npy', Kd_490_n)
np.save(save_dir+'Kd_490_x.npy', Kd_490_x)
np.save(save_dir+'Kd_490_x2.npy', Kd_490_x2)
np.save(save_dir+'poc_n.npy', poc_n)
np.save(save_dir+'poc_x.npy', poc_x)
np.save(save_dir+'poc_x2.npy', poc_x2)
np.save(save_dir+'nflh_n.npy', nflh_n)
np.save(save_dir+'nflh_x.npy', nflh_x)
np.save(save_dir+'nflh_x2.npy', nflh_x2)