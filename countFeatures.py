import netCDF4
import os
import pandas as pd
import xarray as xr
import time
import matplotlib.pyplot as plt
from findMatrixCoordsBedrock import *

florida_x = np.load('florida_x.npy')
florida_y = np.load('florida_y.npy')
florida_z = np.load('florida_z.npy')

data_folder = '/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/requested_files'
data_list = os.listdir(data_folder)
save_folder = 'depth_stats'

angstrom_count = 0
aot_869_count = 0
chlor_a_count = 0
chl_ocx_count = 0
Kd_490_count = 0
pic_count = 0
poc_count = 0
nflh_count = 0
par_count = 0
ipar_count = 0
total_count = 0

for i in range(len(data_list)):
	file_id = data_list[i]
	file_path = data_folder + '/' + file_id

	fh = netCDF4.Dataset(file_path, mode='r')
	collectionDate = fh.time_coverage_start[0:10]
	collectionTimeStamp = pd.Timestamp(int(collectionDate[0:4]), int(collectionDate[5:7]), int(collectionDate[8:10]), 0)

	if(i % 100 == 0):
		print('Processing file #{}'.format(i))

	dataset = xr.open_dataset(file_path, 'geophysical_data')
	angstrom = dataset['angstrom']
	angstrom = np.array(angstrom).flatten()
	aot_869 = dataset['aot_869']
	aot_869 = np.array(aot_869).flatten()
	chlor_a = dataset['chlor_a']
	chlor_a = np.array(chlor_a).flatten()
	chl_ocx = dataset['chl_ocx']
	chl_ocx = np.array(chl_ocx).flatten()
	Kd_490 = dataset['Kd_490']
	Kd_490 = np.array(Kd_490).flatten()
	pic = dataset['pic']
	pic = np.array(pic).flatten()
	poc = dataset['poc']
	poc = np.array(poc).flatten()
	nflh = dataset['nflh']
	nflh = np.array(nflh).flatten()
	par = dataset['par']
	par = np.array(par).flatten()
	ipar = dataset['ipar']
	ipar = np.array(ipar).flatten()

	angstrom_count += np.sum(~np.isnan(angstrom))
	aot_869_count += np.sum(~np.isnan(aot_869))
	chlor_a_count += np.sum(~np.isnan(chlor_a))
	chl_ocx_count += np.sum(~np.isnan(chl_ocx))
	Kd_490_count += np.sum(~np.isnan(Kd_490))
	pic_count += np.sum(~np.isnan(pic))
	poc_count += np.sum(~np.isnan(poc))
	nflh_count += np.sum(~np.isnan(nflh))
	par_count += np.sum(~np.isnan(par))
	ipar_count += np.sum(~np.isnan(ipar))
	total_count += len(angstrom)

print('angstrom: {}'.format(angstrom_count))
print('aot_869: {}'.format(aot_869_count))
print('chlor_a: {}'.format(chlor_a_count))
print('chl_ocx: {}'.format(chl_ocx_count))
print('Kd_490: {}'.format(Kd_490_count))
print('pic: {}'.format(pic_count))
print('poc: {}'.format(pic_count))
print('nflh: {}'.format(nflh_count))
print('par: {}'.format(par_count))
print('ipar: {}'.format(ipar_count))
print('total: {}'.format(total_count))