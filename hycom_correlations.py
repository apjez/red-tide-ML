import datetime as dt
import pandas as pd
import numpy as np
import xarray as xr
import os
import netCDF4
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode

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

pixel_lats = []
pixel_lons = []
pixel_chlor_a = []
pixel_times = []

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
		year = int(collectionDateTime[0:4])
		month = int(collectionDateTime[5:7])
		day = int(collectionDateTime[8:10])
		hour = int(collectionDateTime[11:13])
		minute = int(collectionDateTime[14:16])
		second = int(collectionDateTime[17:19])
		collectionDateTime = dt.datetime(year, month, day, hour, minute, second)

		dataset = xr.open_dataset(file_path_list_sorted[i], 'geophysical_data')
		chlor_a = np.array(dataset['chlor_a']).flatten()
		chlor_a = chlor_a[inds_in_area]
		valid_inds = np.where(np.isnan(chlor_a) == False)[0]
		chlor_a = chlor_a[valid_inds]

		latarr = latarr[inds_in_area]
		latarr = latarr[valid_inds]
		longarr = longarr[inds_in_area]
		longarr = longarr[valid_inds]

		for j in range(len(valid_inds)):
			pixel_lats.append(latarr[j])
			pixel_lons.append(longarr[j])
			pixel_chlor_a.append(chlor_a[j])
			pixel_times.append((collectionDateTime-startDate).total_seconds())
		
		num_valid_pixels = num_valid_pixels + valid_inds.shape[0]

# 1. Setting up the velocity fields in a FieldSet object
fname = '/run/media/rfick/UF10/HYCOM/expt_32.5_netCDF/*.nc'
filename_lon = '/run/media/rfick/UF10/HYCOM/expt_32.5_lon.npy'
filename_lat = '/run/media/rfick/UF10/HYCOM/expt_32.5_lat.npy'
filenames = {'U': fname, 'V': fname}
variables = {'U': 'eastward_eulerian_current_velocity', 'V': 'northward_eulerian_current_velocity'}
dimensions = {'U': {'lat': 'lat', 'lon': 'lon', 'time': 'time'},
              'V': {'lat': 'lat', 'lon': 'lon', 'time': 'time'}}
fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)

study_length = (dt.timedelta(days=29)).total_seconds()

# 2. Defining the particles type and initial conditions in a ParticleSet object
lon_array = np.load(filename_lon)[275:475]
lat_array = np.load(filename_lat)[150:346]
lon_array = lon_array[60:150]
lat_array = lat_array[50:135]

pset = ParticleSet(fieldset=fieldset,   # the fields on which the particles are advected
                   pclass=JITParticle,  # the type of particles (JITParticle or ScipyParticle)
                   lon=pixel_lons,      # release longitudes 
                   lat=pixel_lats,      # release latitudes
                   time=pixel_times)    # release times


# Delete particles that leave the playing area
def DeleteParticle(particle, fieldset, time):
	particle.delete()


# 3. Executing an advection kernel on the given fieldset
output_file = pset.ParticleFile(name="/run/media/rfick/UF10/HYCOM/hycom_correlations.nc", outputdt=3600) # the file name and the time step of the outputs
pset.execute(AdvectionRK4,                 # the kernel (which defines how particles move)
             runtime=study_length,         # the total length of the run
             dt=300,                       # the timestep of the kernel
             output_file=output_file,
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})


# 4. Exporting the simulation output to a netcdf file

output_file.export()
output_file.close()

np.save("/run/media/rfick/UF10/HYCOM/pixel_chlor_a.npy", pixel_chlor_a)