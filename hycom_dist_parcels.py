from os import listdir
from os.path import isfile, join
import datetime as dt
import numpy as np
import netCDF4
import xarray as xr
import math
import sys
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode
from file_search import *
from utils import *

# Initialdate objects are lists of equal size, one element per in situ measurement
def hycom_dist_parcels(simulation_file_list, initialdate_time, initialdate_lat, initialdate_lon, targetdate_time, targetdate_lat, targetdate_lon, filename_lon, filename_lat):
	# 1. Setting up the velocity fields in a FieldSet object
	fname = simulation_file_list
	filenames = {'U': fname, 'V': fname}
	variables = {'U': 'eastward_eulerian_current_velocity', 'V': 'northward_eulerian_current_velocity'}
	dimensions = {'U': {'lat': 'lat', 'lon': 'lon', 'time': 'time'},
	              'V': {'lat': 'lat', 'lon': 'lon', 'time': 'time'}}
	fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)

	date_differences = []
	for i in range(len(initialdate_time)):
		date_differences.append(targetdate_time - initialdate_time[i])

	longest_date_difference = max(date_differences)
	start_times = []
	for i in range(len(date_differences)):
		start_times.append((longest_date_difference - date_differences[i]).total_seconds())

	study_length = longest_date_difference.total_seconds()

	# 2. Defining the particles type and initial conditions in a ParticleSet object
	lon_array = np.load(filename_lon)[275:475]
	lat_array = np.load(filename_lat)[150:346]
	#lon_array = lon_array[60:150]
	#lat_array = lat_array[50:135]
	x = initialdate_lon
	y = initialdate_lat
	lon = x
	lat = y

	pset = ParticleSet(fieldset=fieldset,   # the fields on which the particles are advected
	                   pclass=JITParticle,  # the type of particles (JITParticle or ScipyParticle)
	                   lon=lon,             # release longitudes 
	                   lat=lat,             # release latitudes
	                   time=start_times)    # release times


	# Delete particles that leave the playing area
	def DeleteParticle(particle, fieldset, time):
		particle.delete()


	# 3. Executing an advection kernel on the given fieldset
	output_file = pset.ParticleFile(name="/run/media/rfick/UF10/HYCOM/GCParticles.nc", outputdt=3600) # the file name and the time step of the outputs
	pset.execute(AdvectionRK4,                 # the kernel (which defines how particles move)
	             runtime=study_length,         # the total length of the run
	             dt=300,                       # the timestep of the kernel
	             output_file=output_file,
	             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})


	# 4. Exporting the simulation output to a netcdf file

	output_file.export()
	output_file.close()

	data_netcdf4 = netCDF4.Dataset('/run/media/rfick/UF10/HYCOM/GCParticles.nc', mode='r')

	data_xarray = xr.open_dataset(xr.backends.NetCDF4DataStore(data_netcdf4))

	np.set_printoptions(threshold=sys.maxsize)

	last_valid_index = (~np.isnan(data_xarray['lat'].values) & (data_xarray['lat'].values < 100)).sum(axis=1) - 1

	final_lon = np.squeeze(data_xarray['lon'].values[np.arange(len(last_valid_index)), last_valid_index])
	final_lat = np.squeeze(data_xarray['lat'].values[np.arange(len(last_valid_index)), last_valid_index])

	dist = np.sqrt((final_lat - targetdate_lat)**2 + (final_lon - targetdate_lon)**2)

	return dist