from os import listdir
from os.path import isfile, join
import datetime as dt
import numpy as np
from file_search import *
from utils import *

def hycom_dist(initialdate_time, initialdate_lat, initialdate_lon, targetdate_time, targetdate_lat, targetdate_lon):
	#Assume that red tide does not move with current perfectly
	movement_factor = 0.00001

	hycom_data = 'HYCOM_data'
	onlyfiles = [f for f in listdir(hycom_data) if isfile(join(hycom_data, f))]

	physicalDistance = np.zeros(len(initialdate_time))

	for i in range(len(initialdate_time)):
		total_time = (targetdate_time-initialdate_time[i]).total_seconds()
		found_file = file_search(initialdate_time[i], onlyfiles)

		water_u_file = hycom_data+'/'+found_file[0:19]+'water_u.npy'
		water_v_file = hycom_data+'/'+found_file[0:19]+'water_v.npy'

		water_u = np.load(water_u_file)
		water_v = np.load(water_v_file)
		water_u = water_u/1000
		water_v = water_v/1000

		lat_file = 'expt_50.1_lat.npy'
		lon_file = 'expt_50.1_lon.npy'

		lat = np.load(lat_file)
		lon = np.load(lon_file)
		lat = lat[150:]
		lon = lon[275:475]

		lat_idx = find_nearest(lat, initialdate_lat[i])
		lon_idx = find_nearest(lon, initialdate_lon[i])

		u_val = water_u[0,0,lat_idx,lon_idx]
		v_val = water_v[0,0,lat_idx,lon_idx]

		u_movement = u_val*total_time*movement_factor
		v_movement = v_val*total_time*movement_factor

		new_lat = initialdate_lat[i] + v_movement
		new_lon = initialdate_lon[i] + u_movement

		physicalDistance[i] = np.sqrt((new_lat-targetdate_lat)**2 + (new_lon-targetdate_lon)**2)

	return physicalDistance