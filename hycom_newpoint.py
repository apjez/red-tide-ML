from os import listdir
from os.path import isfile, join
import datetime as dt
import numpy as np
import netCDF4
from file_search import *
from file_search_netCDF import *
from hycom_dist import *
from hycom_dist_parcels import *
from utils import *

hycom_data = '/run/media/rfick/UF10/HYCOM/expt_50.1_netCDF/'

filename_lon = '/run/media/rfick/UF10/HYCOM/expt_50.1_lon.npy'
filename_lat = '/run/media/rfick/UF10/HYCOM/expt_50.1_lat.npy'

dates = np.load('/run/media/rfick/UF10/HYCOM/expt_50.1_dates.npy')

#print(dates)
#print(len(dates))

#onlyfiles = [f for f in listdir(hycom_data) if isfile(join(hycom_data, f))]

#onlyfiles.sort()

searchdatetime = [dt.datetime(2007, 9, 1, 0, 0, 0), dt.datetime(2007, 9, 10, 0, 0, 0), dt.datetime(2007, 9, 1, 0, 0, 0)]

earlieststartdatetime = searchdatetime[0]
for i in range(len(searchdatetime)):
	if(searchdatetime[i] < earlieststartdatetime):
		earlieststartdatetime = searchdatetime[i]

targetdatetime = dt.datetime(2007, 9, 30, 0, 0, 0)

#total_time = (targetdatetime-searchdatetime).total_seconds()

found_file_search_ind = file_search_netCDF(earlieststartdatetime, dates)
found_file_target_ind = file_search_netCDF(targetdatetime, dates)

#print(found_file_search_ind)
#print(found_file_target_ind)
#print(dates[found_file_search_ind])
#print(dates[found_file_target_ind])

simulation_file_list = []
for i in range(found_file_search_ind, found_file_target_ind+1):
	simulation_file_list.append(hycom_data+'netCDF4_file'+str(i)+'.nc')



	#data_netcdf4 = netCDF4.Dataset(hycom_data+onlyfiles[i], mode='r')
	#Dates are saved as days since 1900-12-31
	#basedate = dt.datetime(year=1900, month=12, day=31)
	#datechange = dt.timedelta(days=float(data_netcdf4['time'][0]))
	#print(basedate+datechange)

#lat_file = 'expt_50.1_lat.npy'
#lon_file = 'expt_50.1_lon.npy'

#lat = np.load(lat_file)
#lon = np.load(lon_file)
#lat = lat[150:]
#lon = lon[275:475]

#water_u_file = hycom_data+'/'+found_file[0:19]+'water_u.npy'
#water_v_file = hycom_data+'/'+found_file[0:19]+'water_v.npy'

#water_u = np.load(water_u_file)
#water_v = np.load(water_v_file)
#water_u = water_u/1000
#water_v = water_v/1000

test_lat = [25.5, 27, 27]
test_lon = [-84.5, -83, -83]

target_lat = 26
target_lon = -87

#3.83 km
#pixel_dist = 3.83

#Assume that red tide does not move with current perfectly
#movement_factor = 0.0001

#lat_idx = find_nearest(lat, test_lat)
#lon_idx = find_nearest(lon, test_lon)

#u_val = water_u[0,0,lat_idx,lon_idx]
#v_val = water_v[0,0,lat_idx,lon_idx]

#print(u_val)
#print(v_val)

#print(total_time)

#u_movement = u_val*total_time*movement_factor
#v_movement = v_val*total_time*movement_factor

#print(u_movement)
#print(v_movement)

#new_lat = test_lat + v_movement
#new_lon = test_lon + u_movement

#print(new_lat)
#print(new_lon)

#u_pixel_movement = u_movement/pixel_dist
#v_pixel_movement = v_movement/pixel_dist

#print(u_pixel_movement)
#print(v_pixel_movement)

estimated_dist = hycom_dist_parcels(simulation_file_list, searchdatetime, test_lat, test_lon, targetdatetime, target_lat, target_lon, filename_lon, filename_lat)

print(estimated_dist)