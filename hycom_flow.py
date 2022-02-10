from pydap.client import open_url
import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

save_folder = 'expt_31.0_data/'

while True:
	try:
		dataset = open_url("https://tds.hycom.org/thredds/dodsC/GOMl0.04/expt_31.0")
		break
	except Exception as e:
		print(e)
		print('trying again...')

#time is listed in hours since 2000-01-01 00:00:00
basedate = dt.datetime(2000, 1, 1)
while True:
	try:
		timearray = dataset.time[:].data
		np.save('expt_31.0_timearray.npy', timearray)
		break
	except Exception as e:
		print(e)
		print('trying again...')

while True:
	try:
		latitude = dataset.lat[:].data
		np.save('expt_31.0_lat.npy', latitude)
		break
	except Exception as e:
		print(e)
		print('trying again...')

while True:
	try:
		longitude = dataset.lon[:].data
		np.save('expt_31.0_lon.npy', longitude)
		break
	except Exception as e:
		print(e)
		print('trying again...')

dates = []
date_number = 0

for i in range(0, timearray.shape[0]):
	datechange = dt.timedelta(hours=timearray[i])
	newdate = basedate+datechange

	dates.append(newdate.isoformat())

	if(i%100 == 0):
		print(newdate.isoformat())

	#Ignore data before 2000 as MODIS-Aqua starts in 2000
	if(newdate > basedate):
		while True:
			try:
				water_u=dataset.water_u.array[i,0,150:346,275:475].data
				np.save(save_folder+str(date_number)+'water_u.npy', water_u)
				break
			except Exception as e:
				print(e)
				print('trying again...')
		while True:
			try:
				water_v=dataset.water_v.array[i,0,150:346,275:475].data
				np.save(save_folder+str(date_number)+'water_v.npy', water_v)
				break
			except Exception as e:
				print(e)
				print('trying again...')
		while True:
			try:
				water_temp=dataset.water_temp.array[i,0,150:346,275:475].data
				np.save(save_folder+str(date_number)+'water_temp.npy', water_temp)
				break
			except Exception as e:
				print(e)
				print('trying again...')
		while True:
			try:
				salinity=dataset.salinity.array[i,0,150:346,275:475].data
				np.save(save_folder+str(date_number)+'salinity.npy', salinity)
				break
			except Exception as e:
				print(e)
				print('trying again...')

		date_number = date_number + 1

np.save('expt_31.0_dates.npy', dates)