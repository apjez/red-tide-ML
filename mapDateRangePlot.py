import numpy as np
import sys
import pandas as pd
import xarray as xr
import os
import torch
import netCDF4
import json
import matplotlib.pyplot as plt
import datetime as dt
import time
from configparser import ConfigParser
from convertFeaturesByDepth import *
from findMatrixCoordsBedrock import *
from model import *
from utils import *

folder_name = 'map_images_2017-2018'
startDate = pd.Timestamp(year=2017, month=6, day=1, hour=0)
endDate = pd.Timestamp(year=2018, month=6, day=1, hour=0)

dates = []

testDate = startDate
while(testDate < endDate):
	dates.append(testDate)
	testDate = testDate + pd.Timedelta(days=1)

file_path = 'PinellasMonroeCoKareniabrevis 2010-2020.06.12.xlsx'

df = pd.read_excel(file_path, engine='openpyxl')

step_size = 0.015
florida_x = np.arange(-92, -75, step_size)
florida_y = np.arange(20, 35, step_size)
florida_lats = np.tile(florida_y, len(florida_x))
florida_lons = np.repeat(florida_x, len(florida_y))
florida_lats = np.reshape(florida_lats, (florida_x.shape[0], florida_y.shape[0]), order='C')
florida_lons = np.reshape(florida_lons, (florida_x.shape[0], florida_y.shape[0]), order='C')

florida_lats = florida_lats[520:780, 300:580]
florida_lons = abs(florida_lons[520:780, 300:580])

florida_lats = florida_lats[0, :]
florida_lons = florida_lons[:, 0]

day_counter = 0
for day in dates:
	print('Processing days: {}/{}'.format(day_counter, len(dates)))

	red_tide_output = np.load('{}/red_tide_output{}.png.npy'.format(folder_name, day_counter))

	plt.figure(dpi=500)
	plt.imshow(red_tide_output.T)
	plt.clim(-1, 1)
	plt.colorbar()
	plt.title('Red Tide Prediction {}/{}/{}'.format(day.month, day.day, day.year))
	plt.gca().invert_yaxis()
	fig = plt.gcf()
	ax = fig.gca()

	for i in range(0, df.shape[0]):
		importdate = df.at[i, 'Sample Date']
		latitude = df.at[i, 'Latitude']
		longitude = abs(df.at[i, 'Longitude'])
		redtide_conc = df.at[i, 'Karenia brevis abundance (cells/L)']


		if((day - importdate).days >= 0 and (day - importdate).days <= 7):
			lat_ind = find_nearest(florida_lats, latitude)
			lon_ind = find_nearest(florida_lons, longitude)
			if(abs(florida_lats[lat_ind] - latitude)<2 and abs(florida_lons[lon_ind] - longitude)<2):
				daysback = (day - importdate).days
				opacity = ((7-daysback)+1)/8
				if(redtide_conc < 1000):
					color = (0.5, 0.5, 0.5, opacity)
				elif(redtide_conc > 1000 and redtide_conc < 10000):
					color = (1, 1, 1, opacity)
				elif(redtide_conc > 10000 and redtide_conc < 100000):
					color = (1, 1, 0, opacity)
				elif(redtide_conc > 100000 and redtide_conc < 1000000):
					color = (1, 0.65, 0, opacity)
				else:
					color = (1, 0, 0, opacity)

				radius = 1
				circle = plt.Circle((lon_ind, lat_ind), radius, color=color)
				ax.add_patch(circle)



	plt.savefig('{}/red_tide_combined{}.png'.format(folder_name, str(day_counter).zfill(5)), bbox_inches='tight')

	day_counter = day_counter + 1