import numpy as np
import os

def ensure_folder(folder):
	if not os.path.isdir(folder):
		os.mkdir(folder)

def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

def find_nearest_latlon(lats, lons, lat_value, lon_value):
	lats = np.asarray(lats)
	lons = np.asarray(lons)
	idx = (np.abs(lats - lat_value)**2 + np.abs(lons - lon_value)**2).argmin()
	return idx

def find_nearest_batch(array, values):
	array = np.asarray(array)
	idx = (np.abs(array[:, None] - values)).argmin(axis=0)
	return idx