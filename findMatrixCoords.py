import xarray as xr
import netCDF4
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial

# Finds the nearest matrix coordinates corresponding to the
# Lats and lons of those provided

def findMatrixCoords(file_path, sample_lats, sample_lons):
	nav_dataset = xr.open_dataset(file_path, 'navigation_data')

	latitude = nav_dataset['latitude']
	longitude = nav_dataset['longitude']

	# Exclude point if the nearest lat+lon is not closer than eta
	eta = 0.01

	latarr = np.array(latitude).flatten()
	longarr = np.array(longitude).flatten()
	latarr = np.expand_dims(latarr, axis=1)
	longarr = np.expand_dims(longarr, axis=1)

	points = np.concatenate([latarr, longarr], axis=1)
	latlongKDTree = spatial.KDTree(points)
	orig_indices = []
	for i in range(len(sample_lats)):
		pointNW = [sample_lats[i], sample_lons[i]]
		distance,index = latlongKDTree.query(pointNW)
		orig_index = np.unravel_index(index, latitude.shape)
		if(distance < eta):
			orig_indices.append(orig_index)
		else:
			orig_indices.append((-1, -1))

	return orig_indices