import numpy as np

def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

def convertFeaturesByPosition(features, features_to_use):
	#Find position stats
	florida_x = np.load('florida_x.npy')
	florida_y = np.load('florida_y.npy')

	position_stats = np.zeros((len(features_to_use), 2, florida_y.shape[0], florida_x.shape[0]))
	for i in range(len(features_to_use)):
		position_stats[i, 0, :, :] = np.load('latlon_stats/'+features_to_use[i]+'_mean.npy')
		position_stats[i, 1, :, :] = np.load('latlon_stats/'+features_to_use[i]+'_std.npy')

	featuresPositionConverted = np.zeros((features.shape[0], features.shape[1]-2))

	for i in range(features.shape[0]):
		# Depth is last feature
		latitude = features[i,0]
		longitude = features[i,1]
		closest_latitude_idx = find_nearest(florida_y, latitude)
		closest_longitude_idx = find_nearest(florida_x, longitude)
		for j in range(featuresPositionConverted.shape[1]):
			mean = position_stats[j, 0, closest_latitude_idx, closest_longitude_idx]
			std = position_stats[j, 1, closest_latitude_idx, closest_longitude_idx]
			if(std > 0):
				featuresPositionConverted[i, j] = (features[i, j+2]-mean)/std
			else:
				featuresPositionConverted[i, j] = 0

	return featuresPositionConverted