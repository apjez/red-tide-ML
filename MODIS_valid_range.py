# Check if a value is within valid range for that feature according to https://oceancolor.gsfc.nasa.gov/docs/format/l2oc_modis/
# value - scalar
# feature - string of feature name
def MODIS_valid_range(value, feature):
	returnVal = False
	if(feature == 'angstrom'):
		# Valid range: (-0.500, 3.000)
		if(value > -0.500 and value < 3.000):
			returnVal = True
	elif(feature == 'chlor_a'):
		# Valid range: (0.001, 100)
		if(value > 0.001 and value < 100):
			returnVal = True
	elif(feature == 'chl_ocx'):
		# Valid range: (0.001, 100)
		if(value > 0.001 and value < 100):
			returnVal = True
	elif(feature == 'Kd_490'):
		# Valid range: (0.010, 6.000)
		if(value > 0.010 and value < 6.000):
			returnVal = True
	elif(feature == 'poc'):
		# Valid range: (0.000, 1000.000)
		if(value > 0.000 and value < 1000.000):
			returnVal = True
	elif(feature == 'nflh'):
		# Valid range: (-0.500, 5.000)
		if(value > -0.500 and value < 5.000):
			returnVal = True
	elif(feature == 'par'):
		# Valid range: (0.000, 18.928)
		if(value > 0.000 and value < 18.928):
			returnVal = True
	elif(feature == 'Rrs_443'):
		# Valid range: (-0.010, 0.100)
		if(value > -0.010 and value < 0.100):
			returnVal = True
	elif(feature == 'Rrs_469'):
		# Valid range: (-0.010, 0.100)
		if(value > -0.010 and value < 0.100):
			returnVal = True
	elif(feature == 'Rrs_488'):
		# Valid range: (-0.010, 0.100)
		if(value > -0.010 and value < 0.100):
			returnVal = True
	else:
		print('Feature name invalid!')

	return returnVal