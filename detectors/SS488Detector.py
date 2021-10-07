import numpy as np

# Tomlinson, M. C., T. T. Wynne, and R. P. Stumpf. "An evaluation of remote sensing techniques for enhanced detection of the toxic dinoflagellate, Karenia brevis." Remote Sensing of Environment 113.3 (2009): 598-609.
# Hill, Paul R., et al. "HABNet: machine learning, remote sensing-based detection of harmful algal blooms." IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 13 (2020): 3229-3239.
# This method detects red tide by using the spectral shape at 490 nm
# features in order are ['Rrs_469', 'Rrs_488', 'Rrs_531']
def SS488Detector(features):
	# Constant from Thuillier, G., Herse, M., Labs, D., et al. (2003). The solar spectral irradiance from 200 to 2400 nm as measured by the
	# SOLSPEC spectrometer from the Atlas and Eureca missions. Solar Physics 214: 1. doi:10.1023/A:1024048429145
	# In units of mW/(cm^2)
	extraterrestrial_solar_irradiance = 1367.7

	features = extraterrestrial_solar_irradiance*features

	spectral_shape = features[:, 1] - features[:, 0] - (features[:, 2] - features[:, 0])*((448-469)/(531-469))

	return spectral_shape