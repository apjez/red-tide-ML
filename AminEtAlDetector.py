import numpy as np

# Amin, Ruhul, et al. "Noval optical techniques for detecting and classifying toxic dinoflagellate Karenia brevis blooms using satellite imagery."
# Optics express 17.11 (2009): 9126-9144.
# This method detects red tide by the following thresholds:
# RBD > 0.15 W/(m^2)/microm/sr
# KBBI > 0.3*RBD
# features in order are ['Rrs_667', 'Rrs_678']
def AminEtAlDetector(features):
	# Constant from Thuillier, G., Herse, M., Labs, D., et al. (2003). The solar spectral irradiance from 200 to 2400 nm as measured by the
	# SOLSPEC spectrometer from the Atlas and Eureca missions. Solar Physics 214: 1. doi:10.1023/A:1024048429145
	# In units of mW/(cm^2)
	extraterrestrial_solar_irradiance = 1367.7

	features = extraterrestrial_solar_irradiance*features

	RBD = features[:, 1] - features[:, 0]
	KBBI = (features[:, 1] - features[:, 0])/(features[:, 1] + features[:, 0])

	red_tide = np.zeros_like(RBD)

	for i in range(len(red_tide)):
		if(RBD[i]>0.15 and KBBI[i]>0.3*RBD[i]):
			red_tide[i] = 1

	return red_tide