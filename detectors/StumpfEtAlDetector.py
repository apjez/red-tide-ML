import numpy as np

# Stumpf, R. P., Culver, M. E., Tester, P. A., Tomlinson, M., Kirkpatrick, G. J., Pederson, B. A., ... & Soracco, M. (2003).
# Monitoring Karenia brevis blooms in the Gulf of Mexico using satellite ocean color imagery and other data. Harmful Algae, 2(2), 147-160.
# This method detects red tide by the following thresholds:
# chlorophyll-a > 1 ug/L
# features in order are ['chlor_a']
# MODIS chlorophyll-a is provided in units of mg/(m^3) which is equivalent to ug/L
def StumpfEtAlDetector(features):
	
	#red_tide = np.zeros_like(features)

	#for i in range(len(red_tide)):
	#	if(features[i]>1):
	#		red_tide[i] = 1

	return features