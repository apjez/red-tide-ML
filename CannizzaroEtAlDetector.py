import numpy as np

# Implements red tide detection method from Cannizzaro et al
# Detection of Karenia brevis blooms on the west Florida shelf using in situ backscattering and fluorescence data
# features are in order ['chl_ocx', 'Rrs_443', 'Rrs_555']
def CannizzaroEtAlDetector(features):
	#QAA to derive b_bp from Lee et al. 2002
	rrs_443 = features[:, 1]/(0.52 + 1.7*features[:, 1])
	rrs_555 = features[:, 2]/(0.52 + 1.7*features[:, 2])

	g_0 = 0.0895
	g_1 = 0.1247
	#from Lin and Lee, 2018
	bbw_555 = 0.001
	u_555 = (-g_0 + np.sqrt(g_0**2 + 4*g_1*rrs_555))/(2*g_1)
	rho = np.log(rrs_443/rrs_555)
	a_440_i = np.exp(-2.0 - 1.4*rho + 0.2*(rho**2))
	a_555 = 0.0596 + 0.2*(a_440_i - 0.01)
	bb_555 = ((u_555*a_555)/(1-u_555))
	bbp_555_QAA = ((u_555*a_555)/(1-u_555)) - bbw_555

	red_tide = np.zeros_like(u_555)
	red_tide_inds = np.where((features[:, 0]>1.5) &  (bbp_555_QAA<0.0045))
	red_tide[red_tide_inds[0]] = 1

	return red_tide