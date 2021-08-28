import numpy as np

# Implements red tide detection method from Soto et al
# https://ioccg.org/wp-content/uploads/2021/05/ioccg_report_20-habs-2021-web.pdf (Chapter 6)
# features are in order ['chl_ocx', 'nflh', 'Rrs_443', 'Rrs_555']
def SotoEtAlDetector(features):
	#QAA to derive b_bp from Lee et al. 2002
	rrs_443 = features[:, 2]/(0.52 + 1.7*features[:, 2])
	rrs_555 = features[:, 3]/(0.52 + 1.7*features[:, 3])

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
	bbp_555_MOREL = 0.3*(features[:, 0]**0.62)*(0.002 + 0.02*(0.5-0.25*np.log10(features[:, 0])))

	bbp_555_ratio = bbp_555_QAA/bbp_555_MOREL

	red_tide = np.zeros_like(u_555)
	red_tide_inds = np.where((features[:, 0]>1.5) & (features[:, 1]>0.1) & (bbp_555_ratio<1))
	red_tide[red_tide_inds[0]] = 1

	return red_tide