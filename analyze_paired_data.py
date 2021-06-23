import pandas as pd
from scipy import stats
import numpy as np

paired_df = pd.read_pickle('paired_dataset.pkl')

columns_to_compare=['Sample Depth', 'Latitude', 'Longitude',\
	'aot_869', 'angstrom', 'Rrs_412', 'Rrs_443', 'Rrs_469', 'Rrs_488', 'Rrs_531', 'Rrs_547', 'Rrs_555', 'Rrs_645',\
	'Rrs_667', 'Rrs_678', 'chlor_a', 'chl_ocx', 'Kd_490', 'pic', 'poc', 'ipar', 'nflh', 'par']

red_tide = paired_df['Red Tide Concentration'].to_numpy().copy()

corrs = []

for i in range(len(columns_to_compare)):
	feature = paired_df[columns_to_compare[i]].to_numpy().copy()
	inds = np.argwhere(~np.isnan(feature))
	feature_clean = np.squeeze(feature[inds].copy())
	red_tide_clean = np.squeeze(red_tide[inds].copy())
	corr, _ = stats.pearsonr(feature_clean, red_tide_clean)
	corrs.append(corr)

corrs = np.array(corrs)
abscorrs = np.abs(corrs)
ordering = np.argsort(abscorrs)
ordering = np.flip(ordering)

for i in range(len(ordering)):
	print('{0:20}{1}'.format(columns_to_compare[ordering[i]], corrs[ordering[i]]))