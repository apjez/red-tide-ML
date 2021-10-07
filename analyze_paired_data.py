import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

paired_df = pd.read_pickle('paired_dataset.pkl')

columns_to_compare=['Sample Depth', 'Latitude', 'Longitude',\
	'aot_869', 'angstrom', 'Rrs_412', 'Rrs_443', 'Rrs_469', 'Rrs_488', 'Rrs_531', 'Rrs_547', 'Rrs_555', 'Rrs_645',\
	'Rrs_667', 'Rrs_678', 'chlor_a', 'chl_ocx', 'Kd_490', 'pic', 'poc', 'ipar', 'nflh', 'par']

red_tide = paired_df['Red Tide Concentration'].to_numpy().copy()

corrs = []

for i in range(len(columns_to_compare)):
	feature = paired_df[columns_to_compare[i]].to_numpy().copy()
	inds = np.argwhere(~np.isnan(feature))
	print('{} {}'.format(columns_to_compare[i], len(inds)))

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

ordered_columns = []
for i in range(len(columns_to_compare)):
	ordered_columns.append(columns_to_compare[ordering[i]])

for i in range(len(columns_to_compare)):
	if(i==0):
		table_data = [[columns_to_compare[ordering[i]], np.around(corrs[ordering[i]], decimals=3)]]
	else:
		table_data = np.append(table_data, [[columns_to_compare[ordering[i]], np.around(corrs[ordering[i]], decimals=3)]], axis=0)

fig, ax = plt.subplots(dpi=500)
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
cell_contents = pd.DataFrame(table_data, columns=['Feature', 'Pearson Correlation Coefficient'])
tab1 = ax.table(cellText=cell_contents.values, colLabels=cell_contents.columns, loc='center')
tab1.auto_set_font_size(False)
tab1.set_fontsize(10)
tab1.auto_set_column_width(col=list(range(len(cell_contents.columns))))
fig.tight_layout()
plt.savefig('allConcs.png')

print('All Red Tide Concentrations:')

for i in range(len(ordering)):
	print('{0:20}{1}'.format(columns_to_compare[ordering[i]], corrs[ordering[i]]))

#Now with only low, very low, and background concentrations
corrs = []

for i in range(len(columns_to_compare)):
	feature = paired_df[columns_to_compare[i]].to_numpy().copy()
	inds = np.argwhere(~np.isnan(feature))
	feature_clean = np.squeeze(feature[inds].copy())
	red_tide_clean = np.squeeze(red_tide[inds].copy())
	inds = np.argwhere(red_tide_clean < 100000)
	feature_clean = np.squeeze(feature_clean[inds].copy())
	red_tide_clean = np.squeeze(red_tide_clean[inds].copy())
	corr, _ = stats.pearsonr(feature_clean, red_tide_clean)
	corrs.append(corr)

corrs = np.array(corrs)
abscorrs = np.abs(corrs)
ordering = np.argsort(abscorrs)
ordering = np.flip(ordering)

for i in range(len(columns_to_compare)):
	if(i==0):
		table_data = [[columns_to_compare[ordering[i]], np.around(corrs[ordering[i]], decimals=3)]]
	else:
		table_data = np.append(table_data, [[columns_to_compare[ordering[i]], np.around(corrs[ordering[i]], decimals=3)]], axis=0)

fig, ax = plt.subplots(dpi=500)
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
cell_contents = pd.DataFrame(table_data, columns=['Feature', 'Pearson Correlation Coefficient'])
tab2 = ax.table(cellText=cell_contents.values, colLabels=cell_contents.columns, loc='center')
tab2.auto_set_font_size(False)
tab2.set_fontsize(10)
tab2.auto_set_column_width(col=list(range(len(cell_contents.columns))))
fig.tight_layout()
plt.savefig('concsBelow100000.png')

print('Only Red Tide Concentrations below 100000:')

for i in range(len(ordering)):
	print('{0:20}{1}'.format(columns_to_compare[ordering[i]], corrs[ordering[i]]))