import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

paired_df = pd.read_pickle('paired_dataset.pkl')

features_to_use=['Red Tide Concentration', 'aot_869', 'angstrom', 'Rrs_412', 'Rrs_443', \
                 'Rrs_469', 'Rrs_488', 'Rrs_531', 'Rrs_547', 'Rrs_555', 'Rrs_645', \
                 'Rrs_667', 'Rrs_678', 'chlor_a', 'chl_ocx', 'Kd_490', 'pic', 'poc', \
                 'ipar', 'nflh', 'par']

paired_df = paired_df[features_to_use]

#Remove samples with NaN values
paired_df = paired_df.dropna()

input_features = paired_df[features_to_use[1:]].to_numpy()
output_feature = paired_df['Red Tide Concentration'].to_numpy()

classes = np.zeros((output_feature.shape[0], 1))

for i in range(len(classes)):
	if(output_feature[i] < 100000):
		classes[i] = 0
	else:
		classes[i] = 1

classes = np.squeeze(classes)

forest = RandomForestClassifier(n_estimators=340)

selector = RFE(forest, n_features_to_select=7, step=1)

selector.fit(input_features, classes)

feature_order = np.argsort(selector.ranking_)

for i in range(len(selector.ranking_)):
	print('{}: {}'.format(features_to_use[1:][feature_order[i]], selector.ranking_[feature_order[i]]))