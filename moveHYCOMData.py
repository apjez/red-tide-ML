import numpy as np
from os import listdir
from os.path import isfile, join

start_folder = 'HYCOM_data/'
save_folder = '/run/media/rfick/UF10/HYCOM/expt_50.1_data/'

dates = np.load('/run/media/rfick/UF10/HYCOM/expt_50.1_dates.npy')

onlyfiles = [f for f in listdir(start_folder) if isfile(join(start_folder, f))]

for i in range(len(onlyfiles)):
	if(i%100 == 0):
		print('{}/{}'.format(i, len(onlyfiles)))

	file_date = onlyfiles[i][0:19]
	remaining_file = onlyfiles[i][19:]

	date_ind = np.where(dates == file_date)[0][0]

	data = np.load(start_folder+onlyfiles[i])

	np.save(save_folder+str(date_ind)+remaining_file, data)