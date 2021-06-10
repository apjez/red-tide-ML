import os

def ensure_folder(folder):
	if not os.path.isdir(folder):
		os.mkdir(folder)