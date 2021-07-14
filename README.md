# Extract FWC Data

This repository works with data from MODIS-AQUA (https://oceancolor.gsfc.nasa.gov/cgi/browse.pl?sen=amod), as well as in situ data from FWC (https://myfwc.com/research/redtide/monitoring/database/). It pairs pixels from the remote sensing as well as from the in situ data to create a paired dataset. It also contains code to train and evaluate models based on that paired data.

Note that data products (both raw and derived) are not stored in this repo, but they are described here.

### Data Products:

* angstrom_sums.npy: Derived data product containing information about angstrom mean and standard deviation based on depth.

* chl_ocx_sums.npy: Derived data product containing information about chl_ocx mean and standard deviation based on depth.

* chlor_a_sums.npy: Derived data product containing information about chlor_a mean and standard deviation based on depth.

* Kd_490_sums.npy: Derived data product containing information about Kd_490 mean and standard deviation based on depth.

* nflh_sums.npy: Derived data product containing information about nflh mean and standard deviation based on depth.

* poc_sums.npy: Derived data product containing information about poc mean and standard deviation based on depth.

* ETOPO1_Bed_g_gmt4.grd: Raw bedrock depth data from NOAA's ETOPO1 Global Relief Model.

* florida_x.npy: Derived data product of ETOPO1 Global Relief model covering Florida.

* florida_y.npy: Derived data product of ETOPO1 Global Relief model covering Florida.

* florida_z.npy: Derived data product of ETOPO1 Global Relief model covering Florida.

* paired_dataset.pkl: Derived data product containing pairs of remotely sensed pixels with in situ ground measures.

* PinellasMonroeCoKareniabrevis 2010-2020.06.12.xlsx: Raw in situ data from FWC.

* depth_stats/: Derived data products containing global statistics of various features based on depth.

* saved_model_info/: Contains saved model parameters after training.

* /run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/requested_files/: Raw MODIS-AQUA data from Florida between 2000-2020.

### Functions:

* SotoEtAlDetector.py: Implements a red tide detection method from Soto et al (https://ioccg.org/wp-content/uploads/2021/05/ioccg_report_20-habs-2021-web.pdf, Chapter 6).

* analyze_paired_data.py: Takes paired dataset and calculates correlation coefficients across the dataset.

* convertFeaturesByDepth.py: Helper functions to normalize features by their mean and standard deviation according to depth.

* dataset.py: Helper function to store data and labels into a torch dataset.

* depth_stats.py: Goes through all historical MODIS-AQUA and aggregates information to compute mean and standard deviation.

* findMatrixCoords.py: Finds the correct matrix coordinates in MODIS data to match desired lat/lon coordinates.

* findMatrixCoordsBedrock.py: Finds the correct matrix coordinates in bedrock lat/lon to match desired lat/lon coordinates.

* model.py: Pytorch classification model definition.

* permu_importance.py: Tests saved models using permuation importance to understand the importance of various features. Also gives overall accuracies.

* processFiles.py: Takes raw MODIS-AQUA data and raw in situ red tide measurements and creates a dataset by pairing them.

* test_depth.py: Takes aggregate information from depth_stats.py and computes mean and standard deviation as well as plotting information vs depth.

* trainModel.py: Trains and saves pytorch models.

* utils.py: Utility functions