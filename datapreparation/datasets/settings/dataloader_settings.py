# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by Daniel Pfäffli on 10.10.2019.
#
# Settings for all data loading activities.
#
from settings.settings import Settings


class DataLoaderSettings(Settings):
    """Settings for data-loading"""
    def __init__(self):
        self.data_dir = './data/'  # Directory where the raw data are stored.
        self.train_dir = '/tmp/tourismprediction/'  # Directory to write files to during training.
        self.result_dir = './results/'  # Directory to write the final results to.
        self.file_name = 'normalised.csv'  # File name of the raw data. It must be stored in the data_dir.
        self.column_information_filename = 'translation.csv'  # File name of the Column-information file.
        self.pathto_r_files = "./rmodel/TrainModel.R"  # Path to R files