# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by tapfaeff on 17.10.2019.
#  
# Project: tourism_workflow
#
# Description: 
#

from datapreparation.datasets.dataset import Dataset
import pandas as pd
import pathlib
import time


class CodebookLoader(Dataset):
    """ Loads Codebook loader. """
    def __init__(self, codebook_file:pathlib.Path):
        super().__init__()
        self.codebook_file = codebook_file

        if not self.codebook_file.is_file():
            raise ValueError('File not exists', self.codebook_file)

    def load(self, mask_null_values: bool = True):
        """ Load the csv file and store it as df attribute

        :param mask_null_values:
        """
        timer_start = time.process_time()

        df = pd.read_excel(self.codebook_file, sheet_name=0,
                           header=0, skiprows=11)

        elapsed = timer_start - time.process_time()
        self.log.info("Load dataset {} took {:0.4f}sec.".format(self.codebook_file.name, elapsed))
        return df
