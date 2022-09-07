# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by Daniel Pfäffli on 10.10.2019.
#
# Settings for all data loading activities.
#
from datapreparation.datasets.helper.nullvalues import mask_nullvalues_median

import pandas as pd
import pathlib
import logging
import time


class Visitor(object):
    def visit(self, dataset:object):
        pass


class Dataset(object):
    """ Dataset base class for a classical decorator pattern. """
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.timer_start = time.process_time()
        self.timer_stop = None

    def load(self) -> pd.DataFrame:
        pass

    def accept(self, visitor:Visitor):
        visitor.visit(self)


class Transformation(Dataset):
    """ Applies transformations during load. """
    def __init__(self, dataset:Dataset):
        super().__init__()
        self.dataset = dataset

    def load(self) -> pd.DataFrame:
        return self.dataset.load()

    def accept(self, visitor:Visitor):
        super().accept(visitor = visitor)
        self.dataset.accept(visitor = visitor)
