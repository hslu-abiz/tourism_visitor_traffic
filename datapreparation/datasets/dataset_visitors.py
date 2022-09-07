# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by tapfaeff on 10.10.2019.
#  
# Project: tourism_workflow
#
# Description: 
#

from datapreparation.datasets.dataset import Visitor
from datapreparation.datasets.dataset_transformations import Checkpoint, OneToMany

import logging

class CheckpointVisitor(Visitor):
    """ Gathers checkpoints. """
    def __init__(self):
        self.checkpoints = {}
        self.log = logging.getLogger(self.__class__.__name__)

    def visit(self, dataset:object):
        if isinstance(dataset, Checkpoint):
            if dataset.name in self.checkpoints.keys():
                self.log.warning("Key {} already in dictionary. Value is overwritten.".format(dataset.name))
            self.checkpoints.update({dataset.name:  dataset.checkpoint_df})


class OneToManyVisitor(Visitor):
    """ Gathers the information of a one to many transformation. """
    def __init__(self):
        self.list_column_tuples = []

    def visit(self, dataset:object):
        if isinstance(dataset, OneToMany):
            self.list_column_tuples.append(dataset.created_columns_tuple)
