# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by tapfaeff on 11.11.2019.
#  
# Project: tourism_workflow
#
# Description: 
#
import pathlib

import numpy as np
import pandas as pd
from typing import Dict, Sequence, Union


class CsvGenerator:
    """Class used to generate data from a set of csv files.

    More specifically this class:
        - reads data from a set of csv files specified by `filenames`,
          in chunks of size `chunksize`;
        - returns `steps` consecutive data records;
        - uses column information to determine features and targets;
        - if columns named as a target followed by '_masked' are present,
          it either uses them to remove masked data records
          or it forwards them along with the target.
    """

    def __init__(
            self,
            filenames: Sequence[Union[str, pathlib.Path]],
            feature_names: Sequence[str],
            target_names: Sequence[str],
            delimiter: str = ';',
            index_col: int = 0,
            chunksize: int = 10000,
            steps: int = 1,
            drop_remainder: bool = False,
    ):
        self.filenames = np.array(filenames)
        self.feature_names = feature_names
        self.target_names = target_names
        self.delimiter = delimiter
        self.index_col = index_col
        self.chunksize = chunksize
        self.steps = steps
        self.drop_remainder = drop_remainder
        # WARNING: mask columns are expected to have the name of the target
        #          followed by '_masked'
        self.target_masks = {x: x + '_masked' for x in self.target_names}
        self.keys = ('features', ) + tuple(t for t in self.target_names)
        self.output_shapes = {t: (steps, 2) for t in self.target_names}
        self.output_shapes['features'] = (steps, len(self.feature_names))

    def _get_csv_read_iterator(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(filename, sep=self.delimiter,
                           index_col=self.index_col, chunksize=self.chunksize)

    def _complete_masks(self, chunk: pd.DataFrame) -> None:
        for target_mask in self.target_masks.values():
            if target_mask not in chunk.columns:
                # WARNING: assuming all unmasked (0) if column missing
                chunk[target_mask] = 0

    def generate(self) -> Dict[str, np.ndarray]:
        buffer = None
        missing_steps = self.steps
        for filename in self.filenames:
            for chunk in self._get_csv_read_iterator(filename=filename):
                # Read a chunk
                self._complete_masks(chunk)
                chunk_dict = {
                    target: np.stack(
                        (chunk[target].values, chunk[self.target_masks[target]].values),
                        axis=-1,
                    )
                    for target in self.target_names
                }
                chunk_dict['features'] = chunk[self.feature_names].values
                available_steps = len(chunk)
                # Keep yielding until the chunk does not have enough steps
                while available_steps >= missing_steps:
                    remaining_steps = available_steps - missing_steps
                    if remaining_steps == 0:
                        missing_slice = slice(-available_steps, None)
                    else:
                        missing_slice = slice(-available_steps, -remaining_steps)
                    missing_dict = {
                        k: v[missing_slice]
                        for k, v in chunk_dict.items()
                    }
                    if buffer is None:
                        yield missing_dict
                    else:
                        all_dict = {
                            k: np.concatenate((buffer[k], v))
                            for k, v in missing_dict.items()
                        }
                        yield all_dict
                        buffer = None
                    available_steps = remaining_steps
                    missing_steps = self.steps
                # Store insufficient steps in the buffer
                if available_steps > 0:
                    left_slice = slice(-available_steps, None)
                    buffer = {
                        k: v[left_slice]
                        for k, v in chunk_dict.items()
                    }
                    missing_steps -= available_steps
        # After reading every chunk of every file dump the buffer
        if buffer is not None and not self.drop_remainder:
            yield buffer
