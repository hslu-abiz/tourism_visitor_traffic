# Copyright 2016 Pfäffli. All Rights Reserved.
#
# Created by Daniel Pfäffli on 01.06.17
#
from datapreparation.datasets.column_information import ColumnInformation

from enum import Enum
import pandas as pd
import numpy as np

import random
import pathlib
import re

class PseudonymisationOperation(Enum):
    POSTCODE_INDEXING = 1
    REPLACE = 2


class ColumnInformationPseudonymisation(ColumnInformation):


    def __init__(self, column_information_path: pathlib.Path = None, df: pd.DataFrame = None, column_information_delimiter: str = ';'):
        
        if column_information_path is None and df is None:
            raise ArgumentNullError(argument_name="column_information_path and df")

        ColumnInformation.__init__(self,
                column_information_path = column_information_path, 
                column_information_delimiter = column_information_delimiter)
        # Add columns for pseudonymisation 
        self.column_pseudo_selection = "Pseudonymisation"
        self.column_pseudo_desc = "Pseudonymisation_Description"
        self.column_pseudo_header_operation = "Pseudonymisation_Header_Operation"
        self.column_pseudo_header_name_original = "Original_Column"
        self.column_pseudo_header_name = "ColumnName"
        self.column_pseudo_header_desc = "Description"
        
        self.new_header_row = [self.column_pseudo_header_name, self.column_pseudo_header_desc]
        self.stats_columns = [self.column_mean,
                 self.column_std,
                 self.column_min,
                 self.column_max,
                 self.column_cardinality,
                 self.column_nan]
        self.additional_columns = []

        # Enum switchers
        self.pseudo_operation_switcher = {
            PseudonymisationOperation.POSTCODE_INDEXING: "Postcode-Indexing",
            PseudonymisationOperation.REPLACE: "replace"
            }
        # Set dataframe
        if df is not None:
            self.df = df
        self.df = self.extract_pseudo_colinfo(self.df)
        # Set new column names 
        self.column_name = self.column_pseudo_header_name

    
    def extract_pseudo_colinfo(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.column_pseudo_header_name_original in df.columns:
            # we did that already
            return df
        instruction_columns = [self.column_operation, self.column_pseudo_selection, self.column_pseudo_desc, self.column_pseudo_header_operation] 
        # append stats columns
        instruction_columns.extend(self.stats_columns )
        def apply_selection(row: pd.Series, colname: str):
            annotated = pd.Series(name=row.name)
            for i, c in enumerate(row[colname].split(',')):
                annotated[self.new_header_row[i]] = row[c.strip()]
            
            for c in instruction_columns:
                annotated[c] = row[c]
            
            return annotated
        
        new_df = df.apply(apply_selection, args=(self.column_pseudo_selection,), axis=1)
        # For confirmability, save original name
        new_df[self.column_pseudo_header_name_original] = new_df[self.column_pseudo_header_name]
        return new_df


    def get_inforows_for_pseudo_operation(self, enm_op):
        if enm_op is None:
            raise ValueError("Please give an operation!")
        cur_series = self.df[self.column_pseudo_header_operation]
        cur_operation = self.pseudo_operation_switcher.get(enm_op)
        cond = cur_series.apply(lambda x: cur_operation in str(x).split(','))
        return self.df.loc[cond, :]

    def get_pseudo_column_operation(self, enm_op):
        return self.get_inforows_for_pseudo_operation(enm_op = enm_op).loc[:, self.column_name].values

    def _process_postcode_replacements(self):
        postcode_reg = re.compile("^(.*)([0-9]{4})(.*)$")
        def get_postcodes(val: str):
            matched = postcode_reg.match(val)
            if matched is None:
                return None
            return matched[2]
        
        def postcode_index_replacement(val: str, lookup_list: list) -> str:
            matched = postcode_reg.match(val)
            if matched is None:
                return None
            postcode = matched[2]
            idx = np.where(postcode==lookup_list)[0][0]
            return f"{matched[1]}Place{idx}{matched[3]}"

        sub = self.get_inforows_for_pseudo_operation(PseudonymisationOperation.POSTCODE_INDEXING).copy()
        postcodes = sub[self.column_pseudo_header_name_original].apply(get_postcodes)
        
        assert postcodes.isnull().sum() == 0
        
        lookup = postcodes.unique()
        random.shuffle(lookup)

        new_name = sub[self.column_pseudo_header_name_original].apply(postcode_index_replacement, args=(lookup,))
        test = new_name.apply(get_postcodes)
        
        assert test.isnull().sum() == test.shape[0]
        sub.loc[:, self.column_pseudo_header_name] = new_name
        return sub
    
    def _process_replace(self):
        sub = self.get_inforows_for_pseudo_operation(PseudonymisationOperation.REPLACE).copy()
        replacements = sub[self.column_pseudo_header_operation].apply(lambda x: [w.strip() for w in x.split(',')][1])
        sub.loc[replacements.index, self.column_pseudo_header_name] = replacements
        return sub

    def create_transformation_column_information(self) -> ColumnInformation:
        # No operation needed
        no_operation = self.df[self.df[self.column_pseudo_header_operation].isnull()].copy()
        # Operation Postcode
        postcode_df = self._process_postcode_replacements()
        # Operation Replace
        replaced_df = self._process_replace()

        df = no_operation.append(
                postcode_df, 
                ignore_index=False,
                verify_integrity=True).append(
                        replaced_df,
                        ignore_index=False,
                        verify_integrity=True)
        # test if there are indices missing
        cond = self.df.index.isin(df.index) == False
        if cond.sum() > 0:
            print("Some rows are not included:")
            print(self.df[cond])
            raise ValueError("Check whether all pseudonymisation operations have been processed!")
        return ColumnInformationPseudonymisation(df=df)

    def get_column_rename_dict(self):
        col_rename = {}

        for idx, row in self.df[[self.column_pseudo_header_name_original, self.column_pseudo_header_name ]].iterrows():
            col_rename.update({row[self.column_pseudo_header_name_original]: row[self.column_pseudo_header_name]})
        
        return col_rename



def get_column_information(trans_colinfo: ColumnInformationPseudonymisation) -> ColumnInformation:
    col_info = ColumnInformation(None)
    column_selection = trans_colinfo.new_header_row
    column_selection.extend(trans_colinfo.stats_columns)
    column_selection.extend(trans_colinfo.additional_columns)
    col_info.set_dataframe(df=trans_colinfo.df[column_selection].copy())
    col_info.column_name = trans_colinfo.column_pseudo_header_name
    return col_info


