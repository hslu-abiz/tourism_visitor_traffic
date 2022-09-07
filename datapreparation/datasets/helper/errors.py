# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by tapfaeff on 11.10.2019.
#  
# Project: tourism_workflow
#
# Description: Raise if argument is null
#

class ArgumentNullError(Exception):
    def __init__(self, argument_name:str, message:str = None):
        msg = "Argument '{}' is null.".format(argument_name)
        if message is not None:
            msg += "\nMessage: " + message
        super().__init__(msg)

