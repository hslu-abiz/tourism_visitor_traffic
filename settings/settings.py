# Copyright 2017 HSLU. All Rights Reserved.
#
# Created by Thomas Koller on 12.02.17
#
"""Settings for experiments"""

import json


class Settings:
    def __repr__(self):
        return json.dumps(self.__dict__, indent=True, sort_keys=True)

    def set_from_flags(self, args: dict):
        """Set values from the flags, but only if they are not None in the flags"""
        for key in self.__dict__.keys():
            if key in args and args[key] is not None:
                self.__dict__[key] = args[key]

    def load(self, representation: str):
        loaded = json.loads(representation)
        self.__dict__ = loaded
