# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 21.11.19
#
from typing import Dict


def config_name(
        keyword_abbreviations: Dict[str, str],
        argument_join_string: str = '_',
        iterable_join_string: str = 'x',
        **kwargs,
) -> str:
    config = []
    for keyword, abbreviation in keyword_abbreviations.items():
        if keyword in kwargs:
            value = kwargs[keyword]
            if value is None:
                pass
            elif isinstance(value, bool):
                if value:
                    config.append(abbreviation)
            elif isinstance(value, str) or not hasattr(value, '__iter__'):
                config.append(abbreviation + str(value))
            else:
                joined = iterable_join_string.join(str(el) for el in value)
                config.append(abbreviation + joined)
    return argument_join_string.join(config)
