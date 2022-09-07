# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 21.11.19
#
from datetime import datetime
from typing import Any, Dict, Optional


def date(
        date_string: Optional[str],
        format: str = '%d.%m.%Y',
) -> Optional[datetime]:
    if date_string is None:
        return None
    return datetime.strptime(date_string, format)


def shape_entry(arg: Any) -> Optional[int]:
    try:
        return int(arg)
    except ValueError:
        return None


def str_key_float_value(args_str: str, sep_major: str = ',', sep_minor: str = '=') -> Dict[str, float]:
    args = args_str.split(sep_major)
    output = dict()
    for arg in args:
        if arg.count(sep_minor) != 1:
            raise ValueError('Expected {} to be in the format key{}value.'.format(arg, sep_minor))
        str_key, str_value = arg.split(sep_minor)
        try:
            float_value = float(str_value)
        except ValueError:
            raise ValueError('Expected {} to be convertible to float.'.format(str_value))
        output[str_key] = float_value
    return output
