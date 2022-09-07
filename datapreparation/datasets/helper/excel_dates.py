# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 25.11.2019.
#  
import datetime

OFFSET = datetime.datetime(year=1899, month=12, day=30).toordinal()


def to_excel_days(date: datetime.datetime) -> int:
    return date.toordinal() - OFFSET


def from_excel_days(days: int) -> datetime.datetime:
    return datetime.datetime.fromordinal(days + OFFSET)
