# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 23.11.2019.
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas as pd


pd.plotting.register_matplotlib_converters()


def plot_time_series(
        time_series: Sequence[Dict[str, Any]],
        axes: plt.Axes,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        date_format: str = '%d %b %y',
        x_label_rotation: int = 30,
        x_label_alignment: str = 'right',
):
    # Avoid unnecessary whitespace
    if start_date is None:
        start_date = min(min(plotline['x']) for plotline in time_series)
    if end_date is None:
        end_date = max(max(plotline['x']) for plotline in time_series)
    axes.set_xlim(start_date, end_date)
    # Plot all time series
    for ts in time_series:
        # Default style with a simple solid line
        ts_copy = {k: v for k, v in ts.items()}
        if 'fmt' not in ts_copy:
            if 'marker' not in ts_copy:
                ts_copy['marker'] = None
            if 'linestyle' not in ts_copy:
                ts_copy['linestyle'] = 'solid'
        axes.plot_date(**ts_copy)
    # axes.legend(loc='center left', bbox_to_anchor=[1.04, 1])
    # Format x axis labels
    formatter = dates.DateFormatter(date_format)
    axes.xaxis.set_major_formatter(formatter)
    axes.xaxis.set_tick_params(labelrotation=x_label_rotation)
    for label in axes.xaxis.get_ticklabels():
        label.set_horizontalalignment(x_label_alignment)
    return axes
