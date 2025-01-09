"""
Module: daily_data_utils

This module provides utility functions for handling daily date-related operations,
such as retrieving dates relative to a given date. It is designed to assist with
date manipulations based on a predefined date format.
"""
import datetime

import pandas as pd

from configs.conf import FILENAME_DATE_FORMAT


def get_date_before(
    for_date: int,
    num_days_before: int,
) -> int:
    """Get date before for_date.

    Args:
        for_date: The date to get date before.
        num_days_before: The number of days before for_date.
    """
    date_before = pd.to_datetime(
        for_date, format=FILENAME_DATE_FORMAT
    ) - datetime.timedelta(days=num_days_before)
    date_before = int(date_before.strftime(FILENAME_DATE_FORMAT))
    return date_before
