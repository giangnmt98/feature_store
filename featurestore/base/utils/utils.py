import typing
from datetime import datetime, timedelta

import pandas as pd

from configs.conf import FILENAME_DATE_FORMAT


def get_current_time_stamp() -> str:
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def get_date_before(
    for_date: int,
    num_days_before: int,
) -> int:
    """Get date before for_date.

    Args:
        for_date: The date to get date before.
        num_days_before: The number of days before for_date.
    """
    date_before = pd.to_datetime(for_date, format=FILENAME_DATE_FORMAT) - timedelta(
        days=num_days_before
    )
    date_before = int(date_before.strftime(FILENAME_DATE_FORMAT))
    return date_before


def split_batches(x: typing.Any, batch_size: int) -> typing.List[typing.Any]:
    """
    Split an object (dataframe, array, list, tuple ...) into batches.
    Args:
        x: An object.
        batch_size: The size of each batch.
    Returns:
        A list of dataframes.
    """
    length = len(x)
    if batch_size == -1 or length < batch_size:
        return [x]
    else:
        return [x[i : i + batch_size] for i in range(0, length, batch_size)]


def return_or_load(object_or_path, object_type, load_func):
    """Returns the input directly or load the object from file.
    Returns the input if its type is object_type, otherwise load the object using the
    load_func
    """
    if isinstance(object_or_path, object_type):
        return object_or_path
    return load_func(object_or_path)
