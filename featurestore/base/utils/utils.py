import ast
import typing
from datetime import datetime, timedelta
from glob import glob

import pandas as pd
import yaml

from configs.conf import FILENAME_DATE_COL, FILENAME_DATE_FORMAT, PANDAS_DATE_FORMAT


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


def load_simple_dict_config(path_config):
    with open(path_config) as f:
        config = yaml.safe_load(f)
    return config


def get_date_filters(
    for_date: int,
    num_days_to_load_data: typing.Optional[int],
    is_pyarrow_format: bool = False,
    including_infer_date: bool = False,
) -> typing.List[typing.Any]:
    """Get date filter to avoid reading all data.

    Args:
        for_date: The max date to load data.
        num_days_to_load_data: The maximum number of days on which history data can
            be load.
        is_pyarrow_format: The format of the filter
    """
    date_filters: typing.List[typing.Any] = []
    if including_infer_date:
        date_filters = (
            [f"{FILENAME_DATE_COL} <= {for_date}"]
            if not is_pyarrow_format
            else [(FILENAME_DATE_COL, "<=", for_date)]
        )
    else:
        date_filters = (
            [f"{FILENAME_DATE_COL} < {for_date}"]
            if not is_pyarrow_format
            else [(FILENAME_DATE_COL, "<", for_date)]
        )
    if num_days_to_load_data is not None:
        start_date = get_date_before(for_date, num_days_to_load_data)
        start_date_filter = (
            f"{FILENAME_DATE_COL} >= {start_date}"
            if not is_pyarrow_format
            else (FILENAME_DATE_COL, ">=", start_date)
        )
        date_filters.append(start_date_filter)
    return date_filters


def convert_string_filters_to_pandas_filters(
    filters: typing.List[str],
) -> typing.List[typing.Tuple[str, str, typing.Any]]:
    """Converts a list of string filters to a list of tuple filters.

    Args:
        filters: A list of string filters. Each string filter is a boolean expression
            with 3 components: column name, operator, value. For example,
            ["a > 1", "b == 2"].

    Returns:
        A list of tuple filters. Each tuple filter is a boolean expression with 3
        components: column name, operator, value. For example, [("a", ">", 1), ("b",
        "==", 2)].
    """
    tuple_filters = []
    for f in filters:
        tuple_filter = convert_pandas_query_to_pyarrow_filter(f)
        tuple_filters.append(tuple_filter)
    return tuple_filters


def convert_pandas_query_to_pyarrow_filter(
    query: str,
) -> typing.Tuple[str, str, typing.Any]:
    """
    Convert pandas query to pyarrow filter

    Args:
        query: pandas query string. This is a boolean expression with 3 components:
            column name, operator, value. For example, "a > 1".

    Returns:
        A tuple with 3 components: column name, operator, value. For example,
        ("a", ">", 1).
    """
    ast_op_mapping = {
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
        ast.Is: "==",
        ast.IsNot: "!=",
        ast.In: "in",
        ast.NotIn: "not in",
    }
    ast_node = ast.fix_missing_locations(ast.parse(query))
    assert isinstance(ast_node.body[0], ast.Expr)
    assert isinstance(
        ast_node.body[0].value, ast.Compare
    ), "Only support one condition currently"
    expression = ast_node.body[0].value
    assert isinstance(expression.left, ast.Name)
    column_name = expression.left.id
    op = expression.ops[0]
    op_str = ast_op_mapping[type(op)]
    value = ast.literal_eval(expression.comparators[0])

    return (column_name, op_str, value)


def get_omitted_date(for_date, folder_path, num_days_before=15):
    full_date_list = get_full_date_list(for_date, num_days_before)
    folder_date_list = get_folder_date_list(folder_path)
    omit_date_list = list(set(full_date_list) - set(folder_date_list))
    return omit_date_list


def get_full_date_list(for_date: datetime, num_days_before=15):
    end_date = for_date
    before_date = end_date - timedelta(days=num_days_before)
    end_date_str = end_date.strftime(PANDAS_DATE_FORMAT)
    before_date_str = before_date.strftime(PANDAS_DATE_FORMAT)
    full_date_list = [
        d.strftime(FILENAME_DATE_FORMAT)
        for d in pd.date_range(before_date_str, end_date_str)
    ]
    return full_date_list


def get_folder_date_list(folder_path):
    subfolder_list = glob(str(folder_path) + "*/df*/daily/*/*/*")
    folder_date_list = [
        folder_name.split("daily/")[-1].replace("/", "")
        for folder_name in subfolder_list
    ]
    return folder_date_list
