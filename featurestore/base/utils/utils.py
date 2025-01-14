"""
Module: utils

This module provides utility functions for handling data processing tasks such as date
manipulation, batch splitting, configuration loading, and filter conversions.
These utilities facilitate preprocessing, managing datasets, and working with
configurations in both pandas and PyArrow-supported workflows.
"""
import ast
import typing
from datetime import datetime, timedelta
from glob import glob

import pandas as pd
import pyspark.sql.functions as F
import yaml

from configs.conf import FILENAME_DATE_COL, FILENAME_DATE_FORMAT, PANDAS_DATE_FORMAT
from featurestore.daily_data_utils import get_date_before


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
    """
    Loads a configuration file and returns its content as a dictionary.

    This function reads a configuration file in YAML format from the given path
    and parses it into a Python dictionary.

    Args:
        path_config (str): The file path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration data.
    """
    with open(path_config, encoding="utf8") as f:
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
    """
    Identifies dates that are missing from a folder within a given date range.

    Args:
        for_date (str): The target date in a format compatible with `get_full_date_list`
        folder_path (str): The path to the folder containing date-labeled data.
        num_days_before (int, optional): The number of days before the target date to
            include in the range. Defaults to 15.

    Returns:
        list: A list of dates that are in the full range but missing from the folder.
    """
    full_date_list = get_full_date_list(for_date, num_days_before)
    folder_date_list = get_folder_date_list(folder_path)
    omit_date_list = list(set(full_date_list) - set(folder_date_list))
    return omit_date_list


def get_full_date_list(for_date: datetime, num_days_before=15):
    """
    Generates a list of dates within a specified range leading up to a given date.

    Args:
        for_date (datetime): The target date up to which the date range is generated.
        num_days_before (int, optional): The number of days before the target date to
            include in the range. Defaults to 15.

    Returns:
        list: A list of dates, formatted as strings, from `num_days_before` the target
            date to `for_date`.
    """
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
    """
    Extracts a list of dates from folder structures within a specified directory.

    This function searches for subfolder paths matching a specific pattern under the
    provided `folder_path` and extracts date information from the folder structure.
    The dates are got from the part of the folder path following the `daily/` keyword.

    Args:
        folder_path (str): The root path to search for matching subfolders.

    Returns:
        list: A list of dates as strings, extracted from the folder names matching
            the search pattern.
    """
    subfolder_list = glob(str(folder_path) + "/*")
    folder_date_list = [folder_name.split("/")[-1] for folder_name in subfolder_list]
    return folder_date_list


def norm_content_category(item):
    """
    Normalizes the content category column values.

    Args:
        content_category (str): The raw content category as input.

    Returns:
        str: The normalized content category or 'unknown' if the input cannot
            be processed.
    """
    item = str(item)
    if (
        ("hài kịch" in item.lower())
        | ("hài" in item.lower())
        | ("hước" in item.lower())
    ):
        res = "hai"
    elif (
        ("Kinh dị" in item) | ("kinh dị" in item.lower()) | ("hình sự" in item.lower())
    ):
        res = "hinhsu_kinhdi"
    elif (
        (" Thiếu nhi" == item)
        | ("teen" in item.lower())
        | ("Bé" in item)
        | ("thiếu nhi" in item.lower())
        | ("gia đình" in item.lower())
    ):
        res = "kid_giadinh"
    elif ("cải lương" in item.lower()) | ("nhạc" in item.lower()):
        res = "nhac_tt"
    elif (
        ("edm" in item.lower()) | ("hiphop" in item.lower()) | ("kpop" in item.lower())
    ):
        res = "nhac_hd"
    elif (
        ("yoga" in item.lower())
        | ("trang điểm" in item.lower())
        | ("đẹp" in item.lower())
        | ("thẩm mỹ" in item.lower())
        | ("sức khỏe" in item.lower())
        | ("chăm sóc" in item.lower())
    ):
        res = "suckhoe"
    elif (
        ("24h" in item.lower())
        | ("thời cuộc" in item.lower())
        | ("thời sự" in item.lower())
        | ("tin" in item.lower())
    ):
        res = "tintuc"
    elif (
        ("pool".lower() in item.lower())
        | ("u20".lower() in item.lower())
        | ("olympic".lower() in item.lower())
        | ("Đô vật".lower() in item.lower())
        | ("võ" in item.lower())
        | ("bình luận" in item.lower())
        | ("esport" in item.lower())
        | ("cầu lông" in item.lower())
        | ("f1" in item.lower())
        | ("thể thao" in item.lower())
        | ("tennis" in item.lower())
        | ("bóng" in item.lower())
        | ("quyền anh" in item.lower())
        | ("chung kết" in item.lower())
        | ("vòng Loại" in item.lower())
        | ("cup" in item.lower())
    ):
        res = "thethao"
    elif (
        ("tiếng anh" in item.lower())
        | ("chinh phục" in item.lower())
        | ("lớp" in item.lower())
        | ("bài học" in item.lower())
        | ("tư duy" in item.lower())
        | ("toeic" in item.lower())
        | ("ielts" in item.lower())
        | ("tiếng Anh" in item.lower())
        | ("du học" in item.lower())
        | ("ngữ pháp" in item.lower())
    ):
        res = "giaoduc"
    elif (
        ("hollywood" in item.lower())
        | ("rạp" in item.lower())
        | ("galaxy" in item.lower())
        | ("hbo" in item.lower())
        | ("độc quyền" in item.lower())
    ):
        res = "traphi"
    elif (
        ("GameShow" in item)
        | ("chương trình" in item.lower())
        | ("gameShow" in item.lower())
        | ("tài liệu" in item.lower())
    ):
        res = "gameshow"
    else:
        res = "others"
    return res


def norm_content_category_by_pyspark(df):
    """
    Normalizes the `content_category` column in a PySpark DataFrame.

    Args:
        df (pyspark.sql.DataFrame): The input PySpark DataFrame containing a
            column named `content_category`.

    Returns:
        pyspark.sql.DataFrame: The DataFrame with an additional column
            `clean_content_category` representing the normalized content categories.
    """
    df = df.withColumn(
        "clean_content_category",
        F.when(
            F.col("content_category").rlike(".*hài kịch.*")
            | F.col("content_category").rlike(".*hài.*")
            | F.col("content_category").rlike(".*hước.*"),
            "hai",
        )
        .when(
            F.col("content_category").rlike(".*Kinh dị.*")
            | F.col("content_category").rlike(".*kinh dị.*")
            | F.col("content_category").rlike(".*hình sự.*"),
            "hinhsu_kinhdi",
        )
        .when(
            F.col("content_category").like(" Thiếu nhi")
            | F.col("content_category").rlike(".*teen.*")
            | F.col("content_category").rlike(".*Bé.*")
            | F.col("content_category").rlike(".*thiếu nhi.*")
            | F.col("content_category").rlike(".*gia đình.*"),
            "kid_giadinh",
        )
        .when(
            F.col("content_category").rlike(".*cải lương.*")
            | F.col("content_category").rlike(".*nhạc.*"),
            "nhac_tt",
        )
        .when(
            F.col("content_category").rlike(".*edm.*")
            | F.col("content_category").rlike(".*hiphop.*")
            | F.col("content_category").rlike(".*kpop.*"),
            "nhac_hd",
        )
        .when(
            F.col("content_category").rlike(".*yoga.*")
            | F.col("content_category").rlike(".*trang điểm.*")
            | F.col("content_category").rlike(".*đẹp.*")
            | F.col("content_category").rlike(".*thẩm mỹ.*")
            | F.col("content_category").rlike(".*sức khỏe.*")
            | F.col("content_category").rlike(".*chăm sóc.*"),
            "suckhoe",
        )
        .when(
            F.col("content_category").rlike(".*24h.*")
            | F.col("content_category").rlike(".*thời cuộc.*")
            | F.col("content_category").rlike(".*thời sự.*")
            | F.col("content_category").rlike(".*tin.*"),
            "tintuc",
        )
        .when(
            F.col("content_category").rlike(".*pool.*")
            | F.col("content_category").rlike(".*u20.*")
            | F.col("content_category").rlike(".*olympic.*")
            | F.col("content_category").rlike(".*Đô vật.*")
            | F.col("content_category").rlike(".*võ.*")
            | F.col("content_category").rlike(".*bình luận.*")
            | F.col("content_category").rlike(".*esport.*")
            | F.col("content_category").rlike(".*cầu lông.*")
            | F.col("content_category").rlike(".*f1.*")
            | F.col("content_category").rlike(".*thể thao.*")
            | F.col("content_category").rlike(".*tennis.*")
            | F.col("content_category").rlike(".*bóng.*")
            | F.col("content_category").rlike(".*quyền anh.*")
            | F.col("content_category").rlike(".*chung kết.*")
            | F.col("content_category").rlike(".*vòng Loại.*")
            | F.col("content_category").rlike(".*cup.*"),
            "thethao",
        )
        .when(
            F.col("content_category").rlike(".*tiếng anh.*")
            | F.col("content_category").rlike(".*chinh phục.*")
            | F.col("content_category").rlike(".*lớp.*")
            | F.col("content_category").rlike(".*bài học.*")
            | F.col("content_category").rlike(".*tư duy.*")
            | F.col("content_category").rlike(".*toeic.*")
            | F.col("content_category").rlike(".*ielts.*")
            | F.col("content_category").rlike(".*tiếng Anh.*")
            | F.col("content_category").rlike(".*du học.*")
            | F.col("content_category").rlike(".*ngữ pháp.*"),
            "giaoduc",
        )
        .when(
            F.col("content_category").rlike(".*hollywood.*")
            | F.col("content_category").rlike(".*rạp.*")
            | F.col("content_category").rlike(".*galaxy.*")
            | F.col("content_category").rlike(".*hbo.*")
            | F.col("content_category").rlike(".*galaxy.*"),
            "traphi",
        )
        .when(
            F.col("content_category").rlike(".*GameShow.*")
            | F.col("content_category").rlike(".*chương trình.*")
            | F.col("content_category").rlike(".*gameShow.*")
            | F.col("content_category").rlike(".*tài liệu.*"),
            "gameshow",
        )
        .otherwise("others"),
    )
    return df
