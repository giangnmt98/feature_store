"""
Module: fileops

This module provides essential file operation utilities for loading, transforming, and
saving data, focusing on Parquet file handling using multiple processing libraries.
It is designed to support flexible workflows for both small and large datasets,
with functionality for filtering, partitioning, and schema management.
"""
import shutil
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from featurestore.base.utils.spark import SparkOperations


def load_parquet_data_by_pyspark(
    file_paths: Union[Path, str, List[Path], List[str]],
    with_columns: Optional[List[str]] = None,
    filters: Optional[List[Any]] = None,
    spark: Optional[Any] = None,
    schema: Optional[Any] = None,
    opt_process=True,
):
    """
    Loads Parquet data using PySpark with optional column selection, filters, and schema

    Args:
        file_paths (Union[Path, str, List[Path], List[str]]): A single file path or
            a list of file paths pointing to Parquet file(s) or directories.
        with_columns (Optional[List[str]]): A list of column names to include in the
            output DataFrame. If not provided, all columns will be included.
        filters (Optional[List[Any]]): A list of filter conditions to apply when loading
            the data. Filters must follow PySpark's supported filter expressions.
        spark (Optional[Any]): An existing PySpark session to use.
        schema (Optional[Any]): An optional schema to use when reading the Parquet
            files. If None, the schema will be inferred automatically.

    Returns:
        pyspark.sql.DataFrame: A PySpark DataFrame.
    """
    if opt_process and filters is not None:
        from pyspark.sql.functions import regexp_extract

        df = None
        partition_column = filters[0][0]
        partition_values = filters[0][2]
        paths = [
            f"{file_paths}/{partition_column}={value}"
            for value in partition_values
            if Path(f"{file_paths}/{partition_column}={value}").exists()
        ]
        # Đọc dữ liệu từ những đường dẫn này
        if with_columns is not None:
            df = (
                spark.read.parquet(*paths)
                .withColumn(
                    partition_column,
                    regexp_extract(
                        F.input_file_name(), f"{partition_column}=(\\d+)", 1
                    ),
                )
                .select(with_columns)
            )
        else:
            df = spark.read.parquet(*paths).withColumn(
                partition_column,
                regexp_extract(F.input_file_name(), f"{partition_column}=(\\d+)", 1),
            )
        return df
    else:
        assert spark is not None
        if isinstance(file_paths, list):
            spark_filenames = [
                item.as_posix() if isinstance(item, Path) else item
                for item in file_paths
            ]
            if schema:
                df = spark.read.schema(schema).parquet(*spark_filenames)
            else:
                df = spark.read.parquet(*spark_filenames)
            if with_columns is not None:
                df = df.select(with_columns)
            if filters is None:
                return df
        else:
            file_paths = (
                file_paths.as_posix() if isinstance(file_paths, Path) else file_paths
            )
            if schema:
                df = (
                    spark.read.option("mergeSchema", "true")
                    .schema(schema)
                    .parquet(file_paths)
                )
            else:
                df = spark.read.option("mergeSchema", "true").parquet(file_paths)

            if with_columns is not None:
                df = df.select(with_columns)

            if filters is None:
                return df

        return df


def __convert_pyarrowschema_to_pandasschema(p, is_pass_null=False):
    """
    Converts a PyArrow schema to a Pandas-compatible schema.

    Args:
        p: The PyArrow schema type.
        is_pass_null (bool): Whether to handle nullable types explicitly.

    Returns:
        A Pandas-compatible dtype or None if the type is unsupported.
    """
    if p == pa.string():
        return np.dtype("O")
    if p == pa.int32():
        return "int32" if not is_pass_null else "Int32"
    if p == pa.int64():
        return "int64" if not is_pass_null else "Int64"
    if p == pa.float32():
        return np.dtype("float32")
    if p == pa.float64():
        return np.dtype("float64")
    return None


def load_parquet_data(
    file_paths: Union[Path, str, List[Path], List[str]],
    with_columns: Optional[List[str]] = None,
    process_lib: str = "pandas",
    filters: Optional[List[Any]] = None,
    spark: Optional[Any] = None,
    schema: Optional[Any] = None,
):
    """
    Loads Parquet data using the specified library (pandas, PySpark).

    Args:
        file_paths (Union[Path, str, List[Path], List[str]]): A single file path or
            list of file paths.
        with_columns (Optional[List[str]]): Names of columns to include in the DataFrame
        process_lib (str): The library to process the data.
        filters (Optional[List[Any]]): Filter conditions to apply to the data.
        spark (Optional[Any]): An active PySpark session if using PySpark.
        schema (Optional[Any]): Schema to use while loading the data.
            If None, inferred automatically.

    Returns:
        DataFrame depending on the selected library.
    """
    return load_parquet_data_by_pyspark(
        file_paths=file_paths,
        with_columns=with_columns,
        filters=filters,
        spark=spark,
        schema=schema,
    )


def filters_by_expression_in_pyspark(df, filters):
    """
    Applies multiple filter conditions to a PySpark DataFrame.

    Args:
        df (pyspark.sql.DataFrame): The PySpark DataFrame to filter.
        filters (List[Tuple[str, str, Any]]): A list of filter expressions. Each filter
            expression is a tuple containing:
            - `col` (str): The column name on which the filter is applied.
            - `op` (str): The filter operator
            - `val` (Any): The value to filter the column against.

    Returns:
        pyspark.sql.DataFrame: The filtered DataFrame.
    """
    for fil in filters:
        assert len(fil) == 3
        col = fil[0]
        op = fil[1]
        val = fil[2]
        df = process_filter(df, col, op, val)
    return df


def process_filter(df, col, op, val):
    """
    Applies a filter operation to a PySpark DataFrame.

    This method performs various filtering operations on a given PySpark DataFrame
    based on the specified column, operator, and value.

    Args:
        df (pyspark.sql.DataFrame): The PySpark DataFrame to apply the filter to.
        col (str): The column name on which the filter operation is applied.
        op (str): The filter operator. Supported operators include:
            - "in": Checks if the column values are in a provided list or DataFrame.
            - "not in": Checks if the column values are not in a provided list or df.
            - "=", "==": Checks for equality.
            - "!=": Checks for inequality.
            - "<": Less than comparison.
            - ">": Greater than comparison.
            - "<=": Less than or equal to comparison.
            - ">=": Greater than or equal to comparison.
        val (Any): The value to compare against. Can be a single value, a list, or
            another DataFrame, depending on the operator.

    Returns:
        pyspark.sql.DataFrame: The filtered DataFrame.
    """
    if op == "in":
        df = process_in_operator(df, col, val)
    elif op == "not in":
        df = process_not_in_operator(df, col, val)
    elif op in ["=", "=="]:
        df = df.filter(F.col(col) == val)
    elif op == "<":
        df = df.filter(F.col(col) < val)
    elif op == ">":
        df = df.filter(F.col(col) > val)
    elif op == "<=":
        df = df.filter(F.col(col) <= val)
    elif op == ">=":
        df = df.filter(F.col(col) >= val)
    elif op == "!=":
        df = df.filter(F.col(col) != val)
    else:
        raise ValueError(f'"{(col, op, val)}" is not a valid operator in predicates.')
    return df


def process_in_operator(df, col, val):
    """
    Applies an "in" filter operation to a PySpark DataFrame.

    Args:
        df (pyspark.sql.DataFrame): The PySpark DataFrame to filter.
        col (str): The column name on which the "in" operation is applied.
        val (Union[List[Any], pyspark.sql.DataFrame]): A list or another DataFrame
            containing the values to check for inclusion.

    Returns:
        pyspark.sql.DataFrame: The filtered DataFrame.
    """
    if not isinstance(val, DataFrame):
        df = df.filter(F.col(col).isin(val))
    else:
        df = df.join(val, on=col, how="inner")
    return df


def process_not_in_operator(df, col, val):
    """
    Applies a "not in" filter operation to a PySpark DataFrame.

    Args:
        df (pyspark.sql.DataFrame): The PySpark DataFrame to filter.
        col (str): The column name on which the "not in" operation is applied.
        val (Union[List[Any], pyspark.sql.DataFrame]): A list or another DataFrame
            containing the values to check for exclusion.

    Returns:
        pyspark.sql.DataFrame: The filtered DataFrame.
    """
    if not isinstance(val, DataFrame):
        df = df.filter(~F.col(col).isin(val))
    else:
        df = df.join(val, on=col, how="leftanti")
    return df


def save_parquet_data(
    df,
    save_path: Union[Path, str],
    partition_cols: Optional[List[str]] = None,
    process_lib: str = "pandas",
    overwrite: bool = True,
    existing_data_behavior: str = "delete_matching",
    schema: Optional[Any] = None,
):
    """
    Saves a DataFrame as Parquet files.

    Args:
        df: dataframe to save
        save_path: path to save
        partition_cols: list of partition columns
        process_lib: process library, only support pandas currently
        overwrite: overwrite if save_path exists
        existing_data_behavior: Controls how the dataset will handle data that already
            exists in the destination. More details in
            https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset
    """
    if process_lib == "pandas":
        save_parquet_data_by_pandas(
            df,
            save_path=save_path,
            partition_cols=partition_cols,
            overwrite=overwrite,
            existing_data_behavior=existing_data_behavior,
            schema=schema,
        )
    else:
        save_parquet_data_by_pyspark(
            df,
            save_path=save_path,
            partition_cols=partition_cols,
            overwrite=overwrite,
            schema=schema,
        )


def save_parquet_data_by_pandas(
    df,
    save_path: Union[Path, str],
    partition_cols: Optional[List[str]] = None,
    overwrite: bool = True,
    existing_data_behavior: str = "delete_matching",
    schema: Optional[Any] = None,
):
    """
    Saves a pandas DataFrame as a Parquet dataset.

    This method converts a pandas DataFrame into a PyArrow table and saves it to the
    specified location in Parquet format.

    Args:
        df (pandas.DataFrame): The DataFrame to save.
        save_path (Union[Path, str]): The destination path to save the Parquet dataset.
        partition_cols (Optional[List[str]]): A list of column names to partition the
            dataset by. Defaults to None.
        overwrite (bool): If True, overwrites the existing dataset at the destination.
            Defaults to True.
        existing_data_behavior (str): Controls how existing data at the destination
            is handled. Defaults to "delete_matching".
        schema (Optional[Any]): A schema to enforce on the DataFrame before saving.
            If provided, the DataFrame will be cast to the specified schema.
    """
    if overwrite and Path(save_path).exists():
        shutil.rmtree(save_path)
    if schema:
        df = df[list(schema.keys())].astype(schema)
    pa_table = pa.Table.from_pandas(df)
    pq.write_to_dataset(
        pa_table,
        root_path=save_path,
        existing_data_behavior=existing_data_behavior,
        partition_cols=partition_cols,
    )


def save_parquet_data_by_pyspark(
    df,
    save_path: Union[Path, str],
    partition_cols: Optional[List[str]] = None,
    overwrite: bool = True,
    schema: Optional[Any] = None,
):
    """
    Saves a pyspark DataFrame as a Parquet dataset.

    Args:
        df (pandas.DataFrame): The DataFrame to save.
        save_path (Union[Path, str]): The destination path to save the Parquet dataset.
        partition_cols (Optional[List[str]]): A list of column names to partition the
            dataset by. Defaults to None.
        overwrite (bool): If True, overwrites the existing dataset at the destination.
            Defaults to True.
        schema (Optional[Any]): A schema to enforce on the DataFrame before saving.
            If provided, the DataFrame will be cast to the specified schema.
    """
    mode = "overwrite"
    if not overwrite:
        mode = "append"
    if isinstance(save_path, Path):
        to_save_path = save_path.as_posix()
    else:
        to_save_path = save_path

    if schema:
        spark = SparkOperations().get_spark_session()
        df = spark.createDataFrame(df.select(schema.names).rdd, schema)

    if partition_cols is None:
        df.write.option("header", True).mode(mode).parquet(to_save_path)
    else:
        df.write.option("header", True).partitionBy(partition_cols).mode(mode).parquet(
            to_save_path
        )
