"""
Module: hashing_function

This module provides functionality to compute hashed values for feature data using
the MD5 hashing algorithm, specifically designed for creating hashed features in
data preprocessing pipelines. It supports both Pandas and PySpark processing libraries
for efficient handling of data.
"""
import hashlib
from pathlib import Path

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.functions import broadcast
from pyspark.sql.types import BooleanType, LongType, StringType

from featurestore.base.utils.singleton import SingletonMeta


def md5_64bit_int_hash(input_string, version=1):
    """
    Generates a 64-bit integer hash from an input string using the MD5 algorithm.

    This function computes an MD5 hash of the given input string, extracts either the
    first 15 or last 15 hexadecimal characters based on the specified version, and
    converts them into a 64-bit integer.

    Args:
        input_string (str): The input string to be hashed.
        version (int, optional): Determines which part of the MD5 hash to use:
            - 0 or 1: Extracts the first 15 hexadecimal characters.
            - 2: Extracts the last 15 hexadecimal characters.
            Default is 1.

    Returns:
        int: A 64-bit integer representation of the MD5 hash
            based on the chosen version.
    """
    md5_hex_hash = hashlib.md5(input_string.encode()).hexdigest()
    if version in [0, 1]:
        return int(md5_hex_hash[:15], 16)
    return int(md5_hex_hash[-15:], 16)


class HashingClass(metaclass=SingletonMeta):
    """
    A singleton class for managing and applying hashing operations on user and item data

    This class provides functionality to hash user, item IDs using predefined mappings,
    handle hash collisions, and generate hashed features for use in data preprocessing
    pipelines. It loads precomputed rehash mappings from files to ensure consistency.

    Attributes:
        data_path (Path): The parent directory path of the raw data files.
        rehash_user_id (list): A preloaded list of user_id that could cause
            hash collisions.
        rehash_item_id (list): A preloaded list of item_id that could cause
            hash collisions.
    """

    def __init__(self, raw_data_path):
        self.data_path = Path(raw_data_path).parent
        self.rehash_user_id = np.load(
            self.data_path / "rehash_user_id.npy",
            allow_pickle=True,
        ).tolist()
        self.rehash_item_id = np.load(
            self.data_path / "rehash_item_id.npy",
            allow_pickle=True,
        ).tolist()

    def hashing_func(
        self,
        df,
        output_feature,
        dependency_col,
        hash_bucket_size,
        process_lib,
        version=0,
        is_dev_env=False,
    ):
        """
        Hashes column values into a bucketed hash space while handling collisions.

        This method applies a hashing function to transform values from a specified
        column into a hashed output column that belongs to a predefined range of
        hash buckets. The method supports collision handling for user and item IDs.

        Args:
            df: The input dataframe containing data to be hashed.
            output_feature (str): The name of the new column to store hashed
                output values.
            dependency_col (str): The column name in the dataframe on which
                the hash is computed.
            hash_bucket_size (int): The total number of assigned hash buckets.
            process_lib (str): The library being used for processing.
            version (int, optional): Specifies the hashing version:
                - 0 or 1: Use the first 15 characters of the MD5 hash.
                - 2: Use the last 15 characters of the MD5 hash.
                Default is 0.

        Returns:
            The dataframe with an added column of hashed values.
        """
        cond_fill = -1
        before_length = after_length = 0

        if "user_id" in dependency_col:
            rehash_id = self.rehash_user_id
        elif "item_id" in dependency_col:
            rehash_id = self.rehash_item_id
        else:
            rehash_id = []

        if version in [0, 1]:
            cond_fill = hash_bucket_size
        elif version == 2:
            cond_fill = 0

        if process_lib == "pyspark":
            if is_dev_env:
                before_length = df.count()

            # Xử lý logic hashing dựa vào version
            md5_expr = (
                F.md5(F.col(dependency_col)).substr(1, 15)
                if version in [0, 1]
                else F.md5(F.col(dependency_col)).substr(18, 15)
            )

            # Thực hiện tất cả các biến đổi trong một bước
            df = df.withColumn(
                output_feature,
                F.conv(md5_expr, 16, 10).cast(LongType()) % hash_bucket_size,
            )

            # Xử lý các va chạm (rehash_id) nếu cần
            if rehash_id and version != 0:
                # Tối ưu bằng việc sử dụng broadcast để tránh shuffle
                if len(rehash_id) < 10000:  # Chỉ sử dụng broadcast với dữ liệu nhỏ
                    rehash_set = set(rehash_id)
                    rehash_broadcast = df.sparkSession.sparkContext.broadcast(
                        rehash_set
                    )

                    @F.udf(BooleanType())
                    def is_rehash_id(value):
                        return value in rehash_broadcast.value

                    df = df.withColumn(
                        output_feature,
                        F.when(
                            is_rehash_id(F.col(dependency_col)), cond_fill
                        ).otherwise(F.col(output_feature)),
                    )
                else:
                    # Với dữ liệu lớn, sử dụng broadcast join
                    collision_df = (
                        df.sparkSession.createDataFrame(rehash_id, StringType())
                        .withColumnRenamed("value", dependency_col)
                        .drop_duplicates()
                    )

                    # Áp dụng broadcast join để tối ưu hóa
                    df = df.join(
                        broadcast(collision_df), on=dependency_col, how="left"
                    ).withColumn(
                        output_feature,
                        F.when(F.col(dependency_col).isNotNull(), cond_fill).otherwise(
                            F.col(output_feature)
                        ),
                    )

            # Kiểm tra độ dài chỉ trong môi trường phát triển
            if is_dev_env:
                after_length = df.count()
                if before_length != after_length:
                    print(
                        f"Số lượng bản ghi thay đổi: từ {before_length} "
                        f"thành {after_length}"
                    )

        return df
