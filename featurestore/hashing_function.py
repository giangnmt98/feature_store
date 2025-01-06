import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import LongType, StringType

from featurestore.base.utils.singleton import SingletonMeta


def md5_64bit_int_hash(input_string, version=1):
    md5_hex_hash = hashlib.md5(input_string.encode()).hexdigest()
    if version == 0 or version == 1:
        return int(md5_hex_hash[:15], 16)
    elif version == 2:
        return int(md5_hex_hash[-15:], 16)


class HashingClass(metaclass=SingletonMeta):
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
    ):
        if "user_id" in dependency_col:
            rehash_id = self.rehash_user_id
        elif "item_id" in dependency_col:
            rehash_id = self.rehash_item_id
        else:
            rehash_id = []

        if version == 0 or version == 1:
            cond_str = "1"
            cond_fill = hash_bucket_size
        elif version == 2:
            cond_str = "0"
            cond_fill = 0

        if process_lib == "pandas":
            before_length = len(df)
            df[output_feature] = (
                df[dependency_col]
                .astype(str)
                .apply(lambda x: md5_64bit_int_hash(x, version) % hash_bucket_size)
            )

            if rehash_id != [] and version != 0:
                collision_df = pd.DataFrame(
                    rehash_id, columns=[dependency_col]
                ).drop_duplicates()
                collision_df["cond"] = "1"
                df = df.merge(collision_df, on=dependency_col, how="left").fillna(
                    {"cond": "0"}
                )
                df.loc[df["cond"] == cond_str, output_feature] = cond_fill
                df = df.drop(columns=["cond"])
            after_length = len(df)

        if process_lib == "pyspark":
            before_length = df.count()
            if version == 0 or version == 1:
                df = df.withColumn(
                    "tmp",
                    F.md5(F.col(dependency_col)).substr(1, 15),
                )

            elif version == 2:
                df = df.withColumn(
                    "tmp",
                    F.md5(F.col(dependency_col)).substr(18, 15),
                )

            df = df.withColumn(
                output_feature,
                F.conv(F.col("tmp"), 16, 10).cast(LongType()) % hash_bucket_size,
            ).drop("tmp")

            if rehash_id != [] and version != 0:
                collision_df = (
                    df.sparkSession.createDataFrame(rehash_id, StringType())
                    .withColumnRenamed("value", dependency_col)
                    .drop_duplicates()
                )
                collision_df = collision_df.withColumn("cond", F.lit("1"))
                df = df.join(collision_df, on=dependency_col, how="left").na.fill(
                    {"cond": "0"}
                )

                df = df.withColumn(
                    output_feature,
                    F.when(F.col("cond") == cond_str, cond_fill).otherwise(
                        F.col(output_feature)
                    ),
                ).drop("cond")
            after_length = df.count()
        assert (
            before_length == after_length
        ), f"different length {before_length} {after_length}"
        return df
