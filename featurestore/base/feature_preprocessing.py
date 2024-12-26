from pathlib import Path
from typing import Any, List, Optional, Set, Union

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import LongType, StringType

from configs import conf
from featurestore.base.utils.fileops import load_parquet_data, save_parquet_data
from featurestore.base.utils.spark import SparkOperations
from featurestore.hashing_function import HashingClass


class BaseFeaturePreprocessing:
    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename="",
        data_name_to_get_new_dates="",
    ):
        self.process_lib = process_lib
        self.raw_data_dir = Path(raw_data_path)
        self.save_data_dir = self.raw_data_dir / "preprocessed_features"
        self.save_filename = save_filename
        self.filename_date_col = conf.FILENAME_DATE_COL
        self.data_name_to_get_new_dates = data_name_to_get_new_dates
        self.dates_to_extract = self._get_new_dates()

        if self.process_lib == "pyspark":
            self.spark = SparkOperations().get_spark_session()
        else:
            self.spark = None

    def _get_new_dates(self):
        raw_data_path = self.raw_data_dir / f"{self.data_name_to_get_new_dates}.parquet"
        save_data_path = self.save_data_dir / f"{self.save_filename}.parquet"
        if not raw_data_path.exists():
            return None
        raw_dates = set(
            [
                int(file_path.name.split("=")[-1])
                for file_path in raw_data_path.glob(f"{self.filename_date_col}*")
            ]
        )
        saved_dates = set(
            [
                int(file_path.name.split("=")[-1])
                for file_path in save_data_path.glob(f"{self.filename_date_col}*")
            ]
        )
        new_dates: Union[List[int], Set[int]] = raw_dates - saved_dates
        new_dates = sorted(list(new_dates))
        return new_dates

    def _load_raw_data(
        self,
        data_name: str,
        with_columns: Optional[List[str]] = None,
        filters: Optional[List] = None,
        schema: Optional[Any] = None,
    ) -> pd.DataFrame:
        assert (
            self.dates_to_extract is None or len(self.dates_to_extract) > 0
        ), "dates_to_extract must be not empty"
        data_path = self.raw_data_dir / f"{data_name}.parquet"
        date_filters = (
            [(self.filename_date_col, "in", self.dates_to_extract)]
            if self.dates_to_extract
            else None
        )
        if filters:
            filters = (filters + date_filters) if date_filters else filters
        else:
            filters = date_filters
        if self.process_lib == "pandas":
            pandas_filters = filters
            df = load_parquet_data(
                file_paths=data_path,
                with_columns=with_columns,
                process_lib=self.process_lib,
                filters=pandas_filters,
            )
            if len(df) == 0:
                raise ValueError(
                    f"No data found in " f"{data_path} with filters: {pandas_filters}"
                )
        else:
            pyspark_filters = (
                [
                    (
                        item[0],
                        item[1],
                        [str(it) for it in item[2]]
                        if (isinstance(item[2], list) and (item[0] == "filename_date"))
                        else item[2],
                    )
                    for item in filters
                ]
                if filters
                else None
            )
            df = load_parquet_data(
                file_paths=data_path,
                with_columns=with_columns,
                process_lib=self.process_lib,
                filters=pyspark_filters,
                spark=self.spark,
                schema=schema,
            )
            if df.rdd.isEmpty():
                raise ValueError(
                    f"No data found in " f"{data_path} with filters: {pyspark_filters}"
                )
        return df

    def read_processed_data(self):
        pass

    def create_user_key(self, df):
        if self.process_lib in ["pandas"]:
            df["user_id"] = df["profile_id"].copy()
            df.loc[df["user_id"] == -1, "user_id"] = 0
            df["user_id"] = df["user_id"].fillna(0)
            df["user_id"] = df["user_id"].astype(int).astype(str)
            df["username"] = df["username"].str.lower()
            df["user_id"] = df["user_id"] + "#" + df["username"]
            df = df[~df["user_id"].isnull()]
        else:
            df = df.withColumn("user_id", F.col("profile_id"))
            df = df.withColumn(
                "user_id",
                F.when(F.col("user_id").cast(StringType()) == "-1", "0").otherwise(
                    F.col("user_id")
                ),
            )
            df = df.na.fill({"user_id": "0"})
            df = df.withColumn("user_id", F.round(F.col("user_id")).cast(LongType()))
            df = df.withColumn("user_id", F.col("user_id").cast(StringType()))
            df = df.withColumn("username", F.lower(F.col("username")))
            df = df.withColumn(
                "user_id",
                F.concat(F.col("user_id"), F.lit("#"), F.col("username")),
            )
            df = df.filter(F.col("user_id").isNotNull())
        return df

    def create_item_key(self, df):
        if self.process_lib in ["pandas"]:
            df["item_id"] = df["content_type"] + "#" + df["content_id"]
            df = df[~df["item_id"].isnull()]
        else:
            df = df.withColumn(
                "item_id",
                F.concat(F.col("content_type"), F.lit("#"), F.col("content_id")),
            )
            df = df.filter(F.col("item_id").isNotNull())
        return df

    def initialize_dataframe(self, df):
        return df

    def preprocess_feature(self, df):
        return df

    def preprocess_hashed_id(
        self,
        df,
        output_feature_names,
        hash_dependency_info,
        spare_feature_info=conf.SpareFeatureInfo(),
        version=1,
    ):
        if self.process_lib in ["pandas"]:
            for output_feature in output_feature_names:
                dependency_col, hash_bucket_size = self._get_hash_dependency_info(
                    output_feature, hash_dependency_info, spare_feature_info
                )
                df = HashingClass().hashing_func(
                    df,
                    output_feature,
                    dependency_col,
                    hash_bucket_size,
                    "pandas",
                    version,
                )
            return df

        else:
            for output_feature in output_feature_names:
                dependency_col, hash_bucket_size = self._get_hash_dependency_info(
                    output_feature, hash_dependency_info, spare_feature_info
                )
                df = HashingClass().hashing_func(
                    df,
                    output_feature,
                    dependency_col,
                    hash_bucket_size,
                    "pyspark",
                    version,
                )
            return df

    def _get_hash_dependency_info(
        self,
        feature_name,
        hash_dependency_info,
        spare_feature_info=conf.SpareFeatureInfo(),
    ):
        # validate hash_dependency_info
        dependency_col = hash_dependency_info.get(feature_name, None)
        hash_bucket_size = spare_feature_info.hashed_features.get(feature_name, None)
        assert (
            dependency_col is not None
        ), f"Cannot found dependency feature of feature {feature_name}"
        assert (
            hash_bucket_size is not None
        ), f"Cannot found hash bucket size of feature {feature_name}"
        return dependency_col, hash_bucket_size

    def save_preprocessed_data(self, df):
        save_path = self.save_data_dir / f"{self.save_filename}.parquet"
        save_parquet_data(
            df,
            save_path=save_path,
            process_lib=self.process_lib,
        )

    def run(self):
        df = self.read_processed_data()
        df = self.initialize_dataframe(df)
        df = self.preprocess_feature(df)
        self.save_preprocessed_data(df)
