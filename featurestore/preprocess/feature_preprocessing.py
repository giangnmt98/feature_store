"""
Module: feature_preprocessing

This module contains classes and functions for preprocessing feature data for various
datasets, including user, account, profile, content, and interaction data.
"""
import numpy as np
import pandas as pd
import pyspark.sql.functions as F

from configs import conf
from featurestore.base.feature_preprocessing import BaseFeaturePreprocessing
from featurestore.base.utils.fileops import load_parquet_data
from featurestore.constants import DataName


class AccountFeaturePreprocessing(BaseFeaturePreprocessing):
    """
    Preprocessing logic for account-related feature data.

    Attributes:
        process_lib (str): Library used for processing ('pandas' or 'pyspark').
        raw_data_path (str): Path to the raw data directory.
    """

    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
    ):
        super().__init__(process_lib, raw_data_path)

    def read_processed_data(self):
        data_path = self.raw_data_dir / f"{DataName.ACCOUNT_MYTV_INFO}.parquet"
        df = load_parquet_data(
            file_paths=data_path,
            process_lib=self.process_lib,
            spark=self.spark,
        )
        self.raw_data[DataName.ACCOUNT_MYTV_INFO] = df

    def initialize_dataframe(self):
        # create feature values for new users
        if self.process_lib == "pandas":
            new_values = {
                "username": ["empty"],
                "province": ["HCM"],
                "package_code": ["MYTV021"],
                "sex": ["1"],
                "age": [0],
                "platform": ["b2c-android"],
                "profile_id": [0],
            }
            new_df = pd.DataFrame(data=new_values)
            df = pd.concat([new_df, self.raw_data[DataName.ACCOUNT_MYTV_INFO]])
        else:
            columns = [
                "username",
                "province",
                "package_code",
                "sex",
                "age",
                "platform",
                "profile_id",
            ]
            vals = [("empty", "HCM", "MYTV021", "1", 0, "b2c-android", 0)]
            new_df = self.spark.createDataFrame(vals, columns)
            df = new_df.union(self.raw_data[DataName.ACCOUNT_MYTV_INFO].select(columns))
        return df

    def preprocess_feature(self, df):
        if self.process_lib in ["pandas"]:
            df.loc[(df.age <= 5) | (df.age >= 95), "age"] = np.nan

            df["age_group"] = np.nan
            df.loc[df.age < 15, "age_group"] = "child"
            df.loc[(df.age >= 15) & (df.age < 22), "age_group"] = "student"
            df.loc[(df.age >= 22) & (df.age < 30), "age_group"] = "play"
            df.loc[(df.age >= 30) & (df.age < 40), "age_group"] = "married"
            df.loc[(df.age >= 40) & (df.age < 65), "age_group"] = "senior"
            df.loc[(df.age >= 65), "age_group"] = "older"
            df.loc[
                ~df.package_code.isin(conf.valid_package_code.keys()), "package_code"
            ] = np.nan
        else:
            df = df.withColumn(
                "age",
                F.when(
                    (F.col("age") >= 95) | (F.col("age") <= 5), F.lit(None)
                ).otherwise(F.col("age")),
            )
            df = df.withColumn("age_group", F.lit(None))
            df = df.withColumn(
                "age_group",
                F.when(F.col("age") < 15, F.lit("child")).otherwise(F.col("age_group")),
            )
            df = df.withColumn(
                "age_group",
                F.when(
                    (F.col("age") >= 15) & (F.col("age") < 22), F.lit("student")
                ).otherwise(F.col("age_group")),
            )
            df = df.withColumn(
                "age_group",
                F.when(
                    (F.col("age") >= 22) & (F.col("age") < 30), F.lit("play")
                ).otherwise(F.col("age_group")),
            )
            df = df.withColumn(
                "age_group",
                F.when(
                    (F.col("age") >= 30) & (F.col("age") < 40), F.lit("married")
                ).otherwise(F.col("age_group")),
            )
            df = df.withColumn(
                "age_group",
                F.when(
                    (F.col("age") >= 40) & (F.col("age") < 65), F.lit("senior")
                ).otherwise(F.col("age_group")),
            )
            df = df.withColumn(
                "age_group",
                F.when((F.col("age") >= 65), F.lit("older")).otherwise(
                    F.col("age_group")
                ),
            )
            df = df.withColumn(
                "package_code",
                F.when(
                    F.col("package_code").isin(list(conf.valid_package_code.keys())),
                    F.col("package_code"),
                ).otherwise(F.lit("None")),
            )
        return df

    def run(self, is_save):
        self.read_processed_data()
        df = self.initialize_dataframe()
        df = self.preprocess_feature(df)
        return df


class ProfileFeaturePreprocessing(BaseFeaturePreprocessing):
    """
    Preprocessing logic for profile-related feature data.

    Attributes:
        process_lib (str): Library used for processing ('pandas' or 'pyspark').
        raw_data_path (str): Path to the raw data directory.
    """

    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
    ):
        super().__init__(process_lib, raw_data_path)

    def read_processed_data(self):
        data_path = self.raw_data_dir / f"{DataName.PROFILE_MYTV_INFO}.parquet"
        df = load_parquet_data(
            file_paths=data_path,
            process_lib=self.process_lib,
            spark=self.spark,
        )
        self.raw_data[DataName.PROFILE_MYTV_INFO] = df

    def initialize_dataframe(self):
        df = self._create_data_for_new_user(self.raw_data[DataName.PROFILE_MYTV_INFO])
        df = self.create_user_key(df)
        if self.process_lib in ["pandas"]:
            df = df.drop_duplicates()
        else:
            df = df.dropDuplicates()
        return df

    def _create_data_for_new_user(self, df):
        if self.process_lib == "pandas":
            new_values = {
                "profile_id": [0],
                "username": ["empty"],
            }
            new_df = pd.DataFrame(data=new_values)
            df = pd.concat([new_df, df])
        else:
            columns = ["profile_id", "username"]
            vals = [(0, "empty")]
            new_df = self.spark.createDataFrame(vals, columns)
            df = new_df.union(df.select(columns))
        return df

    def preprocess_feature(self, df):
        spare_feature_info = conf.DhashSpareFeatureInfo()
        df = self.preprocess_hashed_id(
            df,
            output_feature_names=["hashed_user_id"],
            hash_dependency_info={"hashed_user_id": "user_id"},
            spare_feature_info=spare_feature_info,
        )
        df = self.preprocess_hashed_id(
            df,
            output_feature_names=["hashed_user_id_v2"],
            hash_dependency_info={"hashed_user_id_v2": "user_id"},
            spare_feature_info=spare_feature_info,
            version=2,
        )
        return df

    def run(self, is_save=True):
        self.read_processed_data()
        df = self.initialize_dataframe()
        df = self.preprocess_feature(df)
        return df


class UserFeaturePreprocessing(BaseFeaturePreprocessing):
    """
    Combines and preprocesses account and profile feature data.

    Attributes:
        process_lib (str): Library used for processing ('pandas' or 'pyspark').
        raw_data_path (str): Path to the raw data directory.
        save_filename (str): Filename for saving preprocessed user data.
    """

    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename=DataName.USER_INFO,
        spark_config=None,
    ):
        super().__init__(process_lib, raw_data_path, save_filename, spark_config)

    def read_processed_data(self):
        account_df = AccountFeaturePreprocessing(
            self.process_lib, self.raw_data_dir
        ).run(is_save=True)
        profile_df = ProfileFeaturePreprocessing(
            self.process_lib, self.raw_data_dir
        ).run()
        self.raw_data[DataName.ACCOUNT_MYTV_INFO] = account_df
        self.raw_data[DataName.PROFILE_MYTV_INFO] = profile_df

    def initialize_dataframe(self):
        if self.process_lib == "pandas":
            self.raw_data[DataName.ACCOUNT_MYTV_INFO] = self.raw_data[
                DataName.ACCOUNT_MYTV_INFO
            ].drop(columns=["profile_id"])
            df = self.raw_data[DataName.PROFILE_MYTV_INFO].merge(
                self.raw_data[DataName.ACCOUNT_MYTV_INFO], on="username", how="left"
            )
        else:
            self.raw_data[DataName.ACCOUNT_MYTV_INFO] = self.raw_data[
                DataName.ACCOUNT_MYTV_INFO
            ].drop(F.col("profile_id"))
            df = self.raw_data[DataName.PROFILE_MYTV_INFO].join(
                self.raw_data[DataName.ACCOUNT_MYTV_INFO], on="username", how="left"
            )
        return df


class ABUserFeaturePreprocessing(BaseFeaturePreprocessing):
    """
    Preprocesses user data for A/B testing.

    Attributes:
        process_lib (str): Library used for processing ('pandas' or 'pyspark').
        raw_data_path (str): Path to the raw data directory.
        save_filename (str): Filename for storing A/B testing user data.
    """

    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename=DataName.AB_TESTING_USER_INFO,
        spark_config=None,
    ):
        super().__init__(process_lib, raw_data_path, save_filename, spark_config)

    def read_processed_data(self):
        data_path = self.raw_data_dir / f"{DataName.AB_TESTING_USER_INFO}.parquet"
        df = load_parquet_data(
            file_paths=data_path,
            process_lib=self.process_lib,
            spark=self.spark,
        )
        self.raw_data[DataName.AB_TESTING_USER_INFO] = df

    def initialize_dataframe(self):
        df = self.create_user_key(self.raw_data[DataName.AB_TESTING_USER_INFO])
        return df
