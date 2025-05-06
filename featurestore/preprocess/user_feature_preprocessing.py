"""
Module: user_feature_preprocessing

This module contains classes and functions for preprocessing feature data for various
datasets, including user, account, profile, content, and interaction data.
"""
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, desc, lit, row_number

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
            with_columns=["username", "province", "sex", "birthday"],
        )
        self.raw_data[DataName.ACCOUNT_MYTV_INFO] = df

    def initialize_dataframe(self):
        df = self.process_user_data(self.raw_data[DataName.ACCOUNT_MYTV_INFO])
        return df

    def process_user_data(self, df) -> DataFrame:
        """Process user data with optimization"""
        try:
            # Clean and validate data
            df = (
                df.dropDuplicates()
                .filter(F.length("birthday") == 8)
                .filter(col("sex").isin([0, 1]))
            )

            # Process birthday and remove duplicates
            window_spec = Window.partitionBy("username").orderBy(
                col("birthday").cast("int").desc()
            )
            df = (
                df.withColumn("row_num", row_number().over(window_spec))
                .filter(col("row_num") == 1)
                .drop("row_num")
            )

            return df

        except Exception as e:
            print(f"User data processing error: {str(e)}")
            raise

    def run(self):
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
        df = self.process_profile_data(self.raw_data[DataName.PROFILE_MYTV_INFO])
        return df

    def process_profile_data(self, df) -> DataFrame:
        """Process profile data with error handling and optimization"""
        try:
            # Clean and filter data
            df = (
                df.filter(F.length("username") != 0)
                .filter("profile_id != 0")
                .dropDuplicates()
                .cache()
            )

            # Remove duplicates using a window function
            window_spec = Window.partitionBy(["username", "profile_id"]).orderBy(
                desc("profile_id")
            )
            df = (
                df.withColumn("row_num", row_number().over(window_spec))
                .filter("row_num = 1")
                .drop("row_num")
            )

            # Convert username to lowercase and remove duplicates
            df = df.withColumn("username", F.lower(df["username"]))
            df = df.dropDuplicates()

            # Remove profiles that have multiple records in the dataset
            # left_anti join keeps only records where profile_id appears exactly once
            df.join(
                df.groupBy("profile_id")
                .count()
                .filter("count > 1")
                .select("profile_id"),
                on="profile_id",
                how="left_anti",
            )

            return df

        except Exception as e:
            print(f"Profile data processing error: {str(e)}")
            raise

    def preprocess_feature(self, df):
        spare_feature_info = conf.DhashSpareFeatureInfo()
        df = self.preprocess_hashed_id(
            df,
            output_feature_names=["hashed_profile_id"],
            hash_dependency_info={"hashed_profile_id": "profile_id"},
            spare_feature_info=spare_feature_info,
        )
        df = self.preprocess_hashed_id(
            df,
            output_feature_names=["hashed_profile_id_v2"],
            hash_dependency_info={"hashed_profile_id_v2": "profile_id"},
            spare_feature_info=spare_feature_info,
            version=2,
        )
        return df

    def run(self):
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
        config=None,
        spark_config=None,
    ):
        super().__init__(
            process_lib, raw_data_path, save_filename, config, spark_config
        )

    def process_age_calculation(self, df: DataFrame) -> DataFrame:
        """Calculate age and age groups with optimization"""
        try:
            # Calculate age using a broadcast variable
            current_date = lit(datetime.now().date())
            df = (
                df.withColumn("birth_date", F.to_date(col("birthday"), "yyyyMMdd"))
                .dropna(subset=["birth_date"])
                .withColumn(
                    "age",
                    F.floor(F.months_between(current_date, col("birth_date")) / 12),
                )
            )
            # Apply age rules and groups
            df = (
                df.withColumn(
                    "age",
                    F.when((col("age") >= 100) | (col("age") <= 5), lit(-1)).otherwise(
                        col("age")
                    ),
                )
                .transform(self.calculate_age_groups)
                .drop("birth_date", "birthday")
            )
            df = df.filter("age != -1")
            return df

        except Exception as e:
            print(f"Age calculation error: {str(e)}")
            raise

    def calculate_age_groups(self, df: DataFrame) -> DataFrame:
        """Calculate age groups using optimized when clauses"""
        conditions = [
            (col("age") < 15, 0),  # Trẻ em
            ((col("age") >= 15) & (col("age") < 22), 1),
            ((col("age") >= 22) & (col("age") < 30), 2),
            ((col("age") >= 30) & (col("age") < 40), 3),
            ((col("age") >= 40) & (col("age") < 65), 4),
            (col("age") >= 65, 5),  # Người cao tuổi
        ]

        expr = None
        for condition, value in conditions:
            expr = (
                F.when(condition, value)
                if expr is None
                else expr.when(condition, value)
            )

        return df.withColumn("age_group", expr.otherwise(-1))

    def process_province_encoding(
        self, df: DataFrame, province_mapping: dict
    ) -> DataFrame:
        """Encode province information using broadcast variable"""
        try:
            return df.replace(province_mapping, subset=["province"]).withColumn(
                "province", col("province").cast("int")
            )

        except Exception as e:
            print(f"Province encoding error: {str(e)}")
            raise

    def read_processed_data(self):
        account_df = AccountFeaturePreprocessing(
            self.process_lib, self.raw_data_dir
        ).run()
        profile_df = ProfileFeaturePreprocessing(
            self.process_lib, self.raw_data_dir
        ).run()
        self.raw_data[DataName.ACCOUNT_MYTV_INFO] = account_df
        self.raw_data[DataName.PROFILE_MYTV_INFO] = profile_df

    def initialize_dataframe(self):
        # Join and process data
        df = (
            self.raw_data[DataName.ACCOUNT_MYTV_INFO]
            .join(self.raw_data[DataName.PROFILE_MYTV_INFO], on="username", how="left")
            .filter(col("profile_id").isNotNull())
            .transform(self.process_age_calculation)
            .cache()
        )
        df = df.drop("username, age")
        df = self.process_province_encoding(
            df, conf.SpareFeatureInfo().encoded_features["encoded_user_province"]
        )
        return df

    def preprocess_feature(self, df):
        spare_feature_info = conf.DhashSpareFeatureInfo()
        df = self.preprocess_hashed_id(
            df,
            output_feature_names=["hashed_profile_id"],
            hash_dependency_info={"hashed_profile_id": "profile_id"},
            spare_feature_info=spare_feature_info,
        )

        df = self.preprocess_hashed_id(
            df,
            output_feature_names=["hashed_profile_id_v2"],
            hash_dependency_info={"hashed_profile_id_v2": "profile_id"},
            spare_feature_info=spare_feature_info,
            version=2,
        )
        return df
