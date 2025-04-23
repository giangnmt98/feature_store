"""
Module: interaction_feature_preprocessing

This module focuses on preprocessing the interaction data used for feature engineering
in recommendation systems or predictive modeling. The preprocessing includes tasks to
combine movie and video-on-demand (VOD) interaction data, create user and item keys,
and perform transformations like negative sampling.
"""
from typing import Union

import pandas as pd
import pyspark.sql.functions as F
from pyspark import StorageLevel
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StringType
from tqdm import tqdm

from configs import conf
from featurestore.base.feature_preprocessing import BaseDailyFeaturePreprocessing
from featurestore.base.utils.fileops import load_parquet_data
from featurestore.base.utils.logger import logger
from featurestore.base.utils.utils import split_batches
from featurestore.constants import DataName


class InteractedFeaturePreprocessing(BaseDailyFeaturePreprocessing):
    """
    Preprocesses interaction data for features.

    This includes combining movie and VOD interaction data, performing negative sampling

    Attributes:
        process_lib (str): Library used for processing ('pandas' or 'pyspark').
        raw_data_path (str): Path to the raw data directory.
        save_filename (str): Filename for storing preprocessed interaction data.
        data_name_to_get_new_dates (str): Dataset name to retrieve new interaction data.
    """

    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename=DataName.OBSERVATION_FEATURES,
        data_name_to_get_new_dates=DataName.MOVIE_HISTORY,
        spark_config=None,
    ):
        super().__init__(
            process_lib,
            raw_data_path,
            save_filename,
            data_name_to_get_new_dates,
            spark_config,
        )

    def read_processed_data(self):
        movie_df = self._load_raw_data(
            data_name=DataName.MOVIE_HISTORY,
            with_columns=conf.SELECTED_HISTORY_COLUMNS,
            dates_to_extract=self.dates_to_extract,
        )
        vod_df = self._load_raw_data(
            data_name=DataName.VOD_HISTORY,
            with_columns=conf.SELECTED_HISTORY_COLUMNS,
            dates_to_extract=self.dates_to_extract,
        )
        content_type_df = load_parquet_data(
            file_paths=self.raw_data_dir / f"{DataName.CONTENT_TYPE}.parquet",
            process_lib=self.process_lib,
            spark=self.spark,
        )
        self.raw_data[DataName.MOVIE_HISTORY] = movie_df
        self.raw_data[DataName.VOD_HISTORY] = vod_df
        self.raw_data[DataName.CONTENT_TYPE] = content_type_df

    def initialize_dataframe(self):
        movie_df = self.raw_data[DataName.MOVIE_HISTORY].withColumn(
            "is_vod_content", F.lit(False)
        )
        vod_df = self.raw_data[DataName.VOD_HISTORY].withColumn(
            "is_vod_content", F.lit(True)
        )
        vod_df = vod_df.withColumn(
            "is_vod_content",
            F.when((vod_df["content_type"] == "21"), False).otherwise(
                vod_df["is_vod_content"]
            ),
        )
        big_df = movie_df.union(vod_df)

        big_df = self.create_user_key(big_df)
        big_df = self.create_item_key(big_df)

        big_df = big_df.drop("username")
        big_df = big_df.withColumn("profile_id", F.col("profile_id").cast("int"))
        big_df = big_df.withColumn("content_id", F.col("content_id").cast("int"))
        big_df = big_df.withColumn("content_type", F.col("content_type").cast("int"))
        big_df = big_df.join(
            F.broadcast(self.raw_data[DataName.CONTENT_TYPE].select("content_type")),
            on="content_type",
            how="inner",
        )
        big_df = big_df.filter(F.col("profile_id") != 0)
        big_df = big_df.groupBy(
            "user_id",
            "item_id",
            "profile_id",
            "content_id",
            "content_type",
            "filename_date",
        ).agg(
            F.sum("duration").alias("duration"),
            F.max("is_vod_content").alias("is_vod_content"),
        )
        big_df = big_df.withColumn(
            "date_time",
            F.to_date(F.col("filename_date").cast(StringType()), "yyyyMMdd"),
        )
        # negative sampling
        logger.info("Negative sampling.")
        big_df = self._negative_sample(big_df)
        logger.info("Negative sampling...done!")
        return big_df

    def _negative_sample(self, big_df):
        negative_sample_ratio = 12
        mean_samples_per_day = (
            big_df.groupby(["user_id", "profile_id", "filename_date"])
            .agg(F.count("item_id").alias("count"))
            .agg(F.mean("count").alias("mean"))
            .select("mean")
            .first()[0]
        )
        negative_samples_per_day = int(mean_samples_per_day * negative_sample_ratio)
        item_df = big_df.select(
            "item_id",
            "content_id",
            "content_type",
            "filename_date",
            "is_vod_content",
        ).dropDuplicates()
        user_df = big_df.select(
            "user_id", "profile_id", "filename_date"
        ).dropDuplicates()
        neg_interact_df = self._negative_sample_each_day(
            user_df, item_df, negative_samples_per_day, big_df
        )
        neg_interact_df = neg_interact_df.select(big_df.columns)
        result_df = big_df.union(neg_interact_df)
        result_df = result_df.withColumn(
            "filename_date", F.col("filename_date").cast("int")
        )
        return result_df

    def _negative_sample_each_day(
        self,
        user_df: DataFrame,
        item_df: DataFrame,
        num_negative_samples: int,
        big_df: DataFrame,
    ) -> DataFrame:
        neg_interact_df = user_df.join(
            F.broadcast(item_df), on="filename_date", how="inner"
        )
        # global sampling to reduce sampling pool
        # before perform stratified sampling
        mean_possible_samples_per_day = (
            big_df.groupby(["filename_date"])
            .count()
            .agg(F.mean("count").alias("mean"))
            .select("mean")
            .first()[0]
        )
        # 1000 times is big enough to maintain result of stratified sampling
        reduced_pool_size = 1000 * num_negative_samples
        sampling_fraction = reduced_pool_size / mean_possible_samples_per_day
        if sampling_fraction < 1:
            neg_interact_df = neg_interact_df.sample(
                fraction=sampling_fraction, seed=40
            )
        # draw (num_negative_samples) sample
        # for each user-date pair from smaller pool
        neg_interact_df = neg_interact_df.withColumn(
            "random_group", F.floor(F.rand(seed=42) * num_negative_samples)
        ).withColumn("random_selection", F.rand(seed=41))
        neg_interact_df = (
            neg_interact_df.groupby(
                [
                    "user_id",
                    "profile_id",
                    "filename_date",
                    "is_vod_content",
                    "random_group",
                ]
            ).agg(F.max_by("item_id", "random_selection").alias("item_id"))
        ).drop("random_group", "random_selection")
        neg_interact_df = neg_interact_df.withColumn(
            "content_type",
            F.concat_ws("_", F.lit("type"), F.split(F.col("item_id"), "_", 2)[0]),
        ).withColumn("content_id", F.split(F.col("item_id"), "_", 2)[1])
        neg_interact_df = neg_interact_df.withColumn(
            "content_id", F.col("content_id").cast("int")
        )
        neg_interact_df = neg_interact_df.withColumn(
            "content_type", F.col("content_type").cast("int")
        )
        neg_interact_df = neg_interact_df.withColumn(
            "date_time",
            F.to_date(F.col("filename_date").cast("string"), "yyyyMMdd"),
        ).withColumn("duration", F.lit(0))
        neg_interact_df = neg_interact_df.withColumn(
            "filename_date", F.col("filename_date").cast("int")
        )
        return neg_interact_df

    def preprocess_feature(self, df):
        df = df.withColumn("is_interacted", F.lit(2))
        df = df.withColumn(
            "is_interacted",
            F.when(
                (
                    (F.col("duration") < conf.VOD_DIRTY_CLICK_DURATION_THRESHOLD)
                    & F.col("is_vod_content")
                )
                | (
                    (F.col("duration") < conf.MOVIE_DIRTY_CLICK_DURATION_THRESHOLD)
                    & ~F.col("is_vod_content")
                ),
                F.lit(0),
            ).otherwise(F.col("is_interacted")),
        )
        df = df.withColumn(
            "is_interacted",
            F.when(F.col("duration") == 0, F.lit(1)).otherwise(F.col("is_interacted")),
        )
        return df
