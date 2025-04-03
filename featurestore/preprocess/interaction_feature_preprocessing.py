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
        if self.process_lib in ["pandas"]:
            self.raw_data[DataName.MOVIE_HISTORY]["is_vod_content"] = False
            self.raw_data[DataName.VOD_HISTORY]["is_vod_content"] = True
            self.raw_data[DataName.VOD_HISTORY].loc[
                self.raw_data[DataName.VOD_HISTORY]["content_type"] == "21",
                "is_vod_content",
            ] = False
            big_df = pd.concat(
                [
                    self.raw_data[DataName.MOVIE_HISTORY],
                    self.raw_data[DataName.VOD_HISTORY],
                ],
                ignore_index=True,
            ).drop_duplicates()
        else:
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

        if self.process_lib == "pandas":
            big_df = big_df[
                big_df["content_type"].isin(
                    self.raw_data[DataName.CONTENT_TYPE]["content_type"]
                )
            ]
            big_df = big_df.groupby(
                [
                    "user_id",
                    "item_id",
                    "username",
                    "profile_id",
                    "content_id",
                    "content_type",
                    "filename_date",
                ],
                as_index=False,
            ).agg(
                duration=("duration", "sum"),
                is_vod_content=("is_vod_content", "max"),
            )
            big_df["date_time"] = pd.to_datetime(
                big_df["filename_date"], format=conf.FILENAME_DATE_FORMAT
            )
            # negative sampling
            logger.info("Negative sampling.")
            big_df = self._negative_sample(big_df)
        else:
            big_df = big_df.join(
                self.raw_data[DataName.CONTENT_TYPE].select("content_type"),
                on="content_type",
                how="inner",
            )
            big_df = big_df.filter(F.col("profile_id") != 0)
            big_df = big_df.groupBy(
                "user_id",
                "item_id",
                "username",
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
            big_df.persist(storageLevel=StorageLevel.MEMORY_ONLY)
            # big_df = big_df.checkpoint()
        logger.info("Negative sampling...done!")
        return big_df

    def _negative_sample(self, big_df):
        negative_sample_ratio = 12
        if self.process_lib == "pandas":
            mean_samples_per_day = (
                big_df.groupby(
                    [
                        "user_id",
                        "username",
                        "profile_id",
                        "is_vod_content",
                        "filename_date",
                    ]
                )["item_id"]
                .count()
                .mean()
            )
            negative_samples_per_day = int(mean_samples_per_day * negative_sample_ratio)
            item_df = big_df[
                [
                    "item_id",
                    "content_id",
                    "content_type",
                    "filename_date",
                    "is_vod_content",
                ]
            ].drop_duplicates()
            user_df = big_df[
                ["user_id", "username", "profile_id", "filename_date"]
            ].drop_duplicates()
            neg_interact_df = self._negative_sample_each_day(
                user_df, item_df, negative_samples_per_day, big_df
            )
            neg_interact_df = neg_interact_df[big_df.columns]
            result_df = pd.concat([big_df, neg_interact_df], ignore_index=True)
        else:
            mean_samples_per_day = (
                big_df.groupby(["user_id", "username", "profile_id", "filename_date"])
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
                "user_id", "username", "profile_id", "filename_date"
            ).dropDuplicates()
            neg_interact_df = self._negative_sample_each_day(
                user_df, item_df, negative_samples_per_day, big_df
            )
            neg_interact_df = neg_interact_df.select(big_df.columns)
            result_df = big_df.union(neg_interact_df)
        return result_df

    def _negative_sample_each_day(
        self,
        user_df: Union[pd.DataFrame, DataFrame],
        item_df: Union[pd.DataFrame, DataFrame],
        num_negative_samples: int,
        big_df: Union[pd.DataFrame, DataFrame],
    ) -> Union[pd.DataFrame, DataFrame]:
        if self.process_lib == "pandas":
            assert isinstance(user_df, pd.DataFrame)
            assert isinstance(item_df, pd.DataFrame)
            user_df = user_df.set_index("filename_date")
            item_df = item_df.set_index("filename_date")
            filename_dates = user_df.index.unique()
            date_batch_size = 5
            filename_dates_batches = split_batches(filename_dates, date_batch_size)
            neg_interact_dfs = []
            for filename_date in tqdm(filename_dates_batches):
                sub_user_df = user_df.loc[filename_date].reset_index()
                sub_item_df = item_df.loc[filename_date].reset_index()
                neg_interact_df = sub_user_df.merge(
                    sub_item_df, how="inner", on="filename_date"
                )
                neg_interact_df = neg_interact_df.groupby(
                    [
                        "user_id",
                        "username",
                        "profile_id",
                        "filename_date",
                        "is_vod_content",
                    ],
                    as_index=False,
                ).sample(n=num_negative_samples, random_state=42, replace=True)
                neg_interact_df = neg_interact_df.drop_duplicates()
                neg_interact_df["duration"] = 0
                neg_interact_df["date_time"] = pd.to_datetime(
                    neg_interact_df["filename_date"], format=conf.FILENAME_DATE_FORMAT
                )
                neg_interact_dfs.append(neg_interact_df)
            neg_interact_df = pd.concat(neg_interact_dfs, ignore_index=True)
        else:
            neg_interact_df = user_df.join(item_df, on="filename_date", how="inner")
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
                        "username",
                        "profile_id",
                        "filename_date",
                        "is_vod_content",
                        "random_group",
                    ]
                ).agg(F.max_by("item_id", "random_selection").alias("item_id"))
            ).drop("random_group", "random_selection")
            neg_interact_df = neg_interact_df.withColumn(
                "content_type", F.split(F.col("item_id"), "#", 2)[0]
            ).withColumn("content_id", F.split(F.col("item_id"), "#", 2)[1])
            neg_interact_df = neg_interact_df.withColumn(
                "date_time",
                F.to_date(F.col("filename_date").cast("string"), "yyyyMMdd"),
            ).withColumn("duration", F.lit(0))
        return neg_interact_df

    def preprocess_feature(self, df):
        if self.process_lib in ["pandas"]:
            df["is_interacted"] = 2
            df.loc[
                (df["duration"] < conf.VOD_DIRTY_CLICK_DURATION_THRESHOLD)
                & df["is_vod_content"],
                "is_interacted",
            ] = 0
            df.loc[
                (df["duration"] < conf.MOVIE_DIRTY_CLICK_DURATION_THRESHOLD)
                & ~df["is_vod_content"],
                "is_interacted",
            ] = 0
            df.loc[df["duration"] == 0, "is_interacted"] = 1
        else:
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
                F.when(F.col("duration") == 0, F.lit(1)).otherwise(
                    F.col("is_interacted")
                ),
            )
        return df
