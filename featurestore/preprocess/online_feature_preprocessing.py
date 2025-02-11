"""
Module: online_feature_preprocessing

This module focuses on preprocessing *online* features for both item-related
and user-related data. It provides logic to dynamically extract, transform,
and label features in real-time or near-real-time environments.
"""

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, StructType
from pyspark.sql.window import Window

from configs import conf
from featurestore.base.feature_preprocessing import BaseOnlineFeaturePreprocessing
from featurestore.constants import DataName
from featurestore.daily_data_utils import get_date_before


class OnlineItemFeaturePreprocessing(BaseOnlineFeaturePreprocessing):
    """
    Preprocesses item-related online features.

    This includes calculating item group popularity and assigning dynamic
    group tags based on interaction data.

    Attributes:
        process_lib (str): Library used for processing ('pandas' or 'pyspark').
        raw_data_path (str): Path to the raw data directory.
        save_filename (str): Filename for storing preprocessed item online features.
        data_name_to_get_new_dates (str): Dataset name to retrieve online item data.
    """

    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename=DataName.ONLINE_ITEM_FEATURES,
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

    def preprocess_feature(self, df):
        if self.process_lib in ["pandas"]:
            big_df = df[df["content_type"].astype(str) != "31"]
            big_df = big_df[
                ["profile_id", "content_id", "content_type", "filename_date"]
            ].drop_duplicates()
            big_df = big_df.reset_index(drop=True)
            big_df["item_id"] = big_df["content_type"] + "#" + big_df["content_id"]

            for i, p_date in enumerate(self.dates_to_extract):
                begin_date = get_date_before(
                    p_date, num_days_before=conf.ROLLING_PERIOD_FOR_POPULARITY_ITEM
                )
                df_small = big_df[
                    (big_df.filename_date <= p_date)
                    & (big_df.filename_date > begin_date)
                ]
                if df_small["filename_date"].drop_duplicates().count() < 15:
                    df_small = df_small["item_id"].drop_duplicates().to_frame()
                    df_small["count"] = 0
                    df_small["popularity_item_group"] = "others"
                else:
                    df_small = df_small.groupby("item_id").size().to_frame("count")
                    df_small = df_small.sort_values(by=["count"], ascending=False)
                    df_small = df_small.reset_index().reset_index()
                    df_small["popularity_item_group"] = ">2000"
                    df_small.loc[
                        df_small.index < 2000, "popularity_item_group"
                    ] = "1001-2000"
                    df_small.loc[
                        df_small.index < 1000, "popularity_item_group"
                    ] = "301-1000"
                    df_small.loc[
                        df_small.index < 300, "popularity_item_group"
                    ] = "101-300"
                    df_small.loc[df_small.index < 100, "popularity_item_group"] = "100"
                    df_small.drop(columns=["index"], inplace=True)

                df_small["filename_date"] = p_date
                df_small["date_time"] = pd.to_datetime(
                    df_small["filename_date"], format=conf.FILENAME_DATE_FORMAT
                )
                if i == 0:
                    popular_item_group = df_small
                else:
                    popular_item_group = pd.concat([popular_item_group, df_small])
            popular_item_group = popular_item_group.reset_index(drop=True)
        else:
            big_df = df.filter(F.col("content_type").cast(StringType()) != "31")
            big_df = big_df.select(
                "profile_id", "item_id", "filename_date"
            ).dropDuplicates()
            popular_item_group = self.spark.createDataFrame([], StructType([]))

            for i, p_date in enumerate(self.dates_to_extract):
                begin_date = get_date_before(
                    p_date, num_days_before=conf.ROLLING_PERIOD_FOR_POPULARITY_ITEM
                )
                df_small = big_df.filter(
                    (big_df.filename_date <= p_date)
                    & (big_df.filename_date > begin_date)
                )
                if df_small.select("filename_date").distinct().count() < 15:
                    df_small = df_small.select("item_id").distinct()
                    df_small = df_small.withColumn("count", F.lit(0))
                    df_small = df_small.withColumn(
                        "popularity_item_group", F.lit("others")
                    )
                else:
                    df_small = df_small.groupBy("item_id").count()
                    df_small = df_small.withColumn(
                        "row",
                        F.row_number().over(
                            Window.partitionBy(F.lit("1")).orderBy(
                                F.col("count").desc(),
                            )
                        ),
                    )
                    df_small = df_small.withColumn(
                        "popularity_item_group",
                        F.when(F.col("row") <= 100, F.lit("100"))
                        .when(F.col("row") <= 300, F.lit("101-300"))
                        .when(F.col("row") <= 1000, F.lit("301-1000"))
                        .when(F.col("row") <= 2000, F.lit("1001-2000"))
                        .otherwise(F.lit(">2000")),
                    )
                    df_small = df_small.drop("row")
                df_small = df_small.withColumn("filename_date", F.lit(p_date))
                df_small = df_small.withColumn(
                    "date_time",
                    F.to_date(F.col("filename_date").cast(StringType()), "yyyyMMdd"),
                )
                if i == 0:
                    popular_item_group = df_small
                else:
                    popular_item_group = popular_item_group.union(df_small)

        return popular_item_group


class OnlineUserFeaturePreprocessing(BaseOnlineFeaturePreprocessing):
    """
    Preprocesses user-related online features.

    Computes interaction trends and user preferences for specific content types.

    Attributes:
        process_lib (str): Library used for processing ('pandas' or 'pyspark').
        raw_data_path (str): Path to the raw data directory.
        save_filename (str): Filename for storing preprocessed online user features.
        data_name_to_get_new_dates (str): Dataset name to retrieve online user data.
    """

    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename=DataName.ONLINE_USER_FEATURES,
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

    def preprocess_feature(self, df):
        if self.process_lib in ["pandas"]:
            user_prefer_type = self.preprocess_feature_by_pandas(df)
        else:
            user_prefer_type = self.preprocess_feature_by_pyspark(df)
        return user_prefer_type

    def preprocess_feature_by_pandas(self, big_df):
        """
        Processes user-related data using Pandas to compute feature preferences.

        Args:
            big_df (pandas.DataFrame): Input DataFrame containing user interaction
                data.

        Returns:
            pandas.DataFrame: A pandas DataFrame
        """
        big_df = big_df[big_df["content_type"].astype(str) != "31"]
        big_df = big_df[["user_id", "content_type", "filename_date"]].drop_duplicates()
        big_df = big_df.reset_index(drop=True)
        big_df["movie_or_vod"] = "vod"
        big_df.loc[
            big_df["content_type"].isin(conf.MOVIE_TYPE_GROUP), "movie_or_vod"
        ] = "movie"

        for i, p_date in enumerate(self.dates_to_extract):
            begin_date = get_date_before(
                p_date, num_days_before=conf.ROLLING_PERIOD_FOR_USER_PREFER_TYPE
            )
            df_small = big_df[
                (big_df.filename_date <= p_date) & (big_df.filename_date > begin_date)
            ]
            if df_small["filename_date"].drop_duplicates().count() < int(
                conf.ROLLING_PERIOD_FOR_USER_PREFER_TYPE / 2
            ):
                df_small = df_small["user_id"].drop_duplicates().to_frame()
                df_small["prefer_movie_type"] = "0"
                df_small["prefer_vod_type"] = "0"
            else:
                df_small = (
                    df_small.groupby(["user_id", "movie_or_vod"])
                    .agg({"content_type": "count"})
                    .reset_index()
                )
                df_small = df_small.rename(columns={"content_type": "prefer_type"})
                df_small = df_small.pivot(
                    index="user_id", columns="movie_or_vod", values="prefer_type"
                ).reset_index()
                df_small["movie"] = df_small["movie"].fillna("0")
                df_small["vod"] = df_small["vod"].fillna("0")
                df_small["movie"] = df_small["movie"].astype(int).astype(str)
                df_small["vod"] = df_small["vod"].astype(int).astype(str)
                df_small = df_small.rename(
                    columns={"movie": "prefer_movie_type", "vod": "prefer_vod_type"}
                )
                df_small["prefer_movie_type"] = df_small["prefer_movie_type"].fillna(
                    "0"
                )
                df_small["prefer_vod_type"] = df_small["prefer_vod_type"].fillna("0")

            df_small["filename_date"] = p_date
            df_small["date_time"] = pd.to_datetime(
                df_small["filename_date"], format=conf.FILENAME_DATE_FORMAT
            )
            if i == 0:
                user_prefer_type = df_small
            else:
                user_prefer_type = pd.concat([user_prefer_type, df_small])
        user_prefer_type = user_prefer_type.reset_index(drop=True)
        return user_prefer_type

    def preprocess_feature_by_pyspark(self, big_df):
        """
        Processes user-related data using PySpark to compute feature preferences.

        Args:
            big_df (pyspark.sql.DataFrame): Input PySpark DataFrame containing user
                interaction data.

        Returns:
            pyspark.sql.DataFrame: A PySpark DataFrame
        """
        big_df = big_df.filter(F.col("content_type").cast(StringType()) != "31")
        big_df = big_df.select(
            "user_id", "content_type", "filename_date"
        ).dropDuplicates()
        big_df = big_df.withColumn(
            "movie_or_vod",
            F.when(
                F.col("content_type").isin(conf.MOVIE_TYPE_GROUP), F.lit("movie")
            ).otherwise(F.lit("vod")),
        )
        user_prefer_type = self.spark.createDataFrame([], StructType([]))

        for i, p_date in enumerate(self.dates_to_extract):
            begin_date = get_date_before(
                p_date, num_days_before=conf.ROLLING_PERIOD_FOR_USER_PREFER_TYPE
            )
            df_small = big_df.filter(
                (big_df.filename_date <= p_date) & (big_df.filename_date > begin_date)
            )
            if df_small.select("filename_date").distinct().count() < int(
                conf.ROLLING_PERIOD_FOR_USER_PREFER_TYPE / 2
            ):
                df_small = df_small.select("user_id").distinct()
                df_small = df_small.withColumn("prefer_movie_type", F.lit("0"))
                df_small = df_small.withColumn("prefer_vod_type", F.lit("0"))
            else:
                df_small = df_small.groupBy("user_id", "movie_or_vod").agg(
                    F.count("content_type").alias("prefer_type")
                )
                df_small = (
                    df_small.groupBy("user_id")
                    .pivot("movie_or_vod")
                    .agg(F.first("prefer_type"))
                )
                df_small = df_small.withColumnRenamed(
                    "movie", "prefer_movie_type"
                ).withColumnRenamed("vod", "prefer_vod_type")
                df_small = df_small.na.fill(
                    {"prefer_movie_type": "0", "prefer_vod_type": "0"}
                )
                df_small = df_small.withColumn(
                    "prefer_movie_type",
                    df_small["prefer_movie_type"].cast(StringType()),
                )
                df_small = df_small.withColumn(
                    "prefer_vod_type", df_small["prefer_vod_type"].cast(StringType())
                )
            df_small = df_small.withColumn("filename_date", F.lit(p_date))
            df_small = df_small.withColumn(
                "date_time",
                F.to_date(F.col("filename_date").cast(StringType()), "yyyyMMdd"),
            )
            if i == 0:
                user_prefer_type = df_small
            else:
                user_prefer_type = user_prefer_type.union(df_small)
        return user_prefer_type
