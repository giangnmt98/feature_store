"""
Module: online_feature_preprocessing

This module focuses on preprocessing *online* features for both item-related
and user-related data. It provides logic to dynamically extract, transform,
and label features in real-time or near-real-time environments.
"""

import pyspark.sql.functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel

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
        big_df = (
            df.filter(F.col("content_type") != 31)
            .select("profile_id", "item_id", "filename_date")
            .dropDuplicates()
        )

        # Tạo dataframe các khoảng thời gian tương ứng để xử lý
        date_ranges = [
            (
                p_date,
                get_date_before(
                    p_date, num_days_before=conf.ROLLING_PERIOD_FOR_POPULARITY_ITEM
                ),
            )
            for p_date in self.dates_to_extract
        ]
        date_df = self.spark.createDataFrame(date_ranges, ["end_date", "start_date"])

        # Thực hiện join dữ liệu để xử lý tất cả ngày một cách đồng thời
        big_df = big_df.join(
            date_df,
            (big_df.filename_date <= F.col("end_date"))
            & (big_df.filename_date > F.col("start_date")),
            "inner",
        )

        # Nhóm và đếm số lượng item theo khoảng ngày
        count_df = big_df.groupBy("item_id", "end_date").count()

        # Xử lý rank dựa trên số lượng và xác định Group
        window_spec = Window.partitionBy("end_date").orderBy(F.col("count").desc())
        # "others": "0"
        # "100": "1"
        # "101-300": "2"
        # "301-1000": "3"
        # "1001-2000": "4"
        # ">2000": "5"
        result_df = (
            count_df.withColumn("row", F.row_number().over(window_spec))
            .withColumn(
                "popularity_item_group",
                F.when(F.col("row") <= 100, F.lit(1))
                .when(F.col("row") <= 300, F.lit(2))
                .when(F.col("row") <= 1000, F.lit(3))
                .when(F.col("row") <= 2000, F.lit(4))
                .otherwise(F.lit(5)),
            )
            .drop("row")
        )

        # Trường hợp ít hơn 15 ngày: gán dữ liệu mặc định
        filename_date_count = big_df.groupBy("end_date").agg(
            F.countDistinct("filename_date").alias("day_count")
        )

        result_df = result_df.join(filename_date_count, ["end_date"], "left")
        result_df = result_df.withColumn(
            "popularity_item_group",
            F.when(F.col("day_count") < 15, F.lit(0)).otherwise(
                F.col("popularity_item_group")
            ),
        )
        # Thêm các cột ngày tháng
        popular_item_group = result_df.withColumn(
            "date_time", F.to_date(F.col("end_date").cast(StringType()), "yyyyMMdd")
        ).drop("day_count")
        popular_item_group = popular_item_group.withColumnRenamed(
            "end_date", "filename_date"
        )
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
        """
         Processes user-related data using PySpark to compute feature preferences.

        Args:
            df: Input PySpark DataFrame containing user
                interaction data (profile_id, content_type, filename_date).

        Returns:
            pyspark.sql.DataFrame: A PySpark DataFrame containing preprocessed user
                preferences.
        """

        # 1. Tối ưu cast operations bằng cách gom nhóm
        big_df = (
            df.select(
                "*",
                F.col("content_id").cast("int").alias("content_id_int"),
                F.col("content_type").cast("int").alias("content_type_int"),
                F.col("profile_id").cast("int").alias("profile_id_int"),
            )
            .drop("content_id", "content_type", "profile_id")
            .withColumnRenamed("content_id_int", "content_id")
            .withColumnRenamed("content_type_int", "content_type")
            .withColumnRenamed("profile_id_int", "profile_id")
        )

        # 2. Tối ưu filter và persist với partition
        filtered_df = (
            big_df.filter(F.col("content_type") != 31)
            .dropDuplicates(["profile_id", "content_type", "filename_date"])
            .repartition("profile_id")
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        # 3. Broadcast nhỏ gọn movie_types và cache
        movie_types_df = self.spark.createDataFrame(
            [(x,) for x in conf.MOVIE_TYPE_GROUP], ["content_type"]
        ).persist(StorageLevel.MEMORY_AND_DISK)

        movie_types = F.broadcast(movie_types_df)

        # 4. Tối ưu xác định movie/vod bằng join thay vì UDF
        typed_df = (
            filtered_df.join(movie_types, "content_type", "left")
            .withColumn(
                "movie_or_vod",
                F.when(F.col("content_type").isNotNull(), "movie").otherwise("vod"),
            )
            .select("profile_id", "content_type", "filename_date", "movie_or_vod")
        )

        # 5. Tối ưu date ranges và cache
        date_ranges_df = self.spark.createDataFrame(
            [
                (d, get_date_before(d, conf.ROLLING_PERIOD_FOR_USER_PREFER_TYPE))
                for d in self.dates_to_extract
            ],
            ["end_date", "begin_date"],
        ).persist(StorageLevel.MEMORY_AND_DISK)

        # 6. Sử dụng window function và tối ưu join

        user_prefer_type = (
            typed_df.join(
                F.broadcast(date_ranges_df),
                (typed_df.filename_date <= F.col("end_date"))
                & (typed_df.filename_date > F.col("begin_date")),
            )
            .groupBy("profile_id", "movie_or_vod", "end_date")
            .agg(F.count("*").alias("prefer_count"))
            .groupBy("profile_id", "end_date")
            .pivot("movie_or_vod", ["movie", "vod"])
            .agg(F.first("prefer_count"))
            .na.fill(0)
            .withColumnRenamed("movie", "prefer_movie_type")
            .withColumnRenamed("vod", "prefer_vod_type")
        )

        # 7. Tối ưu chuyển đổi datetime
        user_prefer_type = user_prefer_type.withColumn(
            "date_time", F.to_date(F.col("end_date"), "yyyyMMdd")
        ).withColumnRenamed("end_date", "filename_date")

        # 8. Cleanup resources
        filtered_df.unpersist()
        movie_types_df.unpersist()
        date_ranges_df.unpersist()
        return user_prefer_type
