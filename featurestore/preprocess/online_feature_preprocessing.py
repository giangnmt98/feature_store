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
            big_df = (
                df.filter(F.col("content_type").cast(StringType()) != "31")
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
            date_df = self.spark.createDataFrame(
                date_ranges, ["end_date", "start_date"]
            )

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
        # Ép kiểu dữ liệu content_type trước khi filter để tránh ép kiểu nhiều lần
        big_df = big_df.withColumn(
            "content_type_str", F.col("content_type").cast("string")
        )

        # Cache DataFrame sau khi lọc và distinct để tái sử dụng
        filtered_df = big_df.filter(F.col("content_type_str") != "31").distinct()

        # Chuyển đổi các giá trị MOVIE_TYPE_GROUP thành broadcast set để tối ưu join
        movie_types = self.spark.sparkContext.broadcast(set(conf.MOVIE_TYPE_GROUP))

        # Sử dụng UDF để xác định movie_or_vod, giảm chi phí kiểm tra isin()
        def determine_type(content_type):
            if content_type in movie_types.value:
                return "movie"
            return "vod"

        determine_type_udf = F.udf(determine_type, StringType())

        # Áp dụng UDF và chỉ chọn các cột cần thiết để giảm kích thước dữ liệu
        typed_df = filtered_df.withColumn(
            "movie_or_vod", determine_type_udf("content_type")
        ).select("user_id", "content_type", "filename_date", "movie_or_vod")

        # Tính toán date_ranges một lần và broadcast
        date_ranges = [
            (
                p_date,
                get_date_before(
                    p_date, num_days_before=conf.ROLLING_PERIOD_FOR_USER_PREFER_TYPE
                ),
            )
            for p_date in self.dates_to_extract
        ]

        # Tạo DataFrame cho date_ranges
        date_df = self.spark.createDataFrame(date_ranges, ["end_date", "begin_date"])

        # Sử dụng broadcast join thay vì crossJoin để cải thiện hiệu suất
        # Điều này khả thi vì date_df thường có kích thước nhỏ
        user_prefer_type = typed_df.join(
            F.broadcast(date_df),
            (typed_df.filename_date <= F.col("end_date"))
            & (typed_df.filename_date > F.col("begin_date")),
        )

        # Sử dụng cache cho dữ liệu trung gian trước khi thực hiện các phép tính toán phức tạp
        user_prefer_type = user_prefer_type.cache()

        # Nhóm và tổng hợp dữ liệu
        user_prefer_type = user_prefer_type.groupBy(
            "user_id", "movie_or_vod", "end_date"
        ).agg(F.count("content_type").alias("prefer_count"))

        # Sử dụng pivot để tạo các cột movie và vod
        user_prefer_type = (
            user_prefer_type.groupBy("user_id", "end_date")
            .pivot("movie_or_vod", ["movie", "vod"])
            .agg(F.first("prefer_count"))
            .fillna(0)
            .withColumnRenamed("movie", "prefer_movie_type")
            .withColumnRenamed("vod", "prefer_vod_type")
        )

        # Chuyển đổi kiểu dữ liệu cho cột date_time
        user_prefer_type = user_prefer_type.withColumn(
            "date_time", F.to_date(F.col("end_date"), "yyyyMMdd")
        )

        # Đổi tên cột cuối cùng
        user_prefer_type = user_prefer_type.withColumnRenamed(
            "end_date", "filename_date"
        )

        # Giải phóng bộ nhớ
        user_prefer_type.unpersist()

        return user_prefer_type
