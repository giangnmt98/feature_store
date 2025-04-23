"""Module xử lý tính năng tương tác từ dữ liệu phim và VOD.

Module này cung cấp lớp InteractedFeaturePreprocessing để xử lý và chuẩn bị dữ liệu
tương tác của người dùng với nội dung phim và VOD. Lớp này kế thừa từ
BaseDailyFeaturePreprocessing và thêm các tính năng
 xử lý chuyên biệt cho dữ liệu tương tác.
"""

import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path
from typing import Union

import pyspark.sql.functions as F
from pyspark import StorageLevel
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StringType

from configs import conf
from featurestore.base.feature_preprocessing import BaseDailyFeaturePreprocessing
from featurestore.base.utils.fileops import load_parquet_data
from featurestore.base.utils.logger import logger
from featurestore.constants import DataName


class InteractedFeaturePreprocessing(BaseDailyFeaturePreprocessing):
    """Xử lý và chuẩn bị dữ liệu tương tác người dùng với nội dung phim và VOD.

    Lớp này mở rộng BaseDailyFeaturePreprocessing
    để xử lý dữ liệu lịch sử xem phim và VOD,
    thực hiện việc tạo mẫu âm (negative sampling)
    và tiền xử lý các tính năng tương tác.

    Attributes:
        save_filename (str): Tên file để lưu dữ liệu đã xử lý.
        chunk_size (int): Số lượng partition được xử lý trong mỗi chunk.
        parallel_chunks (bool): Cho phép xử lý song song các chunk hay không.
    """

    def __init__(
        self,
        process_lib="pandas",
        raw_data_path="data/processed/",
        save_filename=DataName.OBSERVATION_FEATURES,
        data_name_to_get_new_dates=DataName.MOVIE_HISTORY,
        spark_config=None,
        chunk_size=10,
        parallel_chunks=True,
    ):
        """Khởi tạo đối tượng InteractedFeaturePreprocessing.

        Args:
            process_lib (str): Thư viện xử lý dữ liệu, mặc định là "pandas".
            raw_data_path (str): Đường dẫn đến dữ liệu gốc đã qua xử lý.
            save_filename (str): Tên file để lưu dữ liệu đã xử lý.
            data_name_to_get_new_dates (str): Tên dữ liệu dùng để lấy ngày mới.
            spark_config (dict, optional): Cấu hình Spark.
            chunk_size (int): Số lượng partition được xử lý trong mỗi chunk.
            parallel_chunks (bool): Cho phép xử lý song song các chunk hay không.
        """
        super().__init__(
            process_lib,
            raw_data_path,
            save_filename,
            data_name_to_get_new_dates,
            spark_config,
        )
        # Đảm bảo tên file lưu trữ có đuôi .parquet
        self.save_filename = (
            self.save_filename + ".parquet"
            if not self.save_filename.endswith(".parquet")
            else self.save_filename
        )
        self.chunk_size = max(1, chunk_size)  # Đảm bảo chunk_size ít nhất là 1
        self.parallel_chunks = parallel_chunks
        self.content_type_df = None
        self.partitions = []

    def get_partitions(self, data_path: Union[str, Path]) -> list:
        """Lấy danh sách các partition từ đường dẫn dữ liệu.

        Args:
            data_path (Union[str, Path]): Đường dẫn đến thư mục dữ liệu.

        Returns:
            list: Danh sách các partition đã sắp xếp theo thứ tự.
        """
        data_path = (
            Path(data_path).with_suffix(".parquet")
            if not str(data_path).endswith(".parquet")
            else Path(data_path)
        )
        try:
            # Tìm tất cả thư mục partition với mẫu filename_date=YYYYMMDD
            partition_dirs = glob(str(data_path / "filename_date=*"))
            partitions = [
                re.search(r"filename_date=(\d{8})", d).group(1)
                for d in partition_dirs
                if re.search(r"filename_date=\d{8}", d)
            ]
            # Lọc ra các partition hợp lệ (8 chữ số)
            valid_partitions = [p for p in partitions if re.match(r"^\d{8}$", p)]
            return sorted(valid_partitions)
        except Exception as e:
            logger.error(f"Failed to list partitions at {data_path}: {str(e)}")
            return []

    def read_processed_data(self):
        """Đọc dữ liệu đã xử lý và xác định các partition chung.

        Đọc dữ liệu loại nội dung và tìm các partition chung giữa dữ liệu
        lịch sử xem phim và lịch sử VOD.

        Raises:
            ValueError: Nếu không tìm thấy partition chung nào.
        """
        # Đọc dữ liệu loại nội dung và lưu vào bộ nhớ
        self.content_type_df = load_parquet_data(
            file_paths=self.raw_data_dir / f"{DataName.CONTENT_TYPE}.parquet",
            process_lib=self.process_lib,
            spark=self.spark,
        ).persist(StorageLevel.MEMORY_ONLY)

        # Xác định đường dẫn dữ liệu lịch sử
        movie_history_path = self.raw_data_dir / DataName.MOVIE_HISTORY
        vod_history_path = self.raw_data_dir / DataName.VOD_HISTORY

        # Lấy danh sách partition từ cả hai nguồn dữ liệu
        movie_partitions = set(self.get_partitions(movie_history_path))
        vod_partitions = set(self.get_partitions(vod_history_path))
        # Tìm các partition chung giữa hai nguồn
        self.partitions = sorted(movie_partitions.intersection(vod_partitions))

        if not self.partitions:
            raise ValueError(
                "No common partitions found between MOVIE_HISTORY and VOD_HISTORY"
            )

        logger.info(f"Found partitions: {self.partitions}")

    def process_chunk(self, chunk_partitions, chunk_idx, temp_dir):
        """Xử lý một nhóm partition dữ liệu.

        Args:
            chunk_partitions (list): Danh sách các partition cần xử lý.
            chunk_idx (int): Chỉ số của chunk hiện tại.
            temp_dir (str): Thư mục tạm thời để lưu kết quả của chunk.

        Returns:
            None
        """
        movie_history_parquet = DataName.MOVIE_HISTORY + ".parquet"
        vod_history_parquet = DataName.VOD_HISTORY + ".parquet"

        logger.info(
            f"Processing chunk {chunk_idx + 1} with partitions: {chunk_partitions}"
        )

        try:
            # Định nghĩa đường dẫn gốc
            movie_base_path = str(self.raw_data_dir / movie_history_parquet)
            vod_base_path = str(self.raw_data_dir / vod_history_parquet)

            # Hàm hỗ trợ để kiểm tra và đọc dữ liệu
            def read_partitioned_data(base_path, selected_columns):
                """Đọc dữ liệu từ các partition đã chọn.

                Args:
                    base_path (str): Đường dẫn cơ sở đến dữ liệu.
                    selected_columns (list): Danh sách các cột cần đọc.

                Returns:
                    DataFrame: DataFrame Spark chứa dữ liệu đã đọc,
                     hoặc None nếu không có dữ liệu.
                """
                # Loại bỏ filename_date khỏi selected_columns nếu có
                clean_columns = [
                    col for col in selected_columns if col != "filename_date"
                ]

                # Kiểm tra xem thư mục có cấu trúc partition không
                partition_dirs = glob(str(Path(base_path) / "filename_date=*"))
                if partition_dirs:
                    # Có partition, đọc trực tiếp với Spark
                    df = (
                        self.spark.read.option("basePath", base_path)
                        .parquet(base_path)
                        .where(F.col("filename_date").isin(chunk_partitions))
                        .select(clean_columns + ["filename_date"])
                    )
                else:
                    # Không có partition, fallback về đọc danh sách file
                    file_paths = [
                        str(Path(base_path) / f"filename_date={partition_date}")
                        for partition_date in chunk_partitions
                        if Path(base_path)
                        .joinpath(f"filename_date={partition_date}")
                        .exists()
                    ]
                    if not file_paths:
                        logger.warning(
                            f"No files found for chunk {chunk_idx + 1} at {base_path}"
                        )
                        return None
                    df = (
                        self.spark.read.parquet(*file_paths)
                        .select(clean_columns)
                        .withColumn(
                            "filename_date",
                            F.regexp_extract(
                                F.input_file_name(), r"filename_date=(\d{8})", 1
                            ),
                        )
                        .filter(F.col("filename_date").isin(chunk_partitions))
                    )
                return df

            # Đọc dữ liệu movie và vod
            movie_df = read_partitioned_data(
                movie_base_path, conf.SELECTED_HISTORY_COLUMNS
            )
            vod_df = read_partitioned_data(vod_base_path, conf.SELECTED_HISTORY_COLUMNS)

        except Exception as e:
            logger.error(f"Failed to load chunk {chunk_idx + 1}: {str(e)}")
            return

        # Thêm cột is_vod_content để đánh dấu loại nội dung
        movie_df = movie_df.withColumn("is_vod_content", F.lit(False))
        vod_df = vod_df.withColumn("is_vod_content", F.lit(True))
        # Điều chỉnh cho content_type 21 (được coi là movie, không phải VOD)
        vod_df = vod_df.withColumn(
            "is_vod_content",
            F.when((vod_df["content_type"] == "21"), False).otherwise(
                vod_df["is_vod_content"]
            ),
        )

        # Union dữ liệu movie và VOD
        big_df = movie_df.unionByName(vod_df)
        movie_df.unpersist()
        vod_df.unpersist()
        big_df.persist(StorageLevel.MEMORY_AND_DISK)

        # Cast các cột sang đúng kiểu dữ liệu
        big_df = big_df.withColumn("profile_id", F.col("profile_id").cast("int"))
        big_df = big_df.withColumn("content_id", F.col("content_id").cast("int"))
        big_df = big_df.withColumn("content_type", F.col("content_type").cast("int"))

        # Tạo user_key và item_key từ dữ liệu
        big_df = self.create_user_key(big_df)
        big_df = self.create_item_key(big_df)

        # Join với content_type_df để lọc các content_type hợp lệ
        big_df = big_df.join(
            F.broadcast(self.content_type_df.select("content_type")),
            on="content_type",
            how="inner",
        )
        # Lọc bỏ các profile_id không hợp lệ
        big_df = big_df.filter(F.col("profile_id") != 0)

        # Group by và aggregate để tổng hợp dữ liệu
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

        # Thêm cột date_time từ filename_date
        big_df = big_df.withColumn(
            "date_time",
            F.to_date(F.col("filename_date").cast(StringType()), "yyyyMMdd"),
        )

        # Thực hiện negative sampling
        logger.info(f"Negative sampling for chunk {chunk_idx + 1}")
        big_df = self._negative_sample(big_df)
        logger.info(f"Negative sampling for chunk {chunk_idx + 1}...done!")

        # Tiền xử lý dữ liệu
        logger.info(f"Preprocessing data for chunk {chunk_idx + 1}")
        big_df = self.preprocess_feature(big_df)
        logger.info(f"Preprocessing data for chunk {chunk_idx + 1}...done!")

        # Coalesce để tối ưu số lượng partition trước khi ghi
        num_partitions = max(1, self.spark.sparkContext.defaultParallelism)
        big_df = big_df.coalesce(num_partitions)

        # Ghi dữ liệu vào thư mục tạm
        chunk_temp_path = str(Path(temp_dir) / f"chunk_{chunk_idx}")
        try:
            big_df.write.mode("overwrite").partitionBy("filename_date").parquet(
                chunk_temp_path
            )
            logger.info(
                f"Chunk {chunk_idx + 1} saved to temporary path {chunk_temp_path}"
            )
        except Exception as e:
            logger.error(f"Failed to write chunk {chunk_idx + 1}: {str(e)}")
            big_df.unpersist()
            return

        big_df.unpersist()

    def initialize_dataframe(self):
        """Khởi tạo DataFrame chứa dữ liệu đã xử lý.

        Phương thức này xử lý dữ liệu theo từng chunk, lưu vào thư mục tạm,
        sau đó hợp nhất tất cả các chunk lại và lưu kết quả cuối cùng.
        """
        save_path = Path(self.save_path)
        logger.info(
            f"Will save processed data to {save_path} with partitionBy filename_date"
        )

        # Tạo thư mục đầu ra nếu chưa tồn tại
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Tạo thư mục tạm thời để lưu các chunk
        temp_dir = str(save_path.parent / f"temp_{save_path.name}")
        Path(temp_dir).mkdir(exist_ok=True)

        # Chia danh sách partition thành các chunk nhỏ hơn
        chunks = [
            self.partitions[i : i + self.chunk_size]
            for i in range(0, len(self.partitions), self.chunk_size)
        ]

        # Xử lý các chunk song song hoặc tuần tự
        if self.parallel_chunks:
            # Xác định số lượng worker tối ưu
            cpu_count = os.cpu_count() or 4
            max_workers = min(
                int(cpu_count * 0.8),  # Sử dụng 80% CPU
                len(chunks),
                8,  # Giới hạn tối đa số lượng worker
            )
            max_workers = max(1, max_workers)
            logger.info(f"Using {max_workers} workers for parallel chunk processing")

            # Xử lý song song các chunk
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.process_chunk, chunk, idx, temp_dir)
                    for idx, chunk in enumerate(chunks)
                ]
                for future in futures:
                    future.result()
        else:
            # Xử lý tuần tự từng chunk
            for idx, chunk in enumerate(chunks):
                self.process_chunk(chunk, idx, temp_dir)

        # Hợp nhất dữ liệu từ các thư mục tạm thời
        try:
            if Path(temp_dir).exists():
                combined_df = None
                # Đọc từng chunk và hợp nhất
                for idx in range(len(chunks)):
                    chunk_temp_path = str(Path(temp_dir) / f"chunk_{idx}")
                    if Path(chunk_temp_path).exists():
                        try:
                            chunk_df = self.spark.read.parquet(chunk_temp_path)
                            combined_df = (
                                chunk_df
                                if combined_df is None
                                else combined_df.union(chunk_df)
                            )
                            logger.info(
                                f"Loaded chunk {idx + 1} from {chunk_temp_path}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to load chunk {idx + 1}"
                                f" from {chunk_temp_path}: {str(e)}"
                            )

                # Ghi dữ liệu đã hợp nhất
                if combined_df is not None:
                    combined_df.write.mode("overwrite").partitionBy(
                        "filename_date"
                    ).parquet(str(save_path))
                else:
                    logger.warning("No valid data was written from chunks")
        finally:
            # Dọn dẹp thư mục tạm
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary directory {temp_dir}")

    def _negative_sample(self, big_df: DataFrame) -> DataFrame:
        """Thực hiện lấy mẫu âm (negative sampling) cho dữ liệu tương tác.

        Args:
            big_df (DataFrame): DataFrame gốc chứa dữ liệu tương tác.

        Returns:
            DataFrame: DataFrame đã bổ sung các mẫu âm.
        """
        # Lấy tỷ lệ negative sample từ config, mặc định là 12
        negative_sample_ratio = (
            conf.NEGATIVE_SAMPLE_RATIO if hasattr(conf, "NEGATIVE_SAMPLE_RATIO") else 12
        )

        # Tính số lượng mẫu trung bình mỗi ngày cho mỗi user/profile
        mean_samples_per_day = (
            big_df.groupby(["user_id", "profile_id", "filename_date"])
            .agg(F.count("item_id").alias("count"))
            .agg(F.mean("count").alias("mean"))
            .select("mean")
            .first()[0]
        )
        # Xác định số lượng negative sample cần tạo mỗi ngày
        negative_samples_per_day = int(mean_samples_per_day * negative_sample_ratio)

        # Tạo DataFrame chứa tất cả các item riêng biệt
        item_df = big_df.select(
            "item_id",
            "content_id",
            "content_type",
            "filename_date",
            "is_vod_content",
        ).dropDuplicates()

        # Tạo DataFrame chứa tất cả các user profile riêng biệt
        user_df = big_df.select(
            "user_id", "profile_id", "filename_date"
        ).dropDuplicates()

        # Tạo negative interaction DataFrame
        neg_interact_df = self._negative_sample_each_day(
            user_df, item_df, negative_samples_per_day, big_df
        )
        neg_interact_df = neg_interact_df.select(big_df.columns)

        # Kết hợp dữ liệu gốc với negative samples
        result_df = big_df.union(neg_interact_df)
        return result_df

    def _negative_sample_each_day(
        self,
        user_df: DataFrame,
        item_df: DataFrame,
        num_negative_samples: int,
        big_df: DataFrame,
    ) -> DataFrame:
        """Tạo negative samples cho từng ngày.

        Args:
            user_df (DataFrame): DataFrame chứa thông tin user.
            item_df (DataFrame): DataFrame chứa thông tin item.
            num_negative_samples (int): Số lượng negative sample cần tạo.
            big_df (DataFrame): DataFrame gốc chứa tương tác thực tế.

        Returns:
            DataFrame: DataFrame chứa các negative sample.
        """
        # Tạo tập hợp các cặp user-item tiềm năng
        neg_interact_df = user_df.join(
            item_df, on="filename_date", how="inner"
        ).persist()
        big_df.persist()

        # Tính số lượng mẫu có thể có mỗi ngày
        mean_possible_samples_per_day = (
            big_df.groupby(["filename_date"])
            .count()
            .agg(F.mean("count").alias("mean"))
            .select("mean")
            .first()[0]
        )
        # Giảm kích thước tập dữ liệu để cải thiện hiệu suất
        # 1000 times là đủ lớn để duy trì kết quả phân tầng mẫu
        reduced_pool_size = 1000 * num_negative_samples
        sampling_fraction = reduced_pool_size / mean_possible_samples_per_day
        if sampling_fraction < 1:
            neg_interact_df = neg_interact_df.sample(
                fraction=sampling_fraction, seed=40
            )

        # Thêm các cột phụ trợ để chọn mẫu ngẫu nhiên
        neg_interact_df = neg_interact_df.withColumn(
            "random_group", F.floor(F.rand(seed=42) * num_negative_samples)
        ).withColumn("random_selection", F.rand(seed=41))

        # Repartition trước khi groupby để giảm thiểu chi phí shuffle
        partition_cols = ["user_id", "filename_date"]
        neg_interact_df = neg_interact_df.repartition(*partition_cols)

        # Chọn các negative sample
        neg_interact_df = (
            neg_interact_df.groupby(
                [
                    "user_id",
                    "profile_id",
                    "filename_date",
                    "is_vod_content",
                    "random_group",
                ]
            )
            .agg(F.max_by("item_id", "random_selection").alias("item_id"))
            .drop("random_group", "random_selection")
            .withColumn("content_type", F.split(F.col("item_id"), "#", 2)[0])
            .withColumn("content_id", F.split(F.col("item_id"), "#", 2)[1])
            .withColumn(
                "date_time",
                F.to_date(F.col("filename_date").cast("string"), "yyyyMMdd"),
            )
            .withColumn("duration", F.lit(0))
        )

        # Loại bỏ các cặp user-item đã tồn tại trong dữ liệu gốc
        neg_interact_df = neg_interact_df.join(
            big_df.select("user_id", "item_id", "filename_date"),
            on=["user_id", "item_id", "filename_date"],
            how="left_anti",
        )
        return neg_interact_df.persist()

    def preprocess_feature(self, df: DataFrame) -> DataFrame:
        """Tiền xử lý các tính năng tương tác.

        Thêm cột is_interacted để phân loại mức độ tương tác dựa trên thời lượng xem.

        Args:
            df (DataFrame): DataFrame chứa dữ liệu tương tác.

        Returns:
            DataFrame: DataFrame đã được bổ sung cột is_interacted.
        """
        # Mặc định gán giá trị 2 cho is_interacted
        df = df.withColumn("is_interacted", F.lit(2))

        # Đặt is_interacted = 0 cho các tương tác "dirty click" (thời lượng xem ngắn)
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

        # Đặt is_interacted = 1 cho các tương tác có thời lượng = 0 (negative samples)
        df = df.withColumn(
            "is_interacted",
            F.when(F.col("duration") == 0, F.lit(1)).otherwise(F.col("is_interacted")),
        )
        return df
