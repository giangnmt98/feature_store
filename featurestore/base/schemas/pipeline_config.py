from typing import Dict, List, Optional

from pydantic import BaseModel


class FeatureTableConfig(BaseModel):
    feature_table_names: List[str]
    setting_name: str = ""
    key_names: List[str]
    feature_names: List[str]


class FeatureQueryConfig(BaseModel):
    feature_names: List[str]
    key_names: List[str]


class TrainingPipelineConfig(BaseModel):
    raw_data_path: str
    feature_queries: List[FeatureQueryConfig]
    spark_execution_config: Dict


class MaterializePipelineConfig(BaseModel):
    feature_tables: List[FeatureTableConfig]
    online_spark_execution_configs: Optional[Dict] = None
    offline_spark_execution_configs: Optional[Dict] = None
    infer_date: str = "today"
    save_dir_path: str = ""


class InferPipelineConfig(BaseModel):
    feature_tables: List[FeatureTableConfig]
