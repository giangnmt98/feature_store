from pydantic import BaseModel


class FeatureRegistryConfig(BaseModel):
    raw_data_path: str
