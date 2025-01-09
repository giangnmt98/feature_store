"""
Module: registry_config

This module defines the configuration schema for managing a feature registry. It uses
Pydantic models to enforce validation and provide a strongly-typed structure for the
required configuration settings.
"""
from pydantic import BaseModel


class FeatureRegistryConfig(BaseModel):
    """
    Configuration for managing the feature registry.

    This class defines the settings required for registering features.

    Attributes:
        raw_data_path (str): The file path to the raw data.
    """

    raw_data_path: str
