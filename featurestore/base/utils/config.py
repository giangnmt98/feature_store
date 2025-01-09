"""
Module: config

This module provides utilities for parsing and deserializing YAML configuration files
into structured objects for various pipelines and feature registry configurations.
It ensures that configuration data is read and transformed into strongly-typed objects,
facilitating consistency and type safety throughout the application.
"""
import yaml

from featurestore.base.schemas import pipeline_config, registry_config


def parse_training_config(yaml_path: str) -> pipeline_config.TrainingPipelineConfig:
    """
    Parses the training pipeline configuration from a YAML file.

    This method reads a YAML file containing the configuration for a training pipeline,
    deserializes it, and returns it as a `TrainingPipelineConfig` object.

    Args:
        yaml_path (str): The file path to the YAML configuration file.

    Returns:
        pipeline_config.TrainingPipelineConfig: The parsed training pipeline config.
    """
    with open(yaml_path, encoding="utf8") as f:
        config = yaml.safe_load(f)
        return pipeline_config.TrainingPipelineConfig(**config)


def parse_materialize_config(
    yaml_path: str,
) -> pipeline_config.MaterializePipelineConfig:
    """
    Parses the materialization pipeline configuration from a YAML file.

    This method reads a YAML file containing the configuration for materializing
    features into offline or online storage, deserializes it, and returns it as a
    `MaterializePipelineConfig` object.

    Args:
        yaml_path (str): The file path to the YAML configuration file.

    Returns:
        pipeline_config.MaterializePipelineConfig: The parsed materialization pipeline
        configuration.
    """
    with open(yaml_path, encoding="utf8") as f:
        config = yaml.safe_load(f)
        return pipeline_config.MaterializePipelineConfig(**config)


def parse_infer_config(yaml_path: str) -> pipeline_config.InferPipelineConfig:
    """
    Parses the inference pipeline configuration from a YAML file.

    This method reads a YAML file containing the configuration for an inference
    pipeline, deserializes it, and returns it as an `InferPipelineConfig` object.

    Args:
        yaml_path (str): The file path to the YAML configuration file.

    Returns:
        pipeline_config.InferPipelineConfig: The parsed inference pipeline config.
    """
    with open(yaml_path, encoding="utf8") as f:
        config = yaml.safe_load(f)
        return pipeline_config.InferPipelineConfig(**config)


def parse_registry_config(yaml_path: str) -> registry_config.FeatureRegistryConfig:
    """
    Parses the feature registry configuration from a YAML file.

    This method reads a YAML file containing the configuration for a feature registry,
    deserializes it, and returns it as a `FeatureRegistryConfig` object.

    Args:
        yaml_path (str): The file path to the YAML configuration file.

    Returns:
        registry_config.FeatureRegistryConfig: The parsed feature registry config.
    """
    with open(yaml_path, encoding="utf8") as f:
        config = yaml.safe_load(f)
        return registry_config.FeatureRegistryConfig(**config)
