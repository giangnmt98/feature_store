import yaml

from featurestore.base.schemas import pipeline_config, registry_config


def parse_training_config(yaml_path: str) -> pipeline_config.TrainingPipelineConfig:
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
        return pipeline_config.TrainingPipelineConfig(**config)


def parse_materialize_config(
    yaml_path: str,
) -> pipeline_config.MaterializePipelineConfig:
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
        return pipeline_config.MaterializePipelineConfig(**config)


def parse_infer_config(yaml_path: str) -> pipeline_config.InferPipelineConfig:
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
        return pipeline_config.InferPipelineConfig(**config)


def parse_registry_config(yaml_path: str) -> registry_config.FeatureRegistryConfig:
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
        return registry_config.FeatureRegistryConfig(**config)
