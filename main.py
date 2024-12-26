import argparse

import yaml

from featurestore.feature_pipeline import FeaturePipeline

if __name__ == "__main__":

    def load_simple_dict_config(path_config):
        with open(path_config) as f:
            config = yaml.safe_load(f)
        return config

    parser = argparse.ArgumentParser(
        description="Process CSV data based on a given configuration."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        nargs="?",
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    config = load_simple_dict_config(args.config_path)
    # main(args.config_path)

    FeaturePipeline(
        raw_data_path=config["raw_data_path"],
        feathr_workspace_folder=config["feathr_workspace_folder"],
        feature_registry_config_path=config["feature_registry_config_path"],
        training_pipeline_config_path=config["training_pipeline_config_path"],
        materialize_pipeline_config_path=config["materialize_pipeline_config_path"],
        infer_pipeline_config_path=config["infer_pipeline_config_path"],
        process_lib=config["process_lib"],
    ).run_all()
