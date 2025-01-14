import argparse
import os

from featurestore.base.utils.utils import load_simple_dict_config
from featurestore.feature_pipeline import FeaturePipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CSV data based on a given configuration."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        nargs="?",
        default="config.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    config = load_simple_dict_config(args.config_path)
    # main(args.config_path)
    try:
        FeaturePipeline(
            raw_data_path=config["raw_data_path"],
            infer_date=config["infer_date"],
            feathr_workspace_folder=config["feathr_workspace_folder"],
            feature_registry_config_path=config["feature_registry_config_path"],
            training_pipeline_config_path=config["training_pipeline_config_path"],
            materialize_pipeline_config_path=config["materialize_pipeline_config_path"],
            infer_pipeline_config_path=config["infer_pipeline_config_path"],
            process_lib=config["process_lib"],
        ).run_all()
    finally:
        print("\n")
        print("=" * 50)
        # Terminate any running Spark-related processes to clean up resources after execution.
        print("Kill spark process")
        os.system("pkill -f spark")
