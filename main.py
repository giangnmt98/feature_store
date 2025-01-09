import os
import argparse
import shutil

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
    try:
        feathr_jar_path = config.get("feathr_jar_path")
        if feathr_jar_path:
            current_directory = os.getcwd()
            destination_file = os.path.join(current_directory, os.path.basename(feathr_jar_path))

            # Kiểm tra nếu file chưa tồn tại thì thực hiện copy
            if not os.path.exists(destination_file):
                shutil.copy(feathr_jar_path, destination_file)
                print(f"File copied from {feathr_jar_path} to {destination_file}")
            else:
                print(f"File already exists at {destination_file}, skipping copy.")
        else:
            print("Config 'feathr_jar_path' is not defined in the configuration.")
    except Exception as e:
        print(f"Error copying file: {e}")

    FeaturePipeline(
        raw_data_path=config["raw_data_path"],
        feathr_workspace_folder=config["feathr_workspace_folder"],
        feature_registry_config_path=config["feature_registry_config_path"],
        training_pipeline_config_path=config["training_pipeline_config_path"],
        materialize_pipeline_config_path=config["materialize_pipeline_config_path"],
        infer_pipeline_config_path=config["infer_pipeline_config_path"],
        process_lib=config["process_lib"],
    ).run_all()
    print("\n")
    print("="*50)
    print("Kill spark process")
    os.system("pkill -f spark")
