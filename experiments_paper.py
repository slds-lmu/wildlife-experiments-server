import os
import subprocess
import hydra
from omegaconf import DictConfig
from datetime import datetime
from pathlib import Path

# Script and configuration paths
RUN_ESTIMATION_PATH = "scripts/distance-estimation/run_estimation.py"

# Parameter grid
MAX_DEPTHS = [20, 30]  # 20, 30
CONFIDENCES = [0.1, 0.3, 0.5, 0.7, 0.9]  # 0.1, 0.3, 0.5, 0.7, 0.9
METHODS = ["BBOX_BOTTOM", "BBOX_PERCENTILE", "SAM"]  # "BBOX_BOTTOM", "BBOX_PERCENTILE", "SAM"

def run_depth_estimation(max_depth, confidence, method, output_dir):
    print(f"Running with Max Depth={max_depth}, Confidence={confidence}, Method={method}")

    # Construct estimation overrides
    estimation_overrides = [
        f"general.max_depth={max_depth}",
        f"detection.bbox_confidence_threshold={confidence}",
        f"sampling.detection_sampling_method={method}"
    ]

    config_dir = f"{method}-{max_depth}_m-confidence_{int(confidence*100)}"

    # Construct the full command
    command = [
                  "python", RUN_ESTIMATION_PATH,
                  f"-cd=configs/depth_estimation",
                  f"--config-name=estimation.yaml",  # Set the config name
                  f"hydra.run.dir={output_dir}/estimation_{config_dir}",  # Set the common output directory
              ] + estimation_overrides

    try:
        # Execute the command
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")


@hydra.main(config_path="configs/depth_estimation", config_name="central_config", version_base="1.1")
def run_estimation(cfg: DictConfig):
    # Create a unified output directory with a timestamp
    script_dir = Path(__file__).resolve().parent / "scripts"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = script_dir.parent / "outputs" / timestamp
    os.makedirs(output_dir, exist_ok=True)

    # Define the list of calibration scripts to run and their corresponding configs
    scripts_to_run = [
        {"script": "data_prep_depth_calibration.py", "config": "calibration"},
        {"script": "seg_masks_depth_calibration.py", "config": "calibration"},
        {"script": "copy_images_depth_estimation.py", "config": "calibration"},
        # {"script": "distance-estimation/run_estimation.py", "config": "estimation"},
    ]

    # Execute each calibration script sequentially
    for entry in scripts_to_run:
        script_path = script_dir / entry["script"]
        config_name = entry["config"] + ".yaml"
        config_dir = cfg.root_config_folder

        command = [
            "python", script_path,
            f"-cd={config_dir}",
            f"--config-name={config_name}",  # Set the config name
            f"hydra.run.dir={output_dir}/{entry['config']}",  # Set the common output directory
        ]
        if cfg.override_present == True:
            current_overrides = cfg[entry["config"]].overrides
            override_str_list = [f"{override}" for override in current_overrides]
            # override_str = " ".join(current_overrides)

            # Run the script using subprocess and Hydra
            command = command + override_str_list

        print(f"Executing {script_path} with config {config_name}")
        subprocess.run(command, check=True)

    # Execute each estimation config sequentially
    result_dir = Path(__file__).resolve().parent / "data" / "DistanceEstimationData" /"results"
    for max_depth in MAX_DEPTHS:
        for confidence in CONFIDENCES:
            for method in METHODS:
                result_dir.mkdir(parents=True, exist_ok=True)
                run_depth_estimation(max_depth, confidence, method, output_dir)


# Iterate over the parameter grid and execute the script
if __name__ == "__main__":
    run_estimation()
