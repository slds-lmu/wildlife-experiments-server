import os
import subprocess
import hydra
from omegaconf import DictConfig
from datetime import datetime
from pathlib import Path

@hydra.main(config_path="../configs/depth_estimation", config_name="central_config", version_base="1.1")
def central_executor(cfg: DictConfig):
    # Create a unified output directory with a timestamp
    script_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = script_dir.parent / "outputs" / timestamp
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the list of scripts to run and their corresponding configs
    scripts_to_run = [
        {"script": "data_prep_depth_calibration.py", "config": "calibration"},
        {"script": "seg_masks_depth_calibration.py", "config": "calibration"},
        {"script": "copy_images_depth_estimation.py", "config": "calibration"},
        {"script": "distance-estimation/run_estimation.py", "config": "estimation"},
    ]
    
    # Execute each script sequentially
    for entry in scripts_to_run:
        script_path = script_dir / entry["script"]
        config_name = entry["config"] + ".yaml"
        config_dir = cfg.root_config_folder

        # Run the script using subprocess and Hydra
        command = [
            "python", script_path,
            f"-cd={config_dir}",
            f"--config-name={config_name}",  # Set the config name
            f"hydra.run.dir={output_dir}/{entry['config']}",  # Set the common output directory
            # f"hydra.output_subdir=null",  # Disable subdirectory creation
        ]
        
        print(f"Executing {script_path} with config {config_name}")
        subprocess.run(command, check=True)

if __name__ == "__main__":
    central_executor()
