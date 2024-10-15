import logging
import os
import shutil
from pathlib import Path
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


# Function to filter empty subfolders
def filter_empty_subfolders(source_directory: Path):
    valid_folders = []
    for root, dirs, files in os.walk(source_directory, topdown=False):
        for name in dirs:
            full_path = Path(root) / name
            if any(full_path.iterdir()):
                valid_folders.append(full_path)
    return valid_folders

# Function to filter 'Kronenschlussgrad' folders
def filter_kronenschlussgrad_folders(source_directory: Path):
    valid_folders = []
    for folder_name in os.listdir(source_directory):
        folder_path = source_directory / folder_name
        if folder_path.is_dir():
            subfolders = [name for name in os.listdir(folder_path) if (folder_path / name).is_dir()]
        if not (len(subfolders) == 1 and subfolders[0] == "Kronenschlussgrad"):
            valid_folders.append(folder_path)
    return valid_folders

# Function to filter unstructured folders
def filter_unstructured_folders(directory_path: Path, possible_sources):
    valid_folders = []
    for folder_path in directory_path.iterdir():
        if folder_path.is_dir():
            found_sources = [folder_path / source for source in possible_sources if (folder_path / source).exists()]
            if len(found_sources) == 1:
                source_path = found_sources[0]
                entries = list(source_path.iterdir())
                if len(entries) != 1:
                    valid_folders.append(folder_path)
    return valid_folders

# Function to copy images with structure
def copy_images_with_structure(valid_folders, destination_directory: Path, possible_sources):
    destination_directory.mkdir(parents=True, exist_ok=True)
    for folder_path in valid_folders:
        source_subfolder = next((sub for sub in possible_sources if (folder_path / sub).exists()), 'Distanzfotos')
        source_path = folder_path / source_subfolder
        destination_path = destination_directory / 'transects' / folder_path.name / 'calibration_frames'
        if source_path.exists() and any(source_path.iterdir()):
            destination_path.mkdir(parents=True, exist_ok=True)
            for file in source_path.iterdir():
                if file.is_file():
                    shutil.copy(file, destination_path)
                    # print(f"Copied file: {file} to {destination_path}")

# Main function to run the script with Hydra configuration
@hydra.main(config_path="../configs/depth_estimation", config_name="calibration", version_base="1.1")
def run_calibration_image_sorting(cfg: DictConfig):
    log.info(f"Configurations: {cfg}")

    clean_data_dir = Path(__file__).resolve().parent.parent / cfg.directories.clean_data_dir  
    clean_data_dir = clean_data_dir.resolve()

    # clean_data_dir = Path(cfg.directories.clean_data_dir)
    result_dir = clean_data_dir / 'results'
    transect_dir = clean_data_dir / 'transects'

    # Remove the directory if it already exists
    if clean_data_dir.exists() and clean_data_dir.is_dir():
        shutil.rmtree(clean_data_dir)

    # Create the destination folder structure
    clean_data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    transect_dir.mkdir(parents=True, exist_ok=True)

    # Define the source directory
    source_dir = Path(__file__).resolve().parent.parent / cfg.directories.raw_calibration_data_dir / "Distancefotos"
    source_dir = source_dir.resolve()
    # source_dir = Path(cfg.directories.raw_calibration_data_dir + "/Distancefotos/")
    source_dir.mkdir(parents=True, exist_ok=True)

    # Get list of valid folders
    valid_folders = filter_empty_subfolders(source_dir)
    valid_folders = [folder for folder in valid_folders if folder in filter_kronenschlussgrad_folders(source_dir)]
    valid_folders = [folder for folder in valid_folders if folder in filter_unstructured_folders(source_dir, cfg.directories.possible_sources)]

    log.info(f"Found {len(valid_folders)} populated transects in raw calibration data directory.")
    # Copy valid calibration frames
    copy_images_with_structure(valid_folders, clean_data_dir, cfg.directories.possible_sources)
    log.info("Data preparation executed.")

if __name__ == "__main__":
    run_calibration_image_sorting()
