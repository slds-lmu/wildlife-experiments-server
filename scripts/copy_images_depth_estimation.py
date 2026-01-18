import logging
import os
import shutil
from pathlib import Path
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def delete_images_in_detection_frames(root_dir):
    """
    Deletes all images in folders named 'detection_frames' within the given directory tree.
    
    Args:
    - root_dir (str): The root directory to start the search.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) == 'detection_frames':
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    os.remove(file_path)
                    log.debug(f"Deleted: {file_path}")

def copy_images_to_detection_frames(source_dir, target_dir):
    """
    Copies images from the source directory tree to corresponding 'detection_frames' folders 
    in the target directory tree if the higher-level folders have the same name.
    
    Args:
    - source_dir (str): The source directory to copy images from.
    - target_dir (str): The target directory to copy images to.
    """
    for src_dirpath, src_dirnames, src_filenames in os.walk(source_dir):
        # src_dirpath is in the format <source_dir>/<dir_name>
        for src_filename in src_filenames:
            if src_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                relative_dir = os.path.relpath(src_dirpath, source_dir)
                tgt_detection_frames = os.path.join(target_dir, relative_dir, 'detection_frames')
                
                if os.path.exists(os.path.dirname(tgt_detection_frames)):
                    Path(tgt_detection_frames).mkdir(parents=True, exist_ok=True)
                    src_file = os.path.join(src_dirpath, src_filename)
                    tgt_file = os.path.join(tgt_detection_frames, src_filename)
                    
                    shutil.copy2(src_file, tgt_file)
                    log.debug(f"Copied {src_file} to {tgt_file}")
                else:
                    log.info(f"Camera {relative_dir} is not calibrated.")

@hydra.main(config_path="../configs/depth_estimation", config_name="calibration", version_base="1.1")
def run_copy_detection_images(cfg: DictConfig):
    log.info(f"Configurations: {cfg}")
    SCRIPT_DIR = Path(__file__).resolve().parent
    DESTINATION_DIR = (SCRIPT_DIR.parent / cfg.directories.clean_data_dir).resolve()
    DESTINATION_TRANSECTS_DIR = DESTINATION_DIR / 'transects'
    SOURCE_DATA_DIR = (SCRIPT_DIR.parent / cfg.directories.raw_trap_data_dir).resolve()

    # Step 1: Delete images in 'detection_frames' folders
    delete_images_in_detection_frames(DESTINATION_TRANSECTS_DIR)

    # Step 2: Copy images from the source directory to the 'detection_frames' folders in the target directory
    copy_images_to_detection_frames(SOURCE_DATA_DIR, DESTINATION_TRANSECTS_DIR)

    
if __name__ == "__main__":
    run_copy_detection_images()