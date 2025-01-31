from collections import OrderedDict
import logging
import os
import shutil
import glob
import math
import csv
import numpy as np
import cv2
import hydra
from omegaconf import DictConfig
from pathlib import Path

from dpt import DPT
from megadetector import MegaDetector, MegaDetectorLabel
from sam import SAM
from custom_types import DetectionSamplingMethod, MultipleAnimalReduction, SampleFrom
from utils import calibrate, crop, resize, exception_to_str, get_calibration_frame_dist, get_extension_agnostic_path, multi_file_extension_glob
from visualization import visualize_detection, visualize_farthest_calibration_frame
from custom_types import DetectionSamplingMethod, SampleFrom, MultipleAnimalReduction, RegressionMethod

log = logging.getLogger(__name__)

@hydra.main(config_path="../../configs/depth_estimation", config_name="estimation", version_base="1.1")
def run_depth_estimation(cfg: DictConfig):
    log.info(f"Configurations: {cfg}")
    
    eps = 1e-6
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_DIR = (SCRIPT_DIR.parent.parent / cfg.general.data_dir).resolve()

    assert os.path.isdir(DATA_DIR), "Data dir is not a directory"
    assert os.path.isdir(os.path.join(DATA_DIR, "transects")) and os.path.isdir(os.path.join(DATA_DIR, "results")), "Data dir must contain 'transect' and 'results' subdirectories. Please consult the manual for the correct directory structure."
    assert len(glob.glob(os.path.join(DATA_DIR, "transects", "*/"))), "The 'transect' subdirectory must contain at least one transect. Please consult the manual for the correct directory structure."
    
    if cfg.calibration.calibration_regression_method == "RANSAC":
        calibration_regression_method = RegressionMethod.RANSAC
    elif cfg.calibration.calibration_regression_method == "LEASTSQUARES":
        calibration_regression_method = RegressionMethod.LEASTSQUARES
    elif cfg.calibration.calibration_regression_method == "POLY":
        calibration_regression_method = RegressionMethod.POLY
    elif cfg.calibration.calibration_regression_method == "RANSAC_POLY":
        calibration_regression_method = RegressionMethod.RANSAC_POLY
        
    if cfg.sampling.detection_sampling_method == "BBOX_PERCENTILE":
        detection_sampling_method = DetectionSamplingMethod.BBOX_PERCENTILE
    elif cfg.sampling.detection_sampling_method == "BBOX_BOTTOM":
        detection_sampling_method = DetectionSamplingMethod.BBOX_BOTTOM
    elif cfg.sampling.detection_sampling_method == "SAM":
        detection_sampling_method = DetectionSamplingMethod.SAM

    if cfg.sampling.multiple_animal_reduction == "ONLY_CENTERMOST":
        multiple_animal_reduction = MultipleAnimalReduction.ONLY_CENTERMOST
    elif cfg.sampling.multiple_animal_reduction == "NONE":
        multiple_animal_reduction = MultipleAnimalReduction.NONE
    elif cfg.sampling.multiple_animal_reduction == "MEDIAN":
        multiple_animal_reduction = MultipleAnimalReduction.MEDIAN

    if cfg.sampling.sample_from == "DETECTION":
        sampling_from = SampleFrom.DETECTION
    elif cfg.sampling.sample_from == "REFERENCE":
        sampling_from = SampleFrom.REFERENCE

    dpt = DPT()
    
    megadetector = MegaDetector()

    if detection_sampling_method == DetectionSamplingMethod.SAM:
        sam = SAM()

    with open(os.path.join(DATA_DIR, "results", "results.csv"), "w", newline="") as result_csv_file, open(os.path.join(DATA_DIR, "results", "results.txt"), "w") as result_distance_file: 
        head_row_csv = ["transect_id", "frame_id", "detection_idx", "detection_confidence", "depth", "world_x", "world_y", "world_z", "location_pixel_x", "location_pixel_y"]
        head_row_txt = ["Camera trap*Label", "Observation*Radial distance"]
        result_csv_writer = csv.writer(result_csv_file) 
        result_csv_writer.writerow(head_row_csv)
        result_distance_file.write("\t".join(head_row_txt) + os.linesep)

        transect_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "transects", "*/")))
        for transect_idx, transect_dir in enumerate(transect_dirs):
            first_image_in_transect = True
            transect_id = os.path.basename(os.path.normpath(transect_dir))
            log.debug(f"Calibrating Transect {str(transect_id)}")

            exp = -1 if cfg.calibration.calibrate_metric else 1
            calibration_frames = {}

            for calibration_frame_filename in (
                multi_file_extension_glob(os.path.join(transect_dir, "calibration_frames", "*"), cfg.file_extensions.intensity_image_extensions) +
                multi_file_extension_glob(os.path.join(transect_dir, "calibration_frames_cropped", "*"), cfg.file_extensions.intensity_image_extensions)  # for backwards compability. use crop configuration instead
            ):
                
                calibration_frame_id = os.path.splitext(
                    os.path.basename(calibration_frame_filename)
                )[0]
                dist = get_calibration_frame_dist(transect_dir, calibration_frame_id)
                img = crop(
                    cv2.imread(calibration_frame_filename),
                    cfg.cropping.crop_top, cfg.cropping.crop_bottom, cfg.cropping.crop_left, cfg.cropping.crop_right,
                )
                mask = crop(
                    cv2.imread(
                        get_extension_agnostic_path(
                            os.path.join(
                                transect_dir,
                                "calibration_frames_masks",
                                calibration_frame_id,
                            ),
                            cfg.file_extensions.intensity_image_extensions,
                        ),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    > 127,
                    cfg.cropping.crop_top, cfg.cropping.crop_bottom, cfg.cropping.crop_left, cfg.cropping.crop_right,
                )
                disp = dpt(img)
                disp = np.ma.masked_where(mask, disp)
                calibration_frames[dist] = disp

            
            # sort calibration frames
            calibration_frames = OrderedDict(sorted(calibration_frames.items(), key=lambda kv: kv[0]))

            # get disparity of the farthest calibration frame
            farthest_calibration_frame_disp = list(calibration_frames.values())[-1] if len(calibration_frames) > 0 else None
            try:
                x,y  = [], []
                for dist, disp in calibration_frames.items():
                    
                    disp = resize(disp, farthest_calibration_frame_disp.shape)
                    if cfg.calibration.calibrate_metric:
                        disp = np.clip(disp, eps, np.inf)

                    disp_calibrated = calibrate(
                        disp ** exp,
                        farthest_calibration_frame_disp ** exp,
                        calibration_regression_method,
                    )(disp.data ** exp) ** exp
                    disp_calibrated = np.ma.masked_where(disp.mask, disp_calibrated)

                    x.append(np.median(disp_calibrated.data[disp_calibrated.mask]))
                    y.append(dist ** -1)

                calibration = calibrate(np.array(x) ** exp, np.array(y) ** exp, calibration_regression_method)
                farthest_calibration_frame_disp = np.ma.masked_where(
                    farthest_calibration_frame_disp.mask,
                    calibration(farthest_calibration_frame_disp.data ** exp) ** exp,
                )
            except Exception as e:
                calibration = None
                farthest_calibration_frame_disp = None
                if not os.path.exists(os.path.join(transect_dir, "detection_frames")):
                    log.warning(f"Failed calibrating transect '{str(transect_id)}' due to exception: {exception_to_str(e)}. Skipping all distance estimations for observations in this transect.")

            if cfg.visualization.make_figures and farthest_calibration_frame_disp is not None:
                visualize_farthest_calibration_frame(DATA_DIR, transect_id, farthest_calibration_frame_disp, cfg.general.min_depth, cfg.general.max_depth)

            detection_frame_filenames = sorted(list(set(
                multi_file_extension_glob(os.path.join(transect_dir, "detection_frames", "*"), cfg.file_extensions.intensity_image_extensions) +
                multi_file_extension_glob(os.path.join(transect_dir, "detection_frames_cropped", "*"), cfg.file_extensions.intensity_image_extensions)  # for backwards compability. use crop configuration instead
            )))
            len_detection_files = len(detection_frame_filenames)
            if len_detection_files > 0:
                log.info(f"Performing depth estimation in transect {transect_id}")
            for detection_idx, detection_frame_filename in enumerate(detection_frame_filenames):
                
                detection_id = os.path.splitext(os.path.basename(detection_frame_filename))[0]

                # load intensity image
                img = cv2.imread(detection_frame_filename)

                # crop and resize intensity image to have the same size as the reference images
                img = crop(
                    img,
                    cfg.cropping.crop_top, cfg.cropping.crop_bottom, cfg.cropping.crop_left, cfg.cropping.crop_right,
                )
                img = resize(img, farthest_calibration_frame_disp.shape)

                # check if depth from stereo camera exists or calibration succeeded
                precomputed_depth_filename = get_extension_agnostic_path(os.path.join(transect_dir, "detection_frames_depth", detection_id), cfg.file_extensions.depth_image_extensions)
                if precomputed_depth_filename is None and farthest_calibration_frame_disp is None:
                    log.warning(f"Unable to perform distance estimation on detection '{detection_id}' due to failed calibration and no precomputed depth maps.")
                    continue
                elif precomputed_depth_filename is not None:
                    assert sampling_from == SampleFrom.DETECTION, "Config must be set to sample from detection if using precomputed depth maps"
                    depth = cv2.imread(precomputed_depth_filename, cv2.IMREAD_UNCHANGED)
                    disp = np.clip(depth, cfg.general.min_depth, cfg.general.max_depth) ** -1
                elif precomputed_depth_filename is None and farthest_calibration_frame_disp is not None:
                    if sampling_from == SampleFrom.DETECTION:
                        disp = dpt(img)
                        if cfg.calibration.calibrate_metric:
                            disp = np.clip(disp, eps, np.inf)
                        disp = calibrate(disp ** exp, farthest_calibration_frame_disp ** exp, calibration_regression_method)(disp ** exp)
                        if cfg.calibration.calibrate_metric:
                            disp = np.clip(disp, cfg.general.min_depth, cfg.general.max_depth) ** -1
                    elif sampling_from == SampleFrom.REFERENCE:
                        disp = farthest_calibration_frame_disp
                    else:
                        raise RuntimeError(f"Invalid configuration value '{sampling_from}' for configuration sample_from")
                    depth = np.clip(disp, cfg.general.max_depth ** -1, cfg.general.min_depth ** -1) ** -1

                # run animal detection
                scores, labels, boxes = megadetector(img)

                # discard all non-animal detections
                if cfg.detection.detect_humans:
                    correct_label_idx = np.nonzero((labels.flatten() == MegaDetectorLabel.ANIMAL) | (labels.flatten() == MegaDetectorLabel.PERSON))
                else:
                    correct_label_idx = np.nonzero(labels.flatten() == MegaDetectorLabel.ANIMAL)
                scores, labels, boxes = scores[correct_label_idx], labels[correct_label_idx], boxes[correct_label_idx]

                # discard all detections with low confidence
                high_confidence_idx = np.nonzero(scores.flatten() >= cfg.detection.bbox_confidence_threshold)
                scores, labels, boxes = scores[high_confidence_idx], labels[high_confidence_idx], boxes[high_confidence_idx]

                # sort from image center outwards
                centerness = [((img.shape[1] / 2) - (box[0] + box[2] / 2)) ** 2 + ((img.shape[0] / 2) - (box[1] + box[3] / 2)) ** 2 for box in boxes]
                centerness_idx = np.argsort(centerness)
                scores, labels, boxes = scores[centerness_idx], labels[centerness_idx], boxes[centerness_idx]

                if detection_sampling_method == DetectionSamplingMethod.SAM:
                    # compute SAM masks
                    masks = sam(img, boxes)

                    
                else:
                    # dummy masks
                    masks = [None for _ in boxes]

                sampled_depths = []
                sample_locations = []
                world_positions = []
                for box, mask in zip(boxes, masks):
                    # megadetector bounding box format: xmin, ymin, xmax, ymax, wih origin at top-left corner
                    if box[2] <= box[0] or box[3] <= box[1]:  # sanity check
                        continue
                    if detection_sampling_method == DetectionSamplingMethod.BBOX_BOTTOM:
                        sample_location = (
                            max(0, min(depth.shape[0] - 1, round(box[3]))),
                            max(0, min(depth.shape[1] - 1, round(box[0] + (box[2] - box[0]) / 2))),
                        )
                        sampled_depths += [depth[sample_location]]
                        sample_locations += [sample_location]
                    elif detection_sampling_method == DetectionSamplingMethod.BBOX_PERCENTILE:
                        ymin, ymax = max(0, min(depth.shape[0] - 2, round(box[1]))), max(0, min(depth.shape[0] - 1, round(box[3])))
                        xmin, xmax = max(0, min(depth.shape[1] - 2, round(box[0]))), max(0, min(depth.shape[1] - 1, round(box[2])))
                        depth_cropped = depth[ymin:ymax, xmin:xmax]
                        sampled_depths += [np.percentile(depth_cropped, cfg.sampling.bbox_sampling_percentile, method="nearest")]
                        sample_location = np.nonzero(depth_cropped == sampled_depths[-1])
                        sample_location = (
                            round(sample_location[0][0] + box[1]),
                            round(sample_location[1][0] + box[0]),
                        )
                        sample_locations += [sample_location]
                    elif detection_sampling_method == DetectionSamplingMethod.SAM:
                        ymin, ymax = max(0, min(depth.shape[0] - 2, round(box[1]))), max(0, min(depth.shape[0] - 1, round(box[3])))
                        xmin, xmax = max(0, min(depth.shape[1] - 2, round(box[0]))), max(0, min(depth.shape[1] - 1, round(box[2])))
                        depth_cropped = depth[ymin:ymax, xmin:xmax]
                        mask_cropped = mask[ymin:ymax, xmin:xmax]
                        mask_paddded = np.pad(mask_cropped, ((1, 1), (1, 1)))
                        dist = cv2.distanceTransform((mask_paddded * 255).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_3)
                        sample_location = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
                        sample_location = (
                            max(0, min(mask_cropped.shape[0], sample_location[0] - 1)),
                            max(0, min(mask_cropped.shape[1], sample_location[1] - 1)),
                        )
                        sampled_depths += [depth_cropped[sample_location[0], sample_location[1]]]
                        sample_location = (
                            round(sample_location[0] + box[1]),
                            round(sample_location[1] + box[0]),
                        )
                        sample_locations += [sample_location]
                    else:
                        raise RuntimeError(f"Invalid configuration value '{detection_sampling_method}' for configuration detection_sampling_method")

                    # compute horizontal angle a (updated from timmh/distance-estimation/commit/6f31bc5)
                    f = (0.5 * depth.shape[1]) / math.tan(0.5 * math.pi * cfg.camera.camera_horizontal_fov / 180)
                    c = np.array([0, 0, f])
                    p = np.array([(box[0] + box[2]) / 2 - depth.shape[1] / 2, 0, f])  # corrections
                    a = math.copysign(1, (box[0] + box[2]) / 2 - depth.shape[1] / 2) * math.acos((c @ p) / (np.linalg.norm(c) * np.linalg.norm(p)))

                    # compute vertical angle b (updated from timmh/distance-estimation/commit/6f31bc5)
                    f = (0.5 * depth.shape[0]) / math.tan(0.5 * math.pi * cfg.camera.camera_vertical_fov / 180)
                    c = np.array([0, 0, f])
                    p = np.array([0, (box[1] + box[3]) / 2 - depth.shape[0] / 2, f])  # corrections
                    b = math.copysign(1, (box[1] + box[3]) / 2 - depth.shape[0] / 2) * math.acos((c @ p) / (np.linalg.norm(c) * np.linalg.norm(p)))

                    # compute world position # corrections
                    z = sampled_depths[-1] / math.sqrt(math.tan(a) ** 2 + math.tan(b) ** 2 + 1)
                    x = z * math.tan(a)
                    y = z * math.tan(b)
                    world_positions += [[x, y, z]]

                    if multiple_animal_reduction == MultipleAnimalReduction.ONLY_CENTERMOST:
                        break

                if multiple_animal_reduction == MultipleAnimalReduction.MEDIAN:
                    sampled_depths = [np.median(sampled_depths)] if len(sampled_depths) > 0 else []
                    world_positions = [np.mean(world_positions, axis=0)]

                if first_image_in_transect:
                    if cfg.visualization.make_figures:
                        visualize_detection(DATA_DIR, detection_id, img, depth, farthest_calibration_frame_disp, boxes, masks, world_positions, sample_locations, cfg.visualization.draw_detection_ids, cfg.visualization.draw_world_position, cfg.general.min_depth, cfg.general.max_depth)

                
                for i, (score, sampled_depth, world_position, sample_location) in enumerate(zip(scores, sampled_depths, world_positions, sample_locations)):
                    detection_i = i if multiple_animal_reduction != MultipleAnimalReduction.MEDIAN else -1
                    result_csv_writer.writerow([transect_id, detection_id, f"{detection_i:03d}", f"{score.item():.4f}", f"{sampled_depth.item():.4f}", f"{world_position[0].item():.4f}", f"{world_position[1].item():.4f}", f"{world_position[2].item():.4f}", f"{sample_location[0]:.4f}", f"{sample_location[1]:.4f}"])
                    result_distance_file.write("\t".join([transect_id, f"{sampled_depth.item():.4f}"]) + os.linesep)

                first_image_in_transect = False

    shutil.move(DATA_DIR / "results", Path(os.getcwd()) / "results")


if __name__ == "__main__":
    run_depth_estimation()