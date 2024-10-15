import logging
import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import shutil
import numpy as np
from matplotlib import pyplot as plt
import torch
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from tqdm import tqdm
import cv2
import regex as re
from pathlib import Path

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

#################################################################################
# Util functions
#################################################################################

### SegFormer SEGMENTATION ###
# https://huggingface.co/docs/transformers/model_doc/segformer
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SegFormer/Segformer_inference_notebook.ipynb
class Segmenter:
    def __init__(self,mode='cityscapes',lightweight=False, resize_first=True, device=torch.device('cpu'), debug=False):
        # mode: 'ade' or 'cityscapes': the dataset on which the model was trained
        # lightweight: True or False: whether to use a lightweight model or not.
        # Pretrained models are available at: https://huggingface.co/nvidia
        # Lightest model is b0, and as the number increases, the model size increases as well. b5 is the largest model.
        # Small note about ADE model: the model was trained on 150 classes.
        # This means they likely used the Scene-Parsing dataset which is a subset of ADE20k dataset. http://sceneparsing.csail.mit.edu/
        if mode == 'ade':
            self.mode = 'ade'
            if lightweight:
                model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
            else:
                model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
        elif mode == 'cityscapes':
            self.mode = 'cityscapes'
            if lightweight:
                model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
            else:
                model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)

        self.resize_first = resize_first

        self.device = device
        self.model = self.model.to(self.device)

        self.debug = debug

    def get_segmentation_labels(self, image):
        # For faster inference, do not calculate gradients.
        with torch.no_grad():
            # image is not PIL image anymore, but a batch of images
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            # Get the predictions
            inputs = inputs.to(self.device)
            outputs = self.model(**inputs)
            # ade --> (B, 150, 512, 512) = (batch_size, num_classes, height, width)
            # outputs.logits.shape:
            # cityscapes_lightweight --> (B, 19, 128, 128) = (batch_size, num_classes, height, width)
            # cityscapes_large       --> (B, 19, 256, 256) = (batch_size, num_classes, height, width)
            # ade_lightweight        --> (B, 150, 128, 128) = (batch_size, num_classes, height, width)
            # ade_large              --> (B, 150, 160, 160) = (batch_size, num_classes, height, width)

            if self.resize_first:
                # Rescale the segmentation mask to the size of the image
                # We can do bilinear interpolation with float data.
                logits = torch.nn.functional.interpolate(outputs.logits,
                                                    size=image.shape[2:],
                                                    mode="bilinear",
                                                    align_corners=False)

                # Do argmax over batch of logits
                seg = logits.argmax(dim=1,keepdim=True) # Each pixel is set to the class with the highest probability
                seg = seg.to(torch.uint8) # Convert to uint8
            else:
                # This variant should be faster, but the output is pixelated.
                # Do argmax over batch of logits
                seg = outputs.logits.argmax(dim=1,keepdim=True) # Each pixel is set to the class with the highest probability
                seg = seg.to(torch.uint8) # Convert to uint8
                # Rescale the segmentation mask to the size of the image
                # We have to use nearest neighbor interpolation this time, because labels are integers, and we can't do bilinear interpolation with integers.
                # if type(image) == torch.Tensor:
                seg = torch.nn.functional.interpolate(seg, size=image.shape[2:],mode="nearest")
                # else if its PIL image:
                    # seg = torch.nn.functional.interpolate(seg,
                    #       size=image.size[::-1], # (width, height) -> (height, width)
                    #       mode="nearest")
        return seg

    def plot_batched_images(self,images,masks):
        batch_size = images.shape[0]
        plt.clf()
        fig, ax = plt.subplots(batch_size,1)
        if batch_size == 1:
            ax = [ax]
        for i in range(batch_size):
            ax[i].imshow(images[i].cpu().permute(1,2,0))
            ax[i].imshow(masks[i].cpu().squeeze(),alpha=0.5)
        plt.show()

    def get_person_masks(self,batched_inputs):
        images = batched_inputs["image"]
        seg_masks = self.get_segmentation_labels(images)

        if self.mode == 'ade':
            # Label 12 corresponds to "person"
            masks = seg_masks == 12
        elif self.mode == 'cityscapes':
            # Label 11 corresponds to "person"
            masks = seg_masks == 11
            # Label 12 corresponds to "rider". One can choose to include this class as well.
            # mask = (seg == 11 or seg == 12)

        # Quick and dirty way to ignore the labels predicted for the banner.
        # We assume that the botom 1/10'th of the image is the area where the banner is:
        batch_size, one, height, width = masks.shape
        masks[:,:,-height//10:,:] = False
        # Future work: It makes more sense to crop out this part entirely.

        if self.debug:
            self.plot_batched_images(images,masks)
        return masks

    def get_label_masks(self,batched_inputs):
        images = batched_inputs["image"]
        seg_masks = self.get_segmentation_labels(images)

        if self.mode == 'ade':
            # Label 12 corresponds to "person"
            masks = seg_masks == 43
        elif self.mode == 'cityscapes':
            raise NotImplementedError

        # Quick and dirty way to ignore the labels predicted for the banner.
        # We assume that the botom 1/10'th of the image is the area where the banner is:
        batch_size, one, height, width = masks.shape
        masks[:,:,-height//10:,:] = False
        # Future work: It makes more sense to crop out this part entirely.

        if self.debug:
            self.plot_batched_images(images)
        return masks
    
class DistanceEstimationData:
    def __init__(self,path):
        self.calibration_frames = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if ('.JPG' in file or '.jpg' in file) and 'calibration_frames' in root and not 'calibration_frames_masks' in root:
                    self.calibration_frames.append(os.path.join(root,file))
    def __getitem__(self, idx):
        # Get image to extract masks from, with its path so that we can save the output.
        # TODO: Preprocess image to resize into a fixed size?
        image = Image.open(self.calibration_frames[idx])
        image = torch.from_numpy(np.array(image))
        if len(image.shape)  == 2:
          print('image.shape: ', image.shape)
          print('img path: ', self.calibration_frames[idx])
        # (H,W,C) -> (C,H,W)
        image = image.permute(2,0,1)
        return {"image": image,
                "path" : self.calibration_frames[idx]}
        pass
    def __len__(self):
        return len(self.calibration_frames)
    
def resize_images_in_directory_if_different(directory_path, target_resolution=(2576, 1984)):
    """
    Resizes images in the given directory (and its subdirectories) to the target resolution,
    only if their current resolution is different. The default target resolution is set to (2576, 1984).
    
    :param directory_path: Path to the directory containing images.
    :param target_resolution: Tuple of the target resolution (width, height), default is (2576, 1984).
    """
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                image_path = os.path.join(root, file)
                with Image.open(image_path) as img:
                    # Check if the image size is different from the target resolution
                    if img.size != target_resolution:
                        resized_img = img.resize(target_resolution)
                        resized_img.save(image_path)
                        log.debug(f"Resized {image_path}")

def generate_seg_masks(segmenter, dataloader, min_mask_pixel_image):
    saved_masks = 0
    ignored_masks = 0
    ignored_masks_list = []
    missed_masks = 0
    missed_masks_list = []

    progress_bar = tqdm(dataloader,desc=f"Saved: {saved_masks}, Ignored: {ignored_masks}, Missed: {missed_masks}")
    for batch in progress_bar:
        # print(batch['path'])
        masks = segmenter.get_person_masks(batch)
        # print(masks.shape)
        for i in range(len(batch['path'])):
            mask = masks[i].squeeze(0).cpu().numpy()
            num_mask_pixels = mask.sum()
            if num_mask_pixels > min_mask_pixel_image:
                saved_masks += 1
      # print('to_save.shape: ',mask.shape)
                mask = mask.astype(np.uint8)
                mask = mask * 255
                mask = Image.fromarray(mask)
      # print('mask.width', mask.width)
      # print('mask.height',mask.height)
                save_path = batch['path'][i].replace('calibration_frames','calibration_frames_masks')

      # Good practice: Do not save binary masks in JPG format. JPG compression is a 'lossy' compression.
      # Due to this 'lossiness', the pixels in the mask will have values other than 0 and 255,
      # so the loaded mask will not be the exact same of the saved mask.
      # Previously, I was not aware of that and saved all the masks in JPG format. Luckily, the loss of
      # information from JPG compression can be ignored for our case. But for the future, I would highly
      # suggest switching to a 'lossless' compression method where no information is lost. For example, png.
      # So in future, consider saving the masks in png format by uncommenting the following line:

      # save_path = save_path.replace('.JPG', '.png')

      # print('save_dir: ', os.path.dirname(save_path))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                mask.save(save_path)
            # print('saved mask to: ', save_path)
            elif num_mask_pixels == 0:
                missed_masks += 1
                missed_masks_list.append(batch['path'][i])
            else:
                ignored_masks += 1
                ignored_masks_list.append(batch['path'][i])
            progress_bar.set_description(f"Saved: {saved_masks}, Ignored: {ignored_masks}, Missed: {missed_masks}")

    log.info(f"Finished Segmentation! Saved masks: {saved_masks}, Ignored masks: {ignored_masks}, Missed masks: {missed_masks}")

    #TODO: find way to use ignored maskes anyway, and investigate why ignored
    for item in ignored_masks_list:
        os.remove(item)
        log.debug('Successfully removed ', item)

    #TODO: find way to use missed maskes anyway
    for item in missed_masks_list:
        os.remove(item)
        log.debug('Successfully removed ', item)

def remove_smaller_regions(binary_mask, size_threshold):
    # Connected Component Labeling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=4)

    # Initialize a new mask
    new_mask = np.zeros_like(binary_mask)

    # Iterate through all regions (excluding background which is label 0)
    for label in range(1, num_labels):
        # Check if the region size is above the threshold
        if stats[label, cv2.CC_STAT_AREA] > size_threshold:
            # Include this region in the new mask
            new_mask[labels == label] = 255

    return new_mask

def get_white_pixels(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Count the number of white pixels (assuming white is 255)
    num_white_pixels = np.sum(image == 255)
    return num_white_pixels

def determine_correct_name(path):
    # Extract the base file name from the path
    file_name = os.path.basename(path)
    
    # Find the position of the first digit between 1-9
    match = re.search(r'([1-9]\d?)', file_name)
    if match:
        # Return the number found followed by 'm.jpg'
        return match.group() + 'm.JPG'
    else:
        # If no matching number sequence is found, return None
        return None
    
def get_corresponding_filepath(file_path, from_subdir="calibration_frames_masks", to_subdir="calibration_frames"):
    return file_path.replace(from_subdir, to_subdir)

def rename_files_with_bigger_mask(file_list):
    for file_path in file_list:
        # file_name = os.path.basename(file_path)
        correct_name = determine_correct_name(file_path)
        directory = os.path.dirname(file_path)
        
        if correct_name:
            correct_path = os.path.join(directory, correct_name)
            
            if os.path.exists(correct_path):
                # If the correct file name already exists, compare mask sizes
                existing_mask_size = get_white_pixels(correct_path)
                current_mask_size = get_white_pixels(file_path)

                # Keep the file with the bigger mask
                if current_mask_size > existing_mask_size:
                    os.remove(correct_path)
                    os.remove(get_corresponding_filepath(correct_path))
                    os.rename(file_path, correct_path)
                    os.rename(get_corresponding_filepath(file_path), get_corresponding_filepath(correct_path))
                    log.debug("Removed ", correct_path, " and kept ", file_path, " under new name, as it has the bigger mask.")
                else: 
                    os.remove(file_path)
                    os.remove(get_corresponding_filepath(file_path))
                    log.debug("Removed ",  file_path, ", as ", correct_path, " has the bigger mask.") 
            else:
                # No conflict with the new name, so we can rename the file
                os.rename(file_path, correct_path)
                os.rename(get_corresponding_filepath(file_path), get_corresponding_filepath(correct_path))
                log.debug("Renamed ", file_path," to ", correct_path, ".")
        else:
            os.remove(file_path)
            os.remove(get_corresponding_filepath(file_path))
            log.debug("File ", file_path, " removed, as it has no correct distance assigned.")
  

@hydra.main(config_path="../configs/depth_estimation", config_name="calibration", version_base="1.1")
def run_segmentation(cfg: DictConfig):
    log.info(f"Configurations: {cfg}")
    # GPU set-up
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu.used_gpu_idx
    DEVICE = torch.device(cfg.gpu.cuda_idx if torch.cuda.is_available() else 'cpu')
    log.info("CUDA_VISIBLE_DEVICES set to:", os.environ.get('CUDA_VISIBLE_DEVICES'))

    # directory set-up
    dataset_path = Path(__file__).resolve().parent.parent / cfg.directories.clean_data_dir
    dataset_path = dataset_path.resolve()
    transect_path = dataset_path / "transects"

    #### Execute mask generation ####
    # Set all images to the same size
    resize_images_in_directory_if_different(directory_path=transect_path)

    # Initialize all objects
    dataset = DistanceEstimationData(dataset_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.mask_generation.batch_size, num_workers=cfg.mask_generation.num_workers)
    segmenter = Segmenter(
        mode=cfg.mask_generation.segmenter_pretrained_in,
        lightweight=cfg.mask_generation.lightweight_segmenter,
        resize_first=cfg.mask_generation.resize_first,
        device=DEVICE,
        debug=cfg.mask_generation.debug_mode,
    )
    log.info("Length of loaded data set: ", len(dataset))

    # Create segmentation masks
    generate_seg_masks(segmenter=segmenter, dataloader=dataloader, min_mask_pixel_image= cfg.mask_generation.min_mask_pixel_image)

    # remove small masks in each picture
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if 'calibration_frames_masks' in root:
                mask_path = os.path.join(root, file)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):
                    # Due to lossy compression, the mask values are not exactly 0 and 255
                    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                elif file.lower().endswith('.png'):
                    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
                new_mask = remove_smaller_regions(binary_mask, size_threshold = cfg.mask_generation.min_mask_pixel_area)
                # if mask is too small, it is removed later
                cv2.imwrite(os.path.join(root, file), new_mask)


    # pictures with irregular names              
    ireggular_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            # File name should be in the format of {integer number}m.JPG
            if 'calibration_frames_masks' in root and not re.match(r'\d+m\.JPG', file):
                ireggular_files.append(os.path.join(root, file))


    # remove all files that have no distance
    # remove all files that have 2x a distance and keep the bigger mask
    # rename in case of irregularities
    rename_files_with_bigger_mask(ireggular_files)

    # rename all files without the "m" for Timm Haucke pipeline
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            os.rename(os.path.join(root, file), os.path.join(root, file.replace('m.JPG', '.JPG')))
            
    # remove small mask pictures
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            # File name should be in the format of {integer number}m.JPG
            if 'calibration_frames_masks' in root:
                file_path = os.path.join(root, file)
                if get_white_pixels(file_path) < cfg.mask_generation.min_mask_pixel_image:
                    log.debug(f"Removed directory, because of small mask: ", file_path)
                    os.remove(file_path)
                    os.remove(get_corresponding_filepath(file_path))


    # remove folders with less then 2 calibration frames
    for vf_dir in os.listdir(transect_path):
        vf_path = os.path.join(transect_path, vf_dir)
        masks_path = os.path.join(vf_path, 'calibration_frames_masks')

        # Check if masks_path is a directory
        if os.path.isdir(masks_path):
            num_files = len([name for name in os.listdir(masks_path) if os.path.isfile(os.path.join(masks_path, name))])

            # Remove the VF_### folder if it contains 0 or 1 image files
            if num_files <= 1:
                shutil.rmtree(vf_path)
                log.debug(f"Removed directory, because of less then 2 calibration frames: {vf_path}")

    # Print final number of transects and calibration frames
    # Initialize counters
    total_transects = 0
    total_calibration_frames = 0

    # Walk through the dataset directory
    for root, dirs, files in os.walk(dataset_path):
        if 'calibration_frames_masks' in root:
            # Check if the current directory has any valid image files
            valid_files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if valid_files:
                total_transects += 1  # Increment transect count for directories with valid image files
                total_calibration_frames += len(valid_files)  # Increment by the number of valid image files

    # Print the final counts
    log.info(f"Total number of transects: {total_transects}")
    log.info(f"Total number of calibration frames: {total_calibration_frames}")


if __name__ == "__main__":
    run_segmentation()