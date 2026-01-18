# wildlife-experiments-server


## Description

The **Wildlife Experiments Server** is a flexible and scalable platform for automated wildlife image classification and depth estimation. This project integrates a comprehensive active learning pipeline for identifying species in camera trap images, enhancing the efficiency of ecological research. The system leverages deep learning models, such as MegaDetector, and both PyTorch and TensorFlow environments for model training and inference.

Researchers can train models on pre-labeled data, fine-tune them using active learning to label additional images, and evaluate models on a test set for generalization. The platform supports both passive and active learning approaches, enabling the iterative improvement of models by selecting the most informative images for labeling. Outputs include detailed metrics for species classification performance and confusion matrices for model evaluation.

Additionally, this project supports predictions on large volumes of unlabeled images, with options to sort outputs into folders by species. Designed for flexibility, users can customize data storage and model training settings for parallel experiments. The project also includes a pre-trained model and detailed instructions for getting started quickly.

For unlabeled data, there is an option for performing depth estistimation with (coming soon) or without classification. It uses the  methodology proposed in "Overcoming the distance estimation bottleneck in estimating animal abundance with camera traps" [[`Ecological Informatics`](https://doi.org/10.1016/j.ecoinf.2021.101536)] [[`arXiv`](https://arxiv.org/abs/2105.04244)] using the since released [MegaDetector 5.0](https://github.com/microsoft/CameraTraps/releases/tag/v5.0), [Dense Prediction Transformers](https://github.com/isl-org/DPT), and [Segment Anything](https://github.com/facebookresearch/segment-anything). The methodology was expanded by LMU Munich by automatically segement the calibration images. We utilize the pre-trained segmentation network **SegFormer** from NVIDIA Research, available on Huggingface ([SegFormer-b5](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640)). 


## Table of Contents
- [Installation](#installation)
- [Data Preperation](#data-preparation)
- [Environments](#environments)
- [Usage](#usage)
- [Other Info](#other-info)
- [Citation](#citation)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/slds-lmu/wildlife-experiments-server.git
   ```
2. Create conda environment:
   ```bash
   conda create -n wildlife_torch python==3.9 # use this for depth estimation
   conda create -n wildlife_tf python==3.10
   ```
3. Install dependencies:
   ```bash
   conda activate wildlife_torch
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements_torch.txt
   pip install - r requirements_depth.txt
   ```

   ```bash
   conda activate wildlife_tf
   pip install -r requirements_tf.txt
   ```

## Data Preperation

### Classification
User has to set up the following before they can get started:
- `data.csv`: This file contains file names and classes of already labeled images, using the following column names: ![Column Names Labeled](https://github.com/slds-lmu/wildlife-experiments-server/raw/main/documentation/column_names_labeled.png)
  - Make sure to include images labeled as `Empty` in the file! The file should include images with animals as well as empty ones to ensure the best performance of the pipeline.
- `unlabeled.csv`: This file contains file names of unlabeled images using the following column names: ![Column Names Unlabeled](https://github.com/slds-lmu/wildlife-experiments-server/raw/main/documentation/column_names_unlabeled.png)

The main difference between the setup of those two files is that the labeled file contains the column `Species_Class`. 

You can either put these in the folder `data`, which would be the default. Or put it wherever you want. This means you can work on multiple projects in parallel. However, if you decide to put the files into your selected folder, you will need to remember to always declare the folder using flag `-d` while running the command (more details on how to do it in the next parts). If you decide to store your data in a separate folder, make sure that the `data.csv` and `unlabeled.csv` corresponding to one project are always stored together and not in separate folders! 

### Depth Estimation

The depth estimation is based on calibrating all locations with calibration images.
The default location is in 
```bash
directories:
  raw_calibration_data_dir: "data/Distancefotos"
```
and can be specified in `configs/depth_estimation/calibration.yaml`. The folder can be obtained from [\\zdvl-sv-lwf1v.stmlf.bayern.de\ff-wolf$\Standorte\Distanzfotos](\\zdvl-sv-lwf1v.stmlf.bayern.de\ff-wolf$\Standorte\Distanzfotos).

Example: 
Distancefotos
	VF_001
		Distanzfotos
		Kronenschlussgrad
	VF_002
		Distanzfotos
		Kronenschlussgrad
	...

The second data set is the one, where depth estimation is applied. It defaults to
```bash
directories:
  raw_trap_data_dir: "../Images/Distance_Estimation/Red_Deer"
```
and it is important that here also the folder contains foldes named after the location the image is taken at.

Example: 
Red_Deer
	VF_006
	VF_009
	...


## Environments 

We utilized two different environments for our scripts. One environment based on PyTorch was used for the data preparation step (`data_prep.py`), where [MegaDetector](https://github.com/agentmorris/MegaDetector) is utilized. You can find the requirements for this environment in `requirements_torch.txt`. It is also used for depth estimation.

The other environment is based on TensorFlow and is used for all the other scripts in this repository. You can find its requirements in `requirements_tf.txt`.

## Usage 
### Classification
#### Data Setup

The training of the models is completed using .csv files. You need to provide a csv file for the labeled and unlabeled data (or either of those depending on the experiments you want to run). The default folder to store the csv files is data. The csv file with labels should be called `data.csv`, and the csv file with images without labels should be called unlabeled.csv. While generating the file with labeled data, make sure to include images that are empty (label `Empty`) and images labeled as other, representing species that aren’t a member of any of your intended classes. 
 
You can also store those files in a different folder, that will however require you to define the folder while starting your scripts (explained later). The column names within the file should look as following,

![Column Names Labeled](https://github.com/slds-lmu/wildlife-experiments-server/raw/main/documentation/column_names_labeled.png)

for unlabeled data (`unlabeled.csv`), and 

![Column Names Unlabeled](https://github.com/slds-lmu/wildlife-experiments-server/raw/main/documentation/column_names_unlabeled.png)

for labeled data (`data.csv`).
 
If the columns are mislabeled it is going to result in errors while running the scripts. The order doesn’t matter, but the names of the columns must remain the same. 

#### Data Preparation

```
python data_prep.py 
	-u <True/False> [required] 
	-n <True/False> [required] 
	-d <path_to_data> [optional][default=”../data”]
```
The first element of our pipeline is data preparation. The step of data preparation involves running the images through the MegaDetector. Thus, depending on the amount of data, it can take several hours for the script to finish. 

Depending on whether you are using labeled or unlabeled data for your task, you can either prepare only the labeled data (`-n True`), only unlabeled data (`-u True`), or both. You can define it by using True/False while running the script, e.g., if you only want to run the script for unlabeled data and the csv file is located in the default folder, you would use the command: `python data_prep.py -u True -n False`. 
Depending on where you store the data, you might have to declare the location of the folder (`-d path_to_folder_with_data`). If you simply added your csv files (`data.csv` and `unlabeled.csv` or either of those) into the data folder, you can skip this step. However, if you decided to store your csv files elsewhere, you will need to declare it while setting up your data, e.g. `python data_prep.py -u True -n False -d ../data_1`

If you set `-n True`, the images included in the `data.csv` are going to be split into the training, validation, and test set. The validation and test set remain the same as you are running both passive and active pipeline to ensure the results of the training are comparable. 

If you set `-u True`, then the code runs over the images in the `unlabeled.csv`. The output of this preparation is used for the active learning to select images for further labeling and also for the final predictions on the unlabeled data in the last step (predictions). 

##### Output

If you ran the script using `-n True`, the following files should be generated into your selected data folder:
- `train.pkl`
- `val.pkl`
- `test.pkl`
- `md_labeled.json`
- `labels.csv`
- `bbox_map.json`
- `labels_map.json`
  
If you ran the script using `-u True`, the following files should be generated into your selected data folder:
- `bbox_map_unlabeled.json`
- `md_unlabeled.json`
- `unlabeled.pkl`

#### Training
```
python experiments.py 
	-a <True\False> [required] 
	-n <True\False> [required] 
	-p <path_to_model> [required] 
	-t <threshold> [float] [required when -a True, otherwise skip]
	-l <num_of_images_to_label> [int] [optional][default=128]
	-d <path_to_data> [optional][default=”../data”]
	-e <num_epochs> [int] [optional][default=100] 
	-b <backbone> [optional][default=’xception’]
	-f <True/False>[optional][default=False][whether you want to only use images ending with “a” for validation]
	-s <training_thresholds> [optional] [default=[0.1,0.3,0.5,0.7,0.9]]
```

This pipeline includes two options for training. The first option is to train the model using your labeled data only (i.e., no active learning, `-a False -n True`). 

#### Passive Learning

For example, when you have some new classes and need to change the number of species. To train the model on new classes, you need to declare the model you want to use for transfer learning (argument `-p`). Given that you prepared your labeled data in the data preparation (`data_prep.py` with `-n True`), you can now start the training using the command:
```
python experiments.py 
	-a False 
	-n True 
	-p <path_to_model>
```
if you store your csv files (only `data.csv` needed at this step) in the default data folder. Otherwise, you need to specify the folder by adding `-d <path_to_data>`. 

The model will be trained on a number of confidence thresholds for the MegaDetector, with the default thresholds being 0.1, 0.3, 0.5, 0.7, and 0.9. You can update the thresholds by utilizing the flag `-s`. You will need to state the flag separately for each flag you want to use, e.g., if you want to train using thresholds of 0.2, 0.3, and 0.4, you would need to use `python experiments.py -a False -n True -s 0.2 -s 0.3 -s 0.4 -p <path_to_model>`. You can download our pre-trained model [here](https://syncandshare.lrz.de/getlink/fiJsgDEKtkLCXfbhWM1GLR/ckpt_final_model.hdf5).

The results of the training are stored in the `training_outputs` folder. An early stopping mechanism is employed so that the training stops after the results stop progressing and the best model only is saved. After the training of each threshold is finished, you can find the results in the corresponding folder. After the training starts, a folder is created in the `training_outputs`, naming the date and time when the training has started. Then inside that folder, you can find the folders corresponding to each threshold. Depending on the amount of data you use for the training, it can take multiple days to train all the models. 

##### Output

The output of the training can be found in training_outputs/new_species. For each training a new folder is generated inside of `training_outputs/new_species` that is named based on the date and the time when the training was started. Then inside of that folder, you can find folders corresponding to each of the thresholds stated for the training. Inside of each of the folders corresponding to the thresholds, you can find the following files:

- `model.h5` - the best model corresponding to that given training
    
    Then, each of the files with the results (which are run on the validation set) correspond to the model you can find in the folder, those files are:
  - `overall_results.csv` - with results for the whole dataset regardless of the class
  - `per_species_results.csv` - with results per each species 
  - `val_pred.csv` - the results of the prediction on each image in the validation set
  - `confusion_matrix.png` - a figure representing the confusion matrix.
 
Based on the results, you can decide which model at which threshold performs best for your given task or which model you would like to use for active learning. The threshold that is at the name of the folder should then be stated using the flag `-t` while running other steps. For example, if you pick a model in folder 0.1, state `-t 0.1` while running the active learning or predictions and evaluations. 

#### Active Learning

Once you have a trained model, with the appropriate classes, if you find your results to not be satisfactory, you can start the active learning process. In the active learning, a set of images is provided for further labeling. It might be best to always specify the number of images you want to label (-l) using multiples of 128. To start the active learning process use the command:
```
python experiments.py 
	-a True 
	-n False 
	-p <path_to_model> 
	-t <threshold> 
	-l <num_of_images_to_label>
```
where `-t` corresponds to the threshold you selected with your model in the previous training step, `-l` is the number of images you want to label further (best in multiples of 128), and `-p` is the path to the model you selected for the training. The threshold should correspond to the model you selected, e.g., if you selected the model in folder corresponding to threshold 0.5, you should specify it as 0.5. Once again, if you store your csv file in a different folder than data, you will need to specify it using the flag `-d`. To run the active learning pipeline, you need to make sure that you prepared unlabeled data using the data preparation command and that the data is in the `unlabeled.csv` file.

You can find the folder with images to label in `data/active` or inside the folder you declared as the one where you store your data. Inside that folder, you will find a csv file (`active_labels.csv`) that you need to fill with the labels and the folder with the images that are selected in the file. The specific bounding box that was found by the MegaDetector is drawn on the images in that folder. In case more than one bounding box is drawn on the image, label the image with -1. You can find which numerical values correspond to the species in `data/labels_map.json`. You can continue running the command until you get results that you find satisfactory. 

The labels from the `active_labels.csv` file are added to `data.csv` after running the command and wiping away the file you just labeled. 

##### Output

The output depends on the initial state of the `data/active folder`. If the folder does not contain the `active_labels.csv` file when running the command, the output is going to be the `active_labels.csv` file and then the corresponding images in the `data/active/images folder`. 

If you start the training after you labeled the images in the `active_labels.csv` file, the training is going to start. Then, you can find the output files in `training_outputs/active`. Similar to passive training, when you run the command, a folder is generated based on the date and time when you run the command. Then, inside that folder, you can find the best model (`model.h5`) and the files corresponding to the results on the validation set using that model. Thus, all the files you can find there are:
- `model.h5` - the best model corresponding to that given training
- `overall_results.csv` - with results for the whole dataset regardless of the class
- `per_species_results.csv` - with results per each specie
- `val_pred.csv` - the results of the prediction on each image in the validation set
- `confusion_matrix.png` - a figure representing the confusion matrix.

#### Test Set Evaluation
Once you find the training results to be satisfactory, you can check the results on the test set to ensure the model is generalizable. To run your model on the test set, you can use the command:
```
python evaluate_test.py 
	-p <path_to_model>[required]
	-t <threshold>[required]
	-d <path_to_data> [optional][default=”../data”]
	-c <combined_train_val_test><True/False> [optional][default=False]
	-f <True/False>[optional][default=False][whether you want to only use images ending with “a” for validation]
```

All you need to specify is the path to the model you want to evaluate and its corresponding threshold. If your data is not stored in the default data folder, you will need to specify the folder using the `-d` flag. The results will be generated in the results folder.

If you want to evaluate the model on the whole `data.csv` file, set `-c True`. Otherwise, if you only want to use the test set, use `-c False` (which is the default setting). 

Additionally, you can filter the data such that only images ending with `a` (the first image in the series) is used. To do that, set `-f True`. That is only going to work if you number the images in the series in alphabetical order, with the letter being the last element of the image name.

##### Output
You can find the output of the command in the results folder. Here, once again the folder are generated based on the date and the time at which the command was started. Inside of the folder corresponding to the run you are looking for, you can find the following files:
- `overall_results.csv` - with results for the whole dataset regardless of the class
- `per_species_results.csv` - with results per each specie 
- `predictions.csv` - the results of the prediction on each image in the test set/the whole data.csv
- `confusion_matrix.png` - a figure representing the confusion matrix.

#### Predictions
After training and evaluating the model, you can run the predictions on your unlabeled data. That data should be a part of the `unlabeled.csv`. In case you want to add more data for the predictions, after the active learning procedure, you can simply rerun the data preparation command for the unlabeled data only. You can run the predictions using the command:
```
python predict.py 
	-p <path_to_model>[required] 
	-t <threshold> [required]
	-f <True\False> [optional][default=False]
  -d <path_to_data> [optional][default=”../data”]
```
where the `-f` states whether you want to generate folders with all the images corresponding to the predictions (`True`) or if you only want to acquire a csv file (`False`). Once again, if your data (here the images come from the `unlabeled.csv` file) is not stored in the default data folder, you will need to specify the folder using the `-d` flag. 

##### Output
Depending, on whether you set `-f` to `True` or `False`, you will find different outputs. The outputs can be found in the predictions and then in the folder corresponding to your run based on the date and time. Inside of the folder you will always find `predictions.csv`. If you set `-f True`, then you will also find folders corresponding to the species names in the dataset. Inside of those folders, you can find the images that were predicted as the given species. 

### Depth Estimation

The depth estimation pipeline is centralized in executing one script:
```bash
cd wildlife-experiments-server
python scripts/central_depth_estimation.py
```

This will execute four scripts:
1- `data_prep_depth_calibration.py`: Converting the LWF raw calibration data into the format required by the pipeline and handle irregularities in the naming convention. The raw data folder setting is explained in [Data Preperation](#data-preperation). The script will sort the data into the then created 
```bash
directories:
  clean_data_dir: "data/DistanceEstimationData"
```
folder, specified again in the `configs/depth_estimation/calibration.yaml` AND `configs/depth_estimation/estimation.yaml`.
This is not altering the raw data folder at all, it just performes copying of files.

Execution: 

`python scripts/data_prep_depth_calibration.py`


2- `seg_masks_depth_calibration.py`: This script processes camera trap calibration images for distance estimation by generating segmentation masks using the SegFormer model. The script iterates through all images in `calibration_frames` to generate masks. Masks are saved if they are successfully created and exceed the minimum pixel threshold defined by `cfg.mask_generation.min_mask_pixel_image` for all segmented areas in one masked image. Images where either no masks is created or it it too small are ignored. After mask creation, any segments within one masked image smaller than `cfg.mask_generation.min_mask_pixel_area` pixels are removed using the `remove_smaller_regions()` function.
For file management, the script renames irregular files using the `rename_files_with_bigger_mask()` function. In cases where multiple images are assigned the same distance, the script keeps the image with the larger mask. Transect directories containing fewer than two calibration images are removed.
The script uses GPU for processing, as specified by `cfg.gpu.used_gpu_idx` (outdated, LWF has only one GPU on the server), and stores the created masks in the `self.directories.clean_data_dir`, specifically within the `calibration_frames_masks` subdirectory for each transect.

Execution: 

`python scripts/seg_masks_depth_calibration.py`

3- `copy_images_depth_estimation.py`: This script copies images of animals into the `detection_frames` folders of the `self.directories.clean_data_dir`. It first deletes any existing detection images in the target directory, and then copies images from the source directory specified in `self.directories.raw_trap_data_dir`. The source directory is expected to contain transect folders with images. The `self.directories.raw_trap_data_dir` itself is not modified during this process.

Execution: 

`python scripts/copy_images_depth_estimation.py`

4- `distance-estimation/run_estimation.py`: This script performs depth estimation for wildlife detection using calibration and detection frames. It utilizes the DPT model for depth prediction, MegaDetector for animal detection, and SAM for more precise depth estimation.

Process Overview:
The script first loads configuration settings and verifies the directory structure. It processes each transect by loading calibration frames, performing depth calibration using regression methods (e.g., RANSAC), and applying depth estimation techniques. It detects animals using the MegaDetector, filters out non-animal detections, and computes depth based on various sampling methods (e.g., BBOX Bottom, BBOX Percentiles, SAM). The world position of each detected animal is calculated using camera geometry.

Results, including depth and world position, are logged in CSV and text files. If enabled, visualizations of the detection results and world positions are also generated. Finally, the script cleans up and moves the results to the specified directory.

Output:
- **CSV**: Logs detection details, including depth and world position.
- **Text File**: Logs radial distances for each detection.
- **Visualization**: If configured, visualizes detections and world positions.

Execution: 

`python scripts/distance-estimation/run_estimation.py`

#### Running multiple configurations

Given the lists of the following three parameters of `estimation.yaml`, an automation script of the same depth estimation pipeline to run all possible combinations of these three parameters is also provided.
```
configs/depth_estimation/estimation.yaml
    -general.max_depth
    -detection.bbox_confidence_threshold
    -sampling.detection_sampling_method
```

This automation script can be run with:

```bash
cd wildlife-experiments-server
python central_depth_estimation.py
```

The following arguments in `central_depth_estimation.py` each overrides the corresponding parameter in `configs/depth_estimation/estimation.yaml` during the estimation step:
- **MAX_DEPTHS**: List of depth values (float) for `general.max_depth` parameter.
- **CONFIDENCES**: List of confidence values (float) for `detection.bbox_confidence_threshold` parameter. Can contain values only between `0` and `1`.
- **METHODS**: List of detection sampling methods (string) for `sampling.detection_sampling_method` parameter. Can contain only the following methods, namely `"BBOX_BOTTOM"`, `"BBOX_PERCENTILE"`, and `"SAM"`.

*Note: The Calibration step is run only once, but the Estimation step is run once for every possible parameter combination.*


## Other Info
- Link to our pre-trained model on which you can start the passive training: [download here](https://syncandshare.lrz.de/getlink/fiJsgDEKtkLCXfbhWM1GLR/ckpt_final_model.hdf5).
- Paper on which the repository is based: [Bothmann et al. (2023)](https://linkinghub.elsevier.com/retrieve/pii/S1574954123002601).
- This repository utilizes a wildlife-ml package you can find in this [repository](https://github.com/slds-lmu/wildlife-ml).
- Original repository from Timm Haucke for depth estimation: [timmh/distance-estimation](https://github.com/timmh/distance-estimation/tree/main)

## Citation
If you use this repository, please consider citing our corresponding paper:
```
@article{bothmann_automated_2023,
	title = {Automated wildlife image classification: {An} active learning tool for ecological applications},
	volume = {77},
	issn = {15749541},
	url = {https://linkinghub.elsevier.com/retrieve/pii/S1574954123002601},
	doi = {10.1016/j.ecoinf.2023.102231},
	language = {en},
	urldate = {2023-08-30},
	journal = {Ecological Informatics},
	author = {Bothmann, Ludwig and Wimmer, Lisa and Charrakh, Omid and Weber, Tobias and Edelhoff, Hendrik and Peters, Wibke and Nguyen, Hien and Benjamin, Caryl and Menzel, Annette},
	year = {2023},
	pages = {102231},
}
```

For the depth estimation, please consider citing the paper from Timm Hacke et al.:
@article{Haucke_2022,
   title={Overcoming the distance estimation bottleneck in estimating animal abundance with camera traps},
   volume={68},
   ISSN={1574-9541},
   url={http://dx.doi.org/10.1016/j.ecoinf.2021.101536},
   DOI={10.1016/j.ecoinf.2021.101536},
   journal={Ecological Informatics},
   publisher={Elsevier BV},
   author={Haucke, Timm and Kühl, Hjalmar S. and Hoyer, Jacqueline and Steinhage, Volker},
   year={2022},
   month=may, pages={101536}}







