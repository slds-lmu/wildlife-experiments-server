# wildlife-experiments-server

## Initial Setup by User

Overview – see below for details:

User has to set up the following before they can get started:
- `data.csv`: This file contains file names and classes of already labeled images, using the following column names: ![Column Names Labeled](https://github.com/slds-lmu/wildlife-experiments-server/raw/main/documentation/column_names_labeled.png)
  - Make sure to include images labeled as `Empty` in the file! The file should include images with animals as well as empty ones to ensure the best performance of the pipeline.
- `unlabeled.csv`: This file contains file names of unlabeled images using the following column names: ![Column Names Unlabeled](https://github.com/slds-lmu/wildlife-experiments-server/raw/main/documentation/column_names_unlabeled.png)

The main difference between the setup of those two files is that the labeled file contains the column `Species_Class`. 

You can either put these in the folder `data`, which would be the default. Or put it wherever you want. This means you can work on multiple projects in parallel. However, if you decide to put the files into your selected folder, you will need to remember to always declare the folder using flag `-d` while running the command (more details on how to do it in the next parts). If you decide to store your data in a separate folder, make sure that the `data.csv` and `unlabeled.csv` corresponding to one project are always stored together and not in separate folders! 

## Environments 

We utilized two different environments for our scripts. One environment based on PyTorch was used for the data preparation step (`data_prep.py`), where MegaDetector is utilized. You can find the requirements for this environment in `requirements_torch.txt`.

The other environment is based on TensorFlow and is used for all the other scripts in this repository. You can find its requirements in `requirements_tf.txt`.

## Data Setup

The training of the models is completed using .csv files. You need to provide a csv file for the labeled and unlabeled data (or either of those depending on the experiments you want to run). The default folder to store the csv files is data. The csv file with labels should be called `data.csv`, and the csv file with images without labels should be called unlabeled.csv. While generating the file with labeled data, make sure to include images that are empty (label `Empty`) and images labeled as other, representing species that aren’t a member of any of your intended classes. 
 
You can also store those files in a different folder, that will however require you to define the folder while starting your scripts (explained later). The column names within the file should look as following,

![Column Names Labeled](https://github.com/slds-lmu/wildlife-experiments-server/raw/main/documentation/column_names_labeled.png)

for unlabeled data (`unlabeled.csv`), and 

![Column Names Unlabeled](https://github.com/slds-lmu/wildlife-experiments-server/raw/main/documentation/column_names_unlabeled.png)

for labeled data (`data.csv`).
 
If the columns are mislabeled it is going to result in errors while running the scripts. The order doesn’t matter, but the names of the columns must remain the same. 

## Data Preparation

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

#### Output

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

## Training
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

### Passive Learning

For example, when you have some new classes and need to change the number of species. To train the model on new classes, you need to declare the model you want to use for transfer learning (argument `-p`). Given that you prepared your labeled data in the data preparation (`data_prep.py` with `-n True`), you can now start the training using the command:
```
python experiments.py 
	-a False 
	-n True 
	-p <path_to_model>
```
if you store your csv files (only `data.csv` needed at this step) in the default data folder. Otherwise, you need to specify the folder by adding `-d <path_to_data>`. 

The model will be trained on a number of confidence thresholds for the MegaDetector, with the default thresholds being 0.1, 0.3, 0.5, 0.7, and 0.9. You can update the thresholds by utilizing the flag `-s`. You will need to state the flag separately for each flag you want to use, e.g., if you want to train using thresholds of 0.2, 0.3, and 0.4, you would need to use `python experiments.py -a False -n True -s 0.2 -s 0.3 -s 0.4 -p <path_to_model>`.

The results of the training are stored in the `training_outputs` folder. An early stopping mechanism is employed so that the training stops after the results stop progressing and the best model only is saved. After the training of each threshold is finished, you can find the results in the corresponding folder. After the training starts, a folder is created in the `training_outputs`, naming the date and time when the training has started. Then inside that folder, you can find the folders corresponding to each threshold. Depending on the amount of data you use for the training, it can take multiple days to train all the models. 

#### Output

The output of the training can be found in training_outputs/new_species. For each training a new folder is generated inside of `training_outputs/new_species` that is named based on the date and the time when the training was started. Then inside of that folder, you can find folders corresponding to each of the thresholds stated for the training. Inside of each of the folders corresponding to the thresholds, you can find the following files:

- `model.h5` - the best model corresponding to that given training
    
    Then, each of the files with the results (which are run on the validation set) correspond to the model you can find in the folder, those files are:
  - `overall_results.csv` - with results for the whole dataset regardless of the class
  - `per_species_results.csv` - with results per each species 
  - `val_pred.csv` - the results of the prediction on each image in the validation set
  - `confusion_matrix.png` - a figure representing the confusion matrix.
 
Based on the results, you can decide which model at which threshold performs best for your given task or which model you would like to use for active learning. The threshold that is at the name of the folder should then be stated using the flag `-t` while running other steps. For example, if you pick a model in folder 0.1, state `-t 0.1` while running the active learning or predictions and evaluations. 

### Active Learning

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

#### Output

The output depends on the initial state of the `data/active folder`. If the folder does not contain the `active_labels.csv` file when running the command, the output is going to be the `active_labels.csv` file and then the corresponding images in the `data/active/images folder`. 

If you start the training after you labeled the images in the `active_labels.csv` file, the training is going to start. Then, you can find the output files in `training_outputs/active`. Similar to passive training, when you run the command, a folder is generated based on the date and time when you run the command. Then, inside that folder, you can find the best model (`model.h5`) and the files corresponding to the results on the validation set using that model. Thus, all the files you can find there are:
- `model.h5` - the best model corresponding to that given training
- `overall_results.csv` - with results for the whole dataset regardless of the class
- `per_species_results.csv` - with results per each specie
- `val_pred.csv` - the results of the prediction on each image in the validation set
- `confusion_matrix.png` - a figure representing the confusion matrix.

## Test Set Evaluation
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

#### Output
You can find the output of the command in the results folder. Here, once again the folder are generated based on the date and the time at which the command was started. Inside of the folder corresponding to the run you are looking for, you can find the following files:
- `overall_results.csv` - with results for the whole dataset regardless of the class
- `per_species_results.csv` - with results per each specie 
- `predictions.csv` - the results of the prediction on each image in the test set/the whole data.csv
- `confusion_matrix.png` - a figure representing the confusion matrix.

## Predictions
After training and evaluating the model, you can run the predictions on your unlabeled data. That data should be a part of the `unlabeled.csv`. In case you want to add more data for the predictions, after the active learning procedure, you can simply rerun the data preparation command for the unlabeled data only. You can run the predictions using the command:
```
python predict.py 
	-p <path_to_model>[required] 
	-t <threshold> [required]
	-f <True\False> [optional][default=False]
  -d <path_to_data> [optional][default=”../data”]
```
where the `-f` states whether you want to generate folders with all the images corresponding to the predictions (`True`) or if you only want to acquire a csv file (`False`). Once again, if your data (here the images come from the `unlabeled.csv` file) is not stored in the default data folder, you will need to specify the folder using the `-d` flag. 

#### Output
Depending, on whether you set `-f` to `True` or `False`, you will find different outputs. The outputs can be found in the predictions and then in the folder corresponding to your run based on the date and time. Inside of the folder you will always find `predictions.csv`. If you set `-f True`, then you will also find folders corresponding to the species names in the dataset. Inside of those folders, you can find the images that were predicted as the given species. 










