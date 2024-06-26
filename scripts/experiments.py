import time

import click
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import gc
from typing import Dict, Final, List
from wildlifeml.data import subset_dataset
from wildlifeml.training.trainer import WildlifeTrainer
from wildlifeml.training.active import ActiveLearner
from wildlifeml.training.evaluator import Evaluator
from wildlifeml.utils.datasets import separate_empties, map_bbox_to_img
from wildlifeml.utils.io import (
    load_csv,
    load_json,
    load_pickle,
    save_as_csv,
    save_as_pickle,
    load_csv_dict
)
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from datetime import datetime
from keras.applications import imagenet_utils
from sklearn.model_selection import train_test_split
from wildlifeml.utils.misc import flatten_list
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils_ours import seed_everything, MyEarlyStopping

TIMESTR: Final[str] = time.strftime("%Y%m%d%H%M")
THRESH_TUNED: Final[float] = 0.1  # 0.5
THRESH_PROGRESSIVE: Final[float] = 0.5
THRESH_NOROUZZADEH: Final[float] = 0.9
BACKBONE_TUNED: Final[str] = 'xception'
FTLAYERS_TUNED: Final[int] = 0

@click.command()
@click.option(
    '--active',
    '-a',
    help='Do you want to perform active traing?',
    required=True,
    type=bool
)
@click.option(
    '--train_new_species',
    '-n',
    help='Do you want to train model with new classes/species?',
    required=True,
    type=bool
)
@click.option(
    '--pretraining_model',
    '-p',
    help='Define the path to the model to be used for pretraining.',
    required=True,
    type=str
)
@click.option(
    '--thresh',
    '-t',
    help='Threshold from the model selected for active learning.',
    required=False,
    default=0.1,
    type=float
)
@click.option(
    '--num_label',
    '-l',
    help='Number of images you want to label.',
    required=False,
    default=128,
    type=int
)
@click.option(
    '--data_folder',
    '-d',
    help='Folder where you store the csv files.',
    required=False,
    type=str,
    default="../data"
)

@click.option(
    '--num_epochs',
    '-e',
    help='Number of epochs for training.',
    required=False,
    type=int,
    default=100
)
@click.option(
    '--backbone',
    '-b',
    help='The backbone of the pretrained model.',
    required=False,
    type=str,
    default='xception'
)
@click.option(
    '--filter_first',
    '-f',
    help='Whether you want to run validation on "a" images only.',
    required=False,
    type=bool,
    default=False
)
@click.option(
    '--thresholds',
    '-s',
    help='Thresholds that you want to use for training.',
    required=False,
    multiple=True,
    default=[0.1, 0.3, 0.5, 0.7, 0.8]
)

def main(active, train_new_species, pretraining_model, thresh, num_label, data_folder, num_epochs, backbone, filter_first, thresholds):
    # read the config file
    cfg: Final[Dict] = load_json(os.path.join("..", 'configs/cfg.json'))
    seed_everything(0)

    try:
        labels_map = load_json(os.path.join(data_folder, 'labels_map.json'))
    except:
        print("Unable to finds labels_map.json file. Make sure you ran the data_prep.py file first.")

    trainer_args: Final[Dict] = {
        'batch_size': 128,
        'loss_func': keras.losses.SparseCategoricalCrossentropy(),
        'num_classes': len(labels_map.keys()), # classes should be a matter of reading the label_map file
        'transfer_epochs': num_epochs,
        'finetune_epochs': 0,
        'finetune_layers': FTLAYERS_TUNED,
        'model_backbone': backbone,
        'num_workers': cfg['num_workers'],
        'eval_metrics': cfg['eval_metrics'],
     }

    if train_new_species:
        # create output folder
        outputs_folder = "../training_outputs/new_species"
        # make folder to store the outputs of the training
        current_datetime = datetime.now()
        folder_name = os.path.join(outputs_folder, current_datetime.strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(folder_name, exist_ok=True)

        dataset_train = load_pickle(os.path.join(data_folder, 'train.pkl'))
        dataset_val = load_pickle(os.path.join(data_folder, 'val.pkl'))

        if filter_first:
            val_keys = dataset_val.keys
            filtered_keys = []
            for k in val_keys:
                file_name = k.split(".JPG")[0]
                if file_name[-1] == 'a':
                    filtered_keys.append(k)
            dataset_val = subset_dataset(dataset_val, filtered_keys)

        for ds in [dataset_val]:
            ds.shuffle = False
            ds.augmentation = None

        # declare thresholds
        # thresholds = np.arange(0.1, 1, 0.2).round(2).tolist()

        for threshold in thresholds:
            # make folder for the threshold
            current_folder = os.path.join(folder_name, str(threshold))
            os.makedirs(current_folder, exist_ok=True)

            _, keys_all_nonempty = separate_empties(
                os.path.join(data_folder, 'md_labeled.json'),
                float(threshold)
            )
            keys_is_train = list(
                set(dataset_train.keys).intersection(set(keys_all_nonempty))
            )
            keys_is_val = list(
                set(dataset_val.keys).intersection(set(keys_all_nonempty))
            )

            dataset_train_thresh = subset_dataset(dataset_train, keys_is_train)
            dataset_val_thresh = subset_dataset(dataset_val, keys_is_val)

            dataset_val_thresh.shuffle = False
            dataset_val_thresh.augmentation = None

            transfer_callbacks = [
                MyEarlyStopping(
                    monitor='val_loss',
                    mode='min',
                    patience=3,
                    start_from_epoch=1,
                    min_delta=0.02
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    patience=cfg['transfer_patience'],
                    factor=0.1
                ),
                ModelCheckpoint(
                    filepath=os.path.join(current_folder, "model.h5"),
                    save_best_only=True,
                    save_weights_only=False,
                    monitor='val_loss',
                    mode='auto',
                    save_freq='epoch'
                )
            ]

            this_trainer_args: Dict = dict(
                {
                    'transfer_callbacks': transfer_callbacks,
                    'transfer_optimizer': Adam(cfg['transfer_learning_rate']),
                    'finetune_optimizer': Adam(cfg['finetune_learning_rate'])
                },
                **trainer_args
            )

            this_trainer_args.update(
                {
                    'pretraining_checkpoint': pretraining_model,
                    'change_class_num': True
                }
            )

            trainer = WildlifeTrainer(**this_trainer_args)

            print('---> Training on wildlife data')
            trainer.fit(
                train_dataset=dataset_train_thresh, val_dataset=dataset_val_thresh
            )

            print('--> Evaluating the best model on validation data')

            empty_class_id = load_json(os.path.join(data_folder, 'labels_map.json')).get('Empty') or 0

            model = load_model(os.path.join(current_folder, "model.h5"), custom_objects={'imagenet_utils': imagenet_utils})

            evaluator = Evaluator(
                detector_file_path=os.path.join(data_folder, "md_labeled.json"),
                num_classes=len(labels_map.keys()),
                empty_class_id=empty_class_id,
                conf_threshold=threshold,
                dataset=dataset_val,
                label_file_path=os.path.join(data_folder, "labels.csv")
            )
            evaluator.evaluate(model)
            evaluator.save_predictions(filepath=os.path.join(current_folder, 'val_pred.csv'), img_level=True)
            
            # add ground truth labels to the prediction file 
            eval_df = pd.read_csv(os.path.join(current_folder, 'val_pred.csv'))
            lbl_df = pd.read_csv(os.path.join(data_folder, "labels.csv"), header=None)
            lbl_df.columns = ['img_key', 'GT_label']
            gt_lbl = dict(zip(lbl_df['img_key'], lbl_df['GT_label']))
            eval_df['gt_label'] = eval_df['img_key'].map(gt_lbl)
            eval_df.to_csv(os.path.join(current_folder, 'val_pred.csv'), index=False)

            # perform the evaluation and save it 
            # without regard to class
            y_true = eval_df['gt_label'].to_numpy()
            y_pred = eval_df['hard_label'].to_numpy()

            whole_data = {
                "accuracy" : [accuracy_score(y_true=y_true, y_pred=y_pred)],
                "f1" : [f1_score(y_true=y_true, y_pred=y_pred, average='macro')],
                "precision" : [precision_score(y_true=y_true, y_pred=y_pred, average='macro')],
                "recall" : [recall_score(y_true=y_true, y_pred=y_pred, average='macro')],
            }

            whole_data_df = pd.DataFrame(whole_data)
            whole_data_df.to_csv(os.path.join(current_folder, "overall_results.csv"), index=False)

            label_map = load_json(os.path.join(data_folder, "labels_map.json"))
            all_labels = list(range(len(label_map.keys())))
            species_names = [list(label_map.keys())[label] for label in all_labels]

            # confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred, labels=all_labels)
            plt.figure(figsize=(18, 15))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=species_names,
                        yticklabels=species_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=45)  # Rotate x-axis labels for better fit
            plt.yticks(rotation=45)
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(current_folder, 'confusion_matrix.png'))

            # per class info
            species = []
            accuracy = []
            f1 = []
            precision = []
            recall = []

            for label in range(0, len(label_map.keys())):
                filtered_df = eval_df[(eval_df['gt_label'] == label) | (eval_df['hard_label'] == label)]
                y_true = filtered_df['gt_label'].to_numpy()
                y_pred = filtered_df['hard_label'].to_numpy()

                # Compute accuracy for the current class
                species.append(list(label_map.keys())[label])
                accuracy.append(accuracy_score(y_true=y_true, y_pred=y_pred))
                f1.append(f1_score(y_true=y_true, y_pred=y_pred, labels=all_labels, average=None)[label])
                precision.append(precision_score(y_true=y_true, y_pred=y_pred, labels=all_labels, average=None)[label])
                recall.append(recall_score(y_true=y_true, y_pred=y_pred, labels=all_labels, average=None)[label])

            per_species_df = pd.DataFrame()
            per_species_df["species"] = species
            per_species_df["accuracy"] = accuracy
            per_species_df["f1"] = f1
            per_species_df["precision"] = precision
            per_species_df["recall"] = recall
            per_species_df.to_csv(os.path.join(current_folder, "per_species_results.csv"), index=False)
        
    if active:
        active_folder = os.path.join(data_folder, "active")
        new_labels = True
        try:
            active_df = pd.read_csv(os.path.join(active_folder, "active_labels.csv"))
        except:
            print("No active labels file found. Looking for images to label.")
            new_labels = False
        outputs_folder = "../training_outputs/active"
        if new_labels:
            print("Training using new labels...")
            data_unlabeled = pd.read_csv(os.path.join(data_folder, "unlabeled.csv"))
            dataset_unlabeled = load_pickle(os.path.join(data_folder, 'unlabeled.pkl'))
            # pick unlabeled data to use for prediction
            if len(data_unlabeled) > 10000:
                percent_to_keep = 10000/len(data_unlabeled)
                key_map = load_json(os.path.join(data_folder, "bbox_map_unlabeled.json"))
                _, data_unlabeled = train_test_split(data_unlabeled, test_size=percent_to_keep, stratify=data_unlabeled['Station'])
                print(len(data_unlabeled))
                data_unlabeled = flatten_list([key_map[k] for k in data_unlabeled['FilePath_new'].to_numpy()])
                dataset_unlabeled = subset_dataset(dataset_unlabeled, data_unlabeled)

            dataset_train = load_pickle(os.path.join(data_folder, 'train.pkl'))
            dataset_val = load_pickle(os.path.join(data_folder, 'val.pkl'))
            _, keys_label_nonempty = separate_empties(
                os.path.join(data_folder, "md_labeled.json"), float(thresh)
            )
            _, keys_unlabeled_nonempty = separate_empties(
                os.path.join(data_folder, "md_unlabeled.json"), float(thresh)
            )
            keys_train = list(set(dataset_train.keys).intersection(keys_label_nonempty))
            keys_val = list(set(dataset_val.keys).intersection(keys_label_nonempty))
            keys_unlabeled = list(set(dataset_unlabeled.keys).intersection(keys_unlabeled_nonempty))
            dataset_train = subset_dataset(dataset_train, keys_train)
            dataset_val = subset_dataset(dataset_val, keys_val)

            if filter_first:
                val_keys = dataset_val.keys
                filtered_keys = []
                for k in val_keys:
                    file_name = k.split(".JPG")[0]
                    if file_name[-1] == 'a':
                        filtered_keys.append(k)
                dataset_val = subset_dataset(dataset_val, filtered_keys)

            dataset_unlabeled = subset_dataset(dataset_unlabeled, keys_unlabeled)
            # make folder to store the outputs of the training
            current_datetime = datetime.now()
            folder_name = os.path.join(outputs_folder, current_datetime.strftime("%Y-%m-%d_%H-%M-%S"))
            os.makedirs(folder_name, exist_ok=True)
            empty_class_id = load_json(os.path.join(data_folder, "labels_map.json")).get("Empty") or 0

            transfer_callbacks = [
                MyEarlyStopping(
                    monitor='val_loss',
                    mode='min',
                    patience=5,
                    start_from_epoch=0,
                    min_delta=0.02
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    patience=cfg['transfer_patience'],
                    factor=0.1
                ),
                ModelCheckpoint(
                    filepath=os.path.join(folder_name, "model.h5"),
                    save_best_only=True,
                    save_weights_only=False,
                    monitor='val_loss',
                    mode='auto',
                    save_freq='epoch'
                )
            ]
            this_trainer_args: Dict = dict(
                {
                    'transfer_callbacks': transfer_callbacks,
                    'transfer_optimizer': Adam(cfg['transfer_learning_rate']),
                    'finetune_optimizer': Adam(cfg['finetune_learning_rate'])
                },
                **trainer_args
            )

            this_trainer_args.update(
                {
                    'pretraining_checkpoint': pretraining_model,
                }
            )

            active_learner = ActiveLearner(
                trainer=WildlifeTrainer(**this_trainer_args),
                pool_dataset=dataset_train,
                val_dataset=dataset_val,
                unlabeled_dataset=dataset_unlabeled,
                label_file_path=os.path.join(data_folder, "labels.csv"),
                conf_threshold=thresh,
                empty_class_id=empty_class_id,
                al_batch_size=num_label,
                start_fresh=False,
                active_directory=active_folder,
                dir_data=data_folder
            )
            active_learner.run()

            if active_learner.missing_labels:
                return

            print('--> Evaluating the model on validation data')

            empty_class_id = load_json(os.path.join(data_folder, 'labels_map.json')).get('Empty') or 0

            dataset_val.shuffle = False
            dataset_val.augmentation = None

            model = load_model(os.path.join(folder_name, "model.h5"), custom_objects={'imagenet_utils': imagenet_utils})

            evaluator = Evaluator(
                detector_file_path=os.path.join(data_folder, "md_labeled.json"),
                num_classes=len(labels_map.keys()),
                empty_class_id=empty_class_id,
                conf_threshold=thresh,
                dataset=dataset_val,
                label_file_path=os.path.join(data_folder, "labels.csv")
            )
            evaluator.evaluate(model)
            evaluator.save_predictions(filepath=os.path.join(folder_name, 'val_pred.csv'), img_level=True)
            
            # add ground truth labels to the prediction file 
            eval_df = pd.read_csv(os.path.join(folder_name, 'val_pred.csv'))
            lbl_df = pd.read_csv(os.path.join(data_folder, "labels.csv"), header=None)
            lbl_df.columns = ['img_key', 'GT_label']
            gt_lbl = dict(zip(lbl_df['img_key'], lbl_df['GT_label']))
            eval_df['gt_label'] = eval_df['img_key'].map(gt_lbl)
            eval_df.to_csv(os.path.join(folder_name, 'val_pred.csv'), index=False)

            # perform the evaluation and save it 
            # without regard to class
            y_true = eval_df['gt_label'].to_numpy()
            y_pred = eval_df['hard_label'].to_numpy()

            whole_data = {
                "accuracy" : [accuracy_score(y_true=y_true, y_pred=y_pred)],
                "f1" : [f1_score(y_true=y_true, y_pred=y_pred, average='macro')],
                "precision" : [precision_score(y_true=y_true, y_pred=y_pred, average='macro')],
                "recall" : [recall_score(y_true=y_true, y_pred=y_pred, average='macro')],
            }

            whole_data_df = pd.DataFrame(whole_data)
            whole_data_df.to_csv(os.path.join(folder_name, "overall_results.csv"), index=False)

            label_map = load_json(os.path.join(data_folder, "labels_map.json"))
            all_labels = list(range(len(label_map.keys())))
            species_names = [list(label_map.keys())[label] for label in all_labels]

            # confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred, labels=all_labels)
            plt.figure(figsize=(18, 15))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=species_names,
                        yticklabels=species_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=45)  # Rotate x-axis labels for better fit
            plt.yticks(rotation=45)
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(folder_name, 'confusion_matrix.png'))

            # per class info
            species = []
            accuracy = []
            f1 = []
            precision = []
            recall = []

            for label in range(0, len(label_map.keys())):
                filtered_df = eval_df[(eval_df['gt_label'] == label) | (eval_df['hard_label'] == label)]
                y_true = filtered_df['gt_label'].to_numpy()
                y_pred = filtered_df['hard_label'].to_numpy()

                # Compute accuracy for the current class
                species.append(list(label_map.keys())[label])
                accuracy.append(accuracy_score(y_true=y_true, y_pred=y_pred))
                f1.append(f1_score(y_true=y_true, y_pred=y_pred, labels=all_labels, average=None)[label])
                precision.append(precision_score(y_true=y_true, y_pred=y_pred, labels=all_labels, average=None)[label])
                recall.append(recall_score(y_true=y_true, y_pred=y_pred, labels=all_labels, average=None)[label])

            per_species_df = pd.DataFrame()
            per_species_df["species"] = species
            per_species_df["accuracy"] = accuracy
            per_species_df["f1"] = f1
            per_species_df["precision"] = precision
            per_species_df["recall"] = recall
            per_species_df.to_csv(os.path.join(folder_name, "per_species_results.csv"), index=False)
        else:
            print("Looking for images to label.")
            dataset_unlabeled = load_pickle(os.path.join(data_folder, "unlabeled.pkl"))
            if len(dataset_unlabeled.keys) > 10000:
                percent_to_keep = 10000/len(dataset_unlabeled.keys)
                _, data_unlabeled = train_test_split(dataset_unlabeled.keys, test_size=percent_to_keep)
                dataset_unlabeled = subset_dataset(dataset_unlabeled, data_unlabeled)
            dataset_unlabeled.shuffle = False
            dataset_unlabeled.augmentation = None 
            empty_class_id = load_json(os.path.join(data_folder, "labels_map.json")).get("Empty") or 0
            # find the non empty images
            _, keys_all_nonempty = separate_empties(
                os.path.join(data_folder, "md_unlabeled.json"),
                float(thresh)
            )
            keys_unlabeled = list(
                set(dataset_unlabeled.keys).intersection(set(keys_all_nonempty))
            )
            dataset_unlabeled = subset_dataset(dataset_unlabeled, keys_unlabeled)

            this_trainer_args: Dict = dict(
                {
                    'transfer_optimizer': Adam(cfg['transfer_learning_rate']),
                    'finetune_optimizer': Adam(cfg['finetune_learning_rate'])
                },
                **trainer_args
            )
            this_trainer_args.update(
                {
                    'pretraining_checkpoint': pretraining_model,
                }
            )
            dataset_train = load_pickle(os.path.join(data_folder, 'train.pkl'))
            dataset_val = load_pickle(os.path.join(data_folder, 'val.pkl'))

            active_learner = ActiveLearner(
                trainer=WildlifeTrainer(**this_trainer_args),
                pool_dataset=dataset_train,
                unlabeled_dataset=dataset_unlabeled,
                val_dataset=dataset_val,
                label_file_path=os.path.join(data_folder, "labels.csv"),
                conf_threshold=thresh,
                empty_class_id=empty_class_id,
                al_batch_size=num_label,
                start_fresh=False,
                active_directory=active_folder,
                dir_data=data_folder
            )
            active_learner.get_csv_to_label()

    print("Training finished successfully. ")

if __name__ == '__main__':
    main()

