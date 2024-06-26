"""Use final model to predict on new data."""
import os
from typing import Final, Dict, List
import albumentations as A
import click
import json
import shutil

from tensorflow.keras.models import load_model
from wildlifeml import WildlifeDataset
from wildlifeml.data import BBoxMapper
from wildlifeml.training.evaluator import Evaluator
from wildlifeml.utils.io import load_json, load_pickle
from wildlifeml.utils.misc import flatten_list
from keras.applications import imagenet_utils
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wildlifeml.data import subset_dataset

@click.command()
@click.option(
    '--model_path', '-p', help='The path to the model you want to use for predictions.', required=True, type=str
)
@click.option(
    '--thresh', '-t', help='The threshold associated with your model of choice.', required=True, type=float
)
@click.option(
    '--filter_first', '-f', help='Whether you only want to use the first image in series.', required=False, type=bool, default=False
)
@click.option(
    '--data_folder', '-d', help='Folder where you store your data csv files.', required=False, type=str, default="../data"
)
@click.option(
    '--combined', '-c', help='Whether you want to run evaluation on train,test,and validation sets combined.', required=False, type=bool, default=False
)

def main(model_path: str, thresh: float, filter_first: bool, data_folder: str, combined: bool):
    results_folder = "../results"

    model = load_model(model_path, custom_objects={'imagenet_utils': imagenet_utils})
    test_dataset = load_pickle(os.path.join(data_folder, "test.pkl"))
    if combined:
        val_dataset = load_pickle(os.path.join(data_folder, "val.pkl"))
        train_dataset = load_pickle(os.path.join(data_folder, "train.pkl"))
        val_keys = val_dataset.keys
        train_keys = train_dataset.keys
        test_keys = test_dataset.keys
        all_keys = val_keys + test_keys + train_keys
        test_dataset = subset_dataset(test_dataset, all_keys)

    if filter_first:
        all_keys = test_dataset.keys
        filtered_keys = []
        for k in all_keys:
            file_name = k.split(".JPG")[0]
            if file_name[-1] == 'a':
                filtered_keys.append(k)
        test_dataset = subset_dataset(test_dataset, filtered_keys)
    labels_map = load_json(os.path.join(data_folder, "labels_map.json"))
    evaluator = Evaluator(
        detector_file_path=os.path.join(data_folder, "md_labeled.json"),
        num_classes=len(labels_map.keys()),
        empty_class_id=labels_map.get("Empty") or 0,
        conf_threshold=thresh,
        dataset=test_dataset,
        label_file_path=os.path.join(data_folder, "labels.csv")
    )
    evaluator.evaluate(model)

    # folder for current predictions
    current_datetime = datetime.now()
    folder_name = os.path.join(results_folder, current_datetime.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(folder_name, exist_ok=True)
    evaluator.save_predictions(filepath=os.path.join(folder_name, 'predictions.csv'), img_level=True)

    # evaluate the predictions
    # add ground truth labels to the prediction file 
    eval_df = pd.read_csv(os.path.join(folder_name, 'predictions.csv'))
    lbl_df = pd.read_csv(os.path.join(data_folder, "labels.csv"), header=None)
    lbl_df.columns = ['img_key', 'GT_label']
    gt_lbl = dict(zip(lbl_df['img_key'], lbl_df['GT_label']))
    eval_df['gt_label'] = eval_df['img_key'].map(gt_lbl)
    eval_df.to_csv(os.path.join(folder_name, 'predictions.csv'), index=False)

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
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",xticklabels=species_names, yticklabels=species_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better fit
    plt.yticks(rotation=45)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(folder_name, 'confusion_matrix.png'))

    # per class info
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
        precision.append(precision_score(y_true=y_true, y_pred=y_pred,labels=all_labels, average=None)[label])
        recall.append(recall_score(y_true=y_true, y_pred=y_pred,labels=all_labels, average=None)[label])

    per_species_df = pd.DataFrame()
    per_species_df["species"] = species
    per_species_df["accuracy"] = accuracy
    per_species_df["f1"] = f1
    per_species_df["precision"] = precision
    per_species_df["recall"] = recall
    per_species_df.to_csv(os.path.join(folder_name, "per_species_results.csv"), index=False)
    
if __name__ == '__main__':
    main()
