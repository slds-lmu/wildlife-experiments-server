import numpy as np
import pandas as pd
from wildlifeml.preprocessing.megadetector import MegaDetector
from sklearn.model_selection import train_test_split
from wildlifeml.data import BBoxMapper, WildlifeDataset, subset_dataset
import albumentations as A
from wildlifeml.utils.io import (
    load_csv_dict,
    save_as_csv,
    load_json,
    save_as_json,
    save_as_pickle,
    load_image
)
import os
from wildlifeml.utils.misc import flatten_list
import json
import click
from tqdm import tqdm

@click.command()
@click.option(
    '--prep_unlabeled',
    '-u',
    help='Do you want to prepare unlabeled data?',
    required=True,
    type=bool
)
@click.option(
    '--new_data',
    '-n',
    help='Do you want to prepare new splits for labeled data?',
    required=True,
    type=bool
)
@click.option(
    '--data_folder',
    '-d',
    help='Define the folder where your data is stored',
    required=True,
    type=str,
    default='../data'
)

def main(prep_unlabeled, new_data, data_folder):
    os.makedirs(os.path.join(data_folder, "active"), exist_ok=True)
    augmentation = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ]
    )

    if not os.path.exists(data_folder):
        print("data folder doesn\'t exist, can\'t prepare the data")
        return
    
    if new_data:
        # prep train, val, test
        # read the csv file with labeled data
        try:
            data_df = pd.read_csv(os.path.join(data_folder, "data.csv"))
        except:
            print("data.csv not found in data folder. Add the file to the folder before starting the script.")
            return
        # clean the csv file
        print("Cleaning the csv file to ensure correctness.")
        print("Initial size of the file: ", len(data_df), " rows.")
        data_df = data_df.dropna(subset=["Species_Class"]) # remove NaN values from labels
        print("Size of the file after cleaning: ", len(data_df), " rows.")
        print("Preparing the data...")
        # prepare labels_map
        lbls = data_df['Species_Class'].to_numpy().astype(str)
        lbls = np.unique(lbls)
        lbls_dict = {}
        for i, lbl in enumerate(lbls):
            lbls_dict[lbl] = i
        # Path to save the JSON file
        json_file_path = os.path.join(data_folder, "labels_map.json")
        with open(json_file_path, "w") as json_file:
            json.dump(lbls_dict, json_file, indent=4)
        # add numerical labels to the file
        ls = []
        for index, row in tqdm(data_df.iterrows(), total=len(data_df)):
            specie = row['Species_Class']
            ls.append(lbls_dict[specie])
        data_df['LabelNum'] = ls
        data_df.to_csv(os.path.join(data_folder, "data.csv")) # update the file after cleaning 
        # load the data as dictionary
        data = load_csv_dict(os.path.join(data_folder, "data.csv")) 
        label_map = load_json(os.path.join(data_folder, "labels_map.json"))
        label_dict = {x['FilePath_new']: label_map[x['Species_Class']] for x in data}
        station_dict = {x['FilePath_new']: x['Station'] for x in data}
        save_as_csv([(k, v) for k, v in label_dict.items()], os.path.join(data_folder, "labels.csv"))
        # MegaDetector
        print("Running the MegaDetector on the data. That can take a couple hours.")
        md = MegaDetector(
            batch_size=1, confidence_threshold=0.1
        )
        md.predict_from_array(
            file_paths=data_df['FilePath_new'].to_numpy(), 
            output_file=os.path.abspath(os.path.join(data_folder, "md_labeled.json"))
        )
        print("Finished running the MegaDetector. Output saved in ../data/md_labeled.json")
        corrupted_files = md.invalid_files
        data_df = data_df[~data_df['FilePath_new'].isin(corrupted_files)]
        print("Removed corrupted files.")
        data_df.to_csv(os.path.join(data_folder, "data.csv"), index=False)
        mapper = BBoxMapper(os.path.join(data_folder, "md_labeled.json"))
        key_map = mapper.get_keymap()
        save_as_json(key_map, os.path.join(data_folder, "bbox_map.json"))
        detector_dict = load_json(os.path.join(data_folder, "md_labeled.json"))
        all_keys = list(detector_dict.keys())
        # perform stratified splitting
        dataset = WildlifeDataset(
            keys=all_keys,
            detector_file_path=os.path.abspath(os.path.join(data_folder, "md_labeled.json")),
            label_file_path=os.path.abspath(os.path.join(data_folder, "labels.csv")),
            bbox_map=key_map,
            batch_size=128,
            augmentation=augmentation,
        )
        train_val_set, test_set = train_test_split(data_df, test_size=0.1, stratify=data_df['Species_Class'])
        train_set, val_set = train_test_split(train_val_set, test_size=0.1, stratify=train_val_set['Species_Class'])
        train_set_bb = flatten_list([key_map[k] for k in train_set['FilePath_new']])
        val_set_bb = flatten_list([key_map[k] for k in val_set['FilePath_new']])
        test_set_bb = flatten_list([key_map[k] for k in test_set['FilePath_new']])

        train_subset = subset_dataset(dataset, train_set_bb)
        val_subset = subset_dataset(dataset, val_set_bb)
        test_subset = subset_dataset(dataset, test_set_bb)

        save_as_pickle(train_subset, os.path.join(data_folder, 'train.pkl'))
        save_as_pickle(val_subset, os.path.join(data_folder, 'val.pkl'))
        save_as_pickle(test_subset, os.path.join(data_folder, 'test.pkl'))

        print("Finished preparing train, validation, and testing splits.")

    if prep_unlabeled:
        # prepare unlabeled data
        try:
            data_df = pd.read_csv(os.path.join(data_folder, "unlabeled.csv"))
        except:
            print("unlabeled.csv not found in data folder. Add the file to the folder before starting the script.")
            return
        # clean the csv file
        print("Preparing the data...")
        # MegaDetector
        print("Running the MegaDetector on the data. That can take a couple hours.")
        md = MegaDetector(
            batch_size=1, confidence_threshold=0.1
        )
        md.predict_from_array(
            file_paths=data_df['FilePath_new'].to_numpy(), 
            output_file=os.path.abspath(os.path.join(data_folder, "md_unlabeled.json"))
        )
        corrupted_files = md.invalid_files
        data_df = data_df[~data_df['FilePath_new'].isin(corrupted_files)]
        print("Removed corrupted files.")
        data_df.to_csv(os.path.join(data_folder, "unlabeled.csv"), index=False)
        print("Finished running the MegaDetector. Output saved in ../data/md_unlabeled.json")
        mapper = BBoxMapper(os.path.join(data_folder, "md_unlabeled.json"))
        key_map = mapper.get_keymap()
        save_as_json(key_map, os.path.join(data_folder, "bbox_map_unlabeled.json"))
        all_keys = flatten_list([key_map[k] for k in data_df['FilePath_new'].to_numpy()])
        md_dict = load_json(os.path.join(data_folder, "md_unlabeled.json"))
        filtered_keys = []

        print(len(all_keys))

        for key in all_keys:
            if md_dict[key]['category'] == 0 or md_dict[key]['category'] == -1:
                filtered_keys.append(key)

        print(len(filtered_keys))

        dataset = WildlifeDataset(
            keys=filtered_keys,
            detector_file_path=os.path.abspath(os.path.join(data_folder, "md_unlabeled.json")),
            bbox_map=key_map,
            batch_size=128,
        )
        save_as_pickle(dataset, os.path.join(data_folder, 'unlabeled.pkl'))
        print("Finished preparing unlabeled data.")


if __name__ == '__main__':
    main()

