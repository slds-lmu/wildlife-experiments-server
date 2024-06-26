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
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
from wildlifeml.preprocessing.cropping import Cropper

from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple


def render_bbox(
        img: Image,
        x_coords: List[Tuple[int, int]],
        y_coords: List[Tuple[int, int]],
        class_name: str,
        confidence: float,
        outline: str = 'red',
        border_width: int = 10,
        font_size: int = 40,
        text_color: str = 'white'
) -> Image:
    """Render bounding boxes with class names and confidences into a PIL Image."""
    img_draw = ImageDraw.Draw(img)
    try:
        # Load a truetype or opentype font file
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # If the specific font is not found, use the default PIL font
        font = ImageFont.load_default()

    for x, y in zip(x_coords, y_coords):
        # Draw the bounding box
        img_draw.rectangle(
            xy=((x[0], y[0]), (x[1], y[1])),
            outline=outline,
            width=border_width,
        )
    # Prepare the label text
    label = f"{class_name}: {confidence:.2f}"
    # Calculate text size to create a background rectangle
    text_bbox = img_draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_location = (x[0], y[0] - text_height)
    background_location = [x[0], y[0] - text_height, x[0] + text_width, y[0]]

    # Draw a rectangle for the text background
    img_draw.rectangle(background_location, fill=outline)
    # Draw the text above the bounding box
    img_draw.text(text_location, label, fill=text_color, font=font)

    return img


@click.command()
@click.option(
    '--model_path', '-p', help='The path to the model you want to use for predictions.', required=True, type=str
)
@click.option(
    '--thresh', '-t', help='The threshold associated with your model of choice.', required=True, type=float
)
@click.option(
    '--make_folders', '-f', help='Whether you want to create folders with images for each species.', required=False, type=bool, default=False
)
@click.option(
    '--data_folder', '-d', help='Folder where you store the csv files.', required=False, type=str, default="../data"
)

def main(model_path: str, thresh: float, make_folders: bool, data_folder: str):
    pred_folder = "../predictions"

    model = load_model(model_path, custom_objects={'imagenet_utils': imagenet_utils})
    dataset = load_pickle(os.path.join(data_folder, "unlabeled.pkl"))
    labels_map = load_json(os.path.join(data_folder, "labels_map.json"))
    evaluator = Evaluator(
        detector_file_path=os.path.join(data_folder, "md_unlabeled.json"),
        num_classes=len(labels_map.keys()),
        empty_class_id=labels_map.get("Empty") or 0,
        conf_threshold=thresh,
        dataset=dataset
    )
    evaluator.evaluate(model)
    # folder for current predictions
    current_datetime = datetime.now()
    folder_name = os.path.join(pred_folder, current_datetime.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(folder_name, exist_ok=True)
    evaluator.save_predictions(filepath=os.path.join(folder_name, 'predictions.csv'), img_level=False)
    # make it clearer for LWF people to read
    df = pd.read_csv(os.path.join(folder_name, 'predictions.csv'))
    labels = df['hard_label'].to_numpy()
    reverse_mapping = {v: k for k, v in labels_map.items()}
    mapped_labels = [reverse_mapping[label] for label in labels]
    df['species_class'] = mapped_labels
    df.to_csv(os.path.join(folder_name, 'predictions.csv'))
    # folder for each species
    print("Generating folders for the species. This can take some time...")
    md_dict = load_json(os.path.join(data_folder, "md_unlabeled.json"))
    md_mapper = load_json(os.path.join(data_folder, "bbox_map_unlabeled.json"))
    if make_folders:
        for species in tqdm(labels_map.keys()):
            current_folder = os.path.join(folder_name, species)
            os.makedirs(current_folder, exist_ok=True)
            for index, row in df.iterrows():
                if row['hard_label'] == labels_map[species]:
                    # map_key = md_mapper[row['img_key']]
                    if md_dict[row['img_key']]['category'] == 0:
                        img_name = row['img_key'].split(".")[0] + ".JPG"
                        img = Image.open(img_name)
                        width, height = img.size
                        box = md_dict[row['img_key']]['bbox']
                        x_coords = []
                        y_coords = []
                        if box is not None:
                            x, y = Cropper.get_absolute_coords(box, (height, width))
                            x_coords.append(x)
                            y_coords.append(y)
                        class_indx = row['hard_label']
                        confidence = row["prob_class_" + str(class_indx)]
                        img = render_bbox(img, x_coords, y_coords, confidence=confidence, class_name=row['species_class'])
                        img.save(os.path.join(current_folder, row['img_key'].split("/")[-1].split(".")[0] + "_" + str(row['img_key'][-3:]) + ".JPG"))

if __name__ == '__main__':
    main()
