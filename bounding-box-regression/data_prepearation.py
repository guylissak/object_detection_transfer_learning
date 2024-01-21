import os
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from PIL import Image
from config import config as conf

def prepare_annotations_data() -> List[Tuple[float, ...]]:
  """ Formatting and scaling annotations data """
  with open(conf.ANNOTS_PATH, 'r') as file:
    csv_annots = file.read().strip().split("\n")

  # loop over the rows
  targets = []
  for row in csv_annots:
      # break the row into the filename and bounding box coordinates
      row = row.split(",")
      (filename, min_x, min_y, max_x, max_y) = row
      image_path = os.path.sep.join([conf.IMAGES_PATH, filename])
      original_image = Image.open(image_path)
      width, height = original_image.size

      # scale the bounding box coordinates relative to the dimensions of the input image
      min_x = float(min_x) / width
      min_y = float(min_y) / height
      max_x = float(max_x) / width
      max_y = float(max_y) / height

      targets.append((min_x, min_y, max_x, max_y))

  return targets

def split_train_test_data(images_paths: List[str], targets = List[Tuple[float, ...]], seed: int = 42, test_size: float = 0.2):
  """ Split data to test/ train """
  train_images, test_images, train_targets, test_targets = train_test_split(
      images_paths, targets, test_size=test_size, random_state=seed
  )

  print(f"Train images: {len(train_images)}")
  print(f"Test images: {len(test_images)}")
  print(f"Train targets: {len(train_targets)}")
  print(f"Test targets: {len(test_targets)}")

  return train_images, test_images, train_targets, test_targets


