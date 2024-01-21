""" Utils """
import torch
from torch import Tensor, nn
from config import config as conf
from typing import List, Tuple, Dict
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np
from PIL import Image
import random
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, EfficientNet
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url



def plot_original_image(seed:int|None= None):
  """ Plots single random original image """
  if seed:
    random.seed(seed)

  image_path = Path(conf.IMAGES_PATH)
  image_path_list = list(image_path.glob("*.jpg"))
  random_image_path = random.choice(image_path_list)
  img = Image.open(random_image_path)
  img_as_array = np.asarray(img)

  plt.figure(figsize=(7, 7))
  plt.title(f"Original image: {random_image_path} , image shape: {img_as_array.shape} H x W x Channels")
  plt.imshow(img_as_array)

import matplotlib.pyplot as plt

def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """Plots a series of random images from image_paths"""
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")


def plot_loss_curves(results: Dict):
    """Plots training curves of a results dictionary.
    """

    loss = results["train_loss"]
    test_loss = results["test_loss"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()


def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)


def get_pre_trained_model_and_weights() -> Tuple[EfficientNet, EfficientNet_B0_Weights]:
  """Returns pre-trained model """
  WeightsEnum.get_state_dict = get_state_dict
  weights = EfficientNet_B0_Weights.DEFAULT # ".DEFAULT" = best available weights
  model = efficientnet_b0(weights=weights).to(conf.DEVICE)
  return model, weights


def scale_bbox_to_image(bbox_tensor: Tensor, img_width, img_height) -> List:
  """Scales a bounding box tensor to the original image size."""
  return [
      bbox_tensor[0].item() * img_width,  # x_min
      bbox_tensor[1].item() * img_height, # y_min
      bbox_tensor[2].item() * img_width,  # x_max
      bbox_tensor[3].item() * img_height  # y_max
  ]


def create_rectangle_plot(original_height: int, original_width: int, bbox_tensor: Tensor, ax) -> patches.Rectangle:
  """ Create rectangle plot for a given bounding box tensor."""

  bbox = scale_bbox_to_image(bbox_tensor, original_width, original_height)
  rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor='r', facecolor='none')
  return rect


def inference_custom_image(model: torch.nn.Module,
                        image_path: str,
                        transform,
                        annotations=None,
                        device: torch.device = conf.DEVICE):
    """Makes a prediction on a target image and plots the image with its bounding box."""
    # 1. Load in image
    target_image = Image.open(image_path).convert("RGB")

    # 2. Keep a copy of the original image for plotting
    original_image = target_image.copy()

    # 3. Transform if necessary
    target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred_bbox = model(target_image.to(device))

        # Squeeze to remove batch dimension and move to CPU
        target_image_pred_bbox = target_image_pred_bbox.squeeze().cpu()

    # 6. Plot the image alongside the prediction bounding box
    fig, ax = plt.subplots(1)
    ax.imshow(original_image)  # make sure it's the right size for matplotlib

    # Scale the bounding box from [0, 1] range to image size
    bbox = target_image_pred_bbox.numpy()
    bbox_scaled = scale_bbox_to_image(bbox, original_image.width, original_image.height)
    rect = patches.Rectangle((bbox_scaled[0], bbox_scaled[1]), bbox_scaled[2] - bbox_scaled[0], bbox_scaled[3] - bbox_scaled[1], linewidth=1, edgecolor='c', facecolor='none', label='predicted')
    ax.add_patch(rect)
    if annotations:
      actual_rect = patches.Rectangle((float(annotations[0]), float(annotations[1])), float(annotations[2]) - float(annotations[0]), float(annotations[3]) - float(annotations[1]), linewidth=1, edgecolor='g', facecolor='none', label='actual')
      ax.add_patch(actual_rect)
    plt.title(f"Bounding box: {bbox_scaled},\n actual bounding box: {annotations}")
    plt.axis(False)
    plt.legend()
    plt.show()


