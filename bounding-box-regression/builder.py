from torch import nn
import torchinfo

def build_model(model: nn.Module, with_model_summary: bool = False) -> nn.Module:
  """ Builds bounding box regression model """
  # Freeze all of the base layers in the pre trained model
  for param in model.features.parameters():
    param.requires_grad = False

  num_features = model.classifier[1].in_features

  model.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(num_features, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 4),
      nn.Sigmoid()
  )

  if with_model_summary:
    print(torchinfo.summary(model=model,
        input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]))

  return model
