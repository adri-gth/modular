from torch.utils.tensorboard import SummaryWriter
import os
import zipfile
from pathlib import Path
import requests
import torch
from torchvision import transforms
from typing import Tuple, Optional, List, Union,Dict
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from torch import nn

"""Contiene codigos de ayuda para el procesamiento de la informacion y el entrenamiento
del los modelo
"""

def create_transforms(resize: Optional[Tuple[int,int]]= None,
                      to_tensor:bool = True,
                      normalize:bool = True,
                      randomRotation: Optional[Tuple[int,int]]= None, #cuanto se puede rotar la imagen, en grados
                      randomAdjustSharpness: Optional[int] = None, #cuánto se debe ajustar la nitidez. Puede ser cualquier número no negativo. 0 da una imagen borrosa, 1 da la imagen original y 2 aumenta la nitidez en un factor de 2.
                      randomAutocontrast: bool = False,
                      randomHorizontalFlip: bool = False,
                      randomVerticalFlip: bool = False

                      ) -> transforms.Compose:

  """
  Crea las transformaciones para hacer un modificaciones al conjunto de datos originales, como girar la imagen, modificar el contraste, etc.

  Toma los datos y devuelve una lista con las transformaciones que se van a aplicar

  Args:
    resize: toma una tupla de enteros, los cuales van a redimencionar la imagen
    to_Tensor: convierten los datos a tensores
    randomRotation: recibe una tupla de enteros, estos establcen los limites superior e inferior, respectivamente, entre los cuales esta la posibilidad de rotar la imagen
    randomAdjustSharpness: recibe un dato entero entre 0 y 2 y establece cuánto se debe ajustar la nitidez. Puede ser cualquier número no negativo. 0 da una imagen borrosa, 1 da la imagen original y 2 aumenta la nitidez en un factor de 2.
    randomAutocontrast: modidica el contraste de las imagenes en una probabilidad del 50%
    randomHorizontalFlip: aplica un efecto espejo en el eje horizontal, con una probabilidad del 50 %
    randomVerticalFlip: aplica un efecto espejo en el eje vertical, con una probabilidad del 50 %

  Ejemplo de uso
    transform = utils.create_transforms(resize = (64,64),randomRotation = (0,180),
                                        randomHorizontalFlip=True,randomAdjustSharpness = 2,
                                        randomAutocontrast = True,randomVerticalFlip=True)
  """
  transform_list = []

  if resize:
    transform_list.append(transforms.Resize(resize))

  if to_tensor:
    transform_list.append(transforms.ToTensor())

  if normalize:
    transform_list.append(transforms.Normalize(mean=[0.485,0.456,0.406],
                                               std = [0.229,0.224,0.225]))

  if randomRotation:
    transform_list.append(transforms.RandomRotation(degrees = randomRotation))

  if randomAdjustSharpness:
    transform_list.append(transforms.RandomAdjustSharpness(sharpness_factor=randomAdjustSharpness))

  if randomAutocontrast:
    transform_list.append(transforms.RandomAutocontrast(p=0.5))

  if randomHorizontalFlip:
    transform_list.append(transforms.RandomHorizontalFlip(p = 0.5))

  if randomVerticalFlip:
    transform_list.append(transforms.RandomVerticalFlip(p=0.5))



  return transforms.Compose(transform_list)


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """
  Guarda el modelo en un diretorio objetivo

  Args:
    model: modelo a guardar
    target_dir: directrio para guardar el modelo
    model_name: el archivo en que se guardara el modelo, este debe incluir
      ".pth" o ".pt" como extension del archivo

    Ejemplo de uso:
      save_model(model_0,
                  target_dir="models",
                  model_name="0_tinyvgg_model.pth")
  """
  #create a target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  #create a model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with .pt or .pth"
  model_save_path = target_dir_path / model_name

  #save the model state dict
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)


def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str = None):
  """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance tracking to a specific directory."""
  from datetime import datetime
  import os

  # Get timestamp of current date in reverse order
  timestamp = datetime.now().strftime("%Y-%m-%d")

  if extra:
    # Create log directory path
    log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
  else:
    log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
  print(f"[INFO] Created SummaryWriter saving to {log_dir}")
  return SummaryWriter(log_dir=log_dir)




def create_effnetb0(OUT_FEATURES:int)-> nn.Module:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Get the weights and setup a model
  weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
  model = torchvision.models.efficientnet_b0(weights=weights).to(device)

  # Freeze the base model layers
  # Congelar los primeros 2 bloques (bloques 0 y 1 en model.features)
  for name, module in model.features.named_children():
      if name in ['0', '1','2','3']:  # Congela los bloques 0,1,2,3
          for param in module.parameters():
              param.requires_grad = False
      else:  # Descongela los bloques restantes
          for param in module.parameters():
              param.requires_grad = True

  # Change the classifier head

  model.classifier = nn.Sequential(
      nn.Dropout(p=0.2, inplace=True),
      nn.Linear(in_features=1280, out_features=OUT_FEATURES)
  ).to(device)

  # Give the model a name
  model.name = "effnetb0"
  print(f"[INFO] Created new {model.name} model...")
  return model

# Create an EffNetB2 feature extractor
def create_mobilnetv3s(OUT_FEATURES:int)-> nn.Module:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Get the weights and setup a model
  weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
  model = torchvision.models.mobilenet_v3_small(weights=weights).to(device)


  # Freeze the base model layers
  # Congelar los primeros 2 bloques (bloques 0 y 1 en model.features)
  for name, module in model.features.named_children():
      if name in ['0', '1','2','3']:  # Congela los bloques 0,1,2,3
          for param in module.parameters():
              param.requires_grad = False
      else:  # Descongela los bloques restantes
          for param in module.parameters():
              param.requires_grad = True

  # Change the classifier head
  model.classifier = nn.Sequential(
      nn.Linear(in_features = 576, out_features = 1024, bias = True),
      nn.Hardswish(),
      nn.Dropout(p=0.2, inplace=True),
      nn.Linear(in_features=1024, out_features=OUT_FEATURES,bias = True)
  ).to(device)

  # Give the model a name
  model.name = "mobilnetv3s"
  print(f"[INFO] Created new {model.name} model...")
  return model

