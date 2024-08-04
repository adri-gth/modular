import os
import zipfile
from pathlib import Path
import requests
import torch
from torchvision import transforms
from typing import Tuple, Optional, List, Union,Dict
import pandas as pd
import matplotlib.pyplot as plt


"""Contiene codigos de ayuda para el procesamiento de la informacion y el entrenamiento
del los modelo
"""

def create_transforms(resize: Optional[Tuple[int,int]]= None,
                      to_tensor:bool = True,
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

  if to_tensor:
    transform_list.append(transforms.ToTensor())

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


