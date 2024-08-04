import torch
from torch import nn
"""
  Contiene codigo PyTorch para instanciar un modelo TinyVGG del la CNN Explainer webside
  """

class TinyVGG(nn.Module):
  """
  Crea la arquitectura TinyVGG

  Replica la arquitectura TinyVGG de la CNN Explainer webside en PyTorch

  Args:
    input_shape: numero entero que indica el numero de canales de la imagen
    hidden_units: numero entero que indica el numero de unidades ocultas entre capas
    output_shape: numero entero que indica el numero de unidades de salida o categorias de salida
  """
  def __init__(self, input_shape: int,
               hidden_units: int,
               output_shape: int)->None:
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)# default stride value is same as kernel_size

    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)# default stride value is same as kernel_size

    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*13*13,
                  out_features=output_shape)
    )
  def forward(self, x):
    #x = self.conv_block_1(x)
    #x = self.conv_block_2(x)
    #x = self.classifier(x)
    #return x
    return self.classifier(self.conv_block_2(self.conv_block_1(x)))