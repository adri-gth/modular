"""
Contiene funcionalidades para crear los Dataset y DataLoader en PyTorch para datos de clasificación de imágenes.
"""
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
):
    """
    Crea los DataLoaders para entrenamiento y prueba de datos

    Toma directorios de entrenamiento y prueba, genera los Datasets y regresa en DataLoaders.

    Args:
        train_dir: Dirección del directorio en el que se ubican los datos de entrenamiento.
        test_dir: Dirección del directorio en el que se ubican los datos de prueba.
        transform: Transformaciones de torchvision para aplicar en los datos de entrenamiento y prueba.
        batch_size: Número de muestras por lote en cada uno de los DataLoader.
        num_workers: Un valor entero para el número de trabajadores por DataLoader.

    Returns:
        Una tupla de (train_dataloader, test_dataloader, class_names).
        Donde class_names es una lista de las clases objetivo.

    Example usage:
        train_dataloader, test_dataloader, class_names = create_dataloaders(
                            train_dir="dirección/a/datos/de/entrenamiento",
                            test_dir="dirección/a/datos/de/prueba",
                            transform=algunas_transformaciones,
                            batch_size=32,
                            num_workers=4)
    """
    # Utilizamos ImageFolder para crear los Datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Obtenemos el nombre de las clases
    class_names = train_data.classes

    # Convertimos las imágenes en data loader
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names