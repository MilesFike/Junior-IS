from pickletools import uint8
import torch
import tensorflow_datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#What EMNIST DATA LOOKS LIKE according to https://www.tensorflow.org/datasets/catalog/emnist:
#FeaturesDict({
#    'image': Image(shape=(28, 28, 1), dtype=uint8),
#    'label': ClassLabel(shape=(), dtype=int64, num_classes=62),
#})

trainset = torchvision.datasets.EMNIST(
    root="./data",
    split="letters",   # try 'letters' first
    train=True,
    download=True,
    transform=transform
)