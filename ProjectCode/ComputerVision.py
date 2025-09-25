#from pickletools import uint8 #Likely not necessary
import torch
import torchvision
#import tensorflow_datasets
import torchvision.transforms
import matplotlib.pyplot

#What EMNIST DATA LOOKS LIKE according to https://www.tensorflow.org/datasets/catalog/emnist:
#FeaturesDict({
#    'image': Image(shape=(28, 28, 1), dtype=uint8), #This part requires pickletools
#    'label': ClassLabel(shape=(), dtype=int64, num_classes=62),
#})

dataset = torchvision.datasets.EMNIST(
    root = "./data", #Pytorch downloads the data in the root
    split= "byclass", #byclass is all characters. train for MNIST?
    train = True,
    download = True,
)
print(dataset) #torchvision is storing dataset metadata
#print(EMNIST) #EMNIST is not printable
#print(dataset[0]['image'])Tuple not dictionary despite name
print(dataset[0][0])
print(dataset[0][1])

#testset = torchvision.datasets.EMNIST(
#    root = "./data",
#    split = "test",
#    train = False,
#    download = True,
#)