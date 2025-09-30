from pickletools import uint8 #Likely not necessary
#from matplotlib.dviread import pyplot as plt
import numpy as np
import torch
import torchvision
#import tensorflow_datasets #Installation failed
import torchvision.transforms
import matplotlib.pyplot
import emnist
from emnist import list_datasets
from PIL import Image
#import pandas
#import plt

#What EMNIST DATA LOOKS LIKE according to https://www.tensorflow.org/datasets/catalog/emnist:
#FeaturesDict({
#    'image': Image(shape=(28, 28, 1), dtype=uint8), #This part requires pickletools
#    'label': ClassLabel(shape=(), dtype=int64, num_classes=62),
#})

initData = torchvision.datasets.EMNIST(
    root = "./data", #Pytorch downloads the data in the root
    split= "letters", #byclass is all characters. train for MNIST?
    train = True,
    download = True,
)

#mapping = emnist.read_mapping('emnist-letters-mapping.txt')#This gets the mapping that gives characters meaning.

labels = initData.targets
print(initData) #torchvision is storing dataset metadata
#print(EMNIST) #EMNIST is not printable
#print(dataset[0]['image'])Tuple not dictionary despite name
print(initData[0][0])
print(initData[0][1])

im = initData[0][0]
#print(chr(mapping[labels[0]]))
rotated = im.rotate(90)
im.show()
rotated.show()
print(f"Label: {labels[0]}, Letter: {chr(labels[0] + 96)}")


im = initData[1][0]
#print(chr(mapping[labels[0]]))
rotated = im.rotate(90)
im.show()
rotated.show()
print(f"Label: {labels[1]}, Letter: {chr(labels[1] + 96)}")


im = initData[10000][0]
#print(chr(mapping[labels[10000]]))

rotated = im.rotate(90)
im.show()
rotated.show()
print(chr(labels[10000]+96))


im = initData[10001][0]
#print(labels[10001])
#print(chr(mapping[labels[10000]]))
rotated = im.rotate(90)
im.show()
rotated.show()
print(chr(labels[10001]+96))




#print(dataset[0][0][0][0][0][1]) Only two levels in dataset

#testset = torchvision.datasets.EMNIST(
#    root = "./data",
#    split = "test",
#    train = False,
#    download = True,
#)

#with tf.Session() as sess:
#    #access first image
#    first_image = EMNIST.train.images[0]
#    first_image = np.array(first_image, dtype='uint8')
#    pixels = first_image.reshape((28, 28))
#    plt.imshow(pixels, cmap='gray')