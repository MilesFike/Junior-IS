import torch
import torchvision
import torch.nn as nn #Provides convolution requirements
import torchvision.transforms
from emnist import list_datasets
from PIL import Image
import torch.optim as optim
from visionClass import net

#What EMNIST DATA LOOKS LIKE according to https://www.tensorflow.org/datasets/catalog/emnist:
#FeaturesDict({
#    'image': Image(shape=(28, 28, 1), dtype=uint8), #This part requires pickletools
#    'label': ClassLabel(shape=(), dtype=int64, num_classes=62),
#})

transform = torchvision.transforms.ToTensor()

if __name__ == "__main__":
    shower = input("Show images(y/n):")
    initData = torchvision.datasets.EMNIST(
        root = "./data", #Pytorch downloads the data in the root
        split= "letters", #byclass is all characters. train for MNIST?
        train = True,
        download = True,
        transform=transform #This converts to tensor so batch works in Trainloader
    )

    initData1 = torchvision.datasets.EMNIST(
        root = "./data", #Pytorch downloads the data in the root
        split= "letters", #byclass is all characters. train for MNIST?
        train = True,
        download = True,
        #No transfer so can look at im data
    )

    trainloader = torch.utils.data.DataLoader(initData, batch_size=32, shuffle=True, num_workers=2)

    testData = torchvision.datasets.EMNIST(
        root = "./data", #Pytorch downloads the data in the root
        split= "letters", #byclass is all characters. train for MNIST?
        train = False,
        download = True,
        transform = transform,
    )
#mapping = emnist.read_mapping('emnist-letters-mapping.txt')#This gets the mapping that gives characters meaning.
    testloader = torch.utils.data.DataLoader(testData, batch_size=32, shuffle=False, num_workers=2)

    labels = initData.targets
    print(initData) #torchvision is storing dataset metadata
#print(EMNIST) #EMNIST is not printable
#print(dataset[0]['image'])Tuple not dictionary despite name
    print(initData1[0][0])
    print(initData1[0][1])

    im = initData1[0][0]
#print(chr(mapping[labels[0]]))
    rotated = im.rotate(90)
    if(shower == "y"):
        im.show()
        rotated.show()
    print(f"Label: {labels[0]}, Letter: {chr(labels[0] + 96)}")


    im = initData1[1][0]
#print(chr(mapping[labels[0]]))
    rotated = im.rotate(90)
    if(shower == "y"):
        im.show()
        rotated.show()
    print(f"Label: {labels[1]}, Letter: {chr(labels[1] + 96)}")


    im = initData1[10000][0]
#print(chr(mapping[labels[10000]]))

    rotated = im.rotate(90)
    if(shower == "y"):
        im.show()
        rotated.show()
    print(chr(labels[10000]+96))


    im = initData1[10001][0]
#print(labels[10001])
#print(chr(mapping[labels[10000]]))
    rotated = im.rotate(90)
    if(shower == "y"):
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


    print("Label testing:")

    alphabeta = [chr(i) for i in range(97, 123)]
    print(alphabeta)

    numAB = [(ord(i) - 96) for i in alphabeta]
    print(numAB)


    locFirstB = [] #A's are difficult to read
    for i in range(1000):
        if labels[i] == numAB[1]:
            locFirstB.append(i)
        if len(locFirstB) > 4:
            break

    for i in locFirstB:
        im = initData1[i][0]
        im.show()

    print("Actual Computer Vision System:")
        
    criterion = nn.CrossEntropyLoss() #taken fom https://www.geeksforgeeks.org/deep-learning/computer-vision-with-pytorch/
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #net from model short for network



    print("Block entered")
    for i in range(2): 
        device = torch.device("cpu")
        net.to(device)  #makes it run on cpu
        running_loss = 0

        for j, data in enumerate(trainloader, 0):
            inputs, labels = data
            labels = labels - 1 #Fixes out of bounds

            inputs, labels = inputs.to(device), labels.to(device)
            
            #print(f"Input batch shape: {inputs.shape}")
            #print(f"Labels batch shape: {labels.shape}")
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if j % 100 == 99:  # print every 100 mini-batches down scaling for efficiency, and because I want to see results.
                print(f'[{i + 1}, {j + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0

    print('This model is trained')

#Test primarily derived from https://www.geeksforgeeks.org/deep-learning/computer-vision-with-pytorch/
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        percent = correct/total
        print(f'Accuracy of the network on test images: {percent:.2f}%')