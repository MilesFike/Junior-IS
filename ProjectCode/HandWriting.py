import torch
import torchvision
import torch.nn as nn #Provides convolution requirements
import torchvision.transforms
from emnist import list_datasets
from PIL import Image
import torch.optim as optim
from visionClass import net
import torchvision.transforms.functional as Tform
import matplotlib.pyplot as plt
from lineToLetter import run, segment_letters
from imageProcessing import process


#What EMNIST DATA LOOKS LIKE according to https://www.tensorflow.org/datasets/catalog/emnist:
#FeaturesDict({
#    'image': Image(shape=(28, 28, 1), dtype=uint8), #This part requires pickletools
#    'label': ClassLabel(shape=(), dtype=int64, num_classes=62),
#})

transform = torchvision.transforms.ToTensor()

if __name__ == "__main__":
    #print("In file")
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
    #print(initData) #torchvision is storing dataset metadata
    #print(EMNIST) #EMNIST is not printable
    #print(dataset[0]['image'])Tuple not dictionary despite name
    #print(initData1[0][0])
    #print(initData1[0][1])

    im = initData1[0][0]
#print(chr(mapping[labels[0]]))
    #rotated = im.rotate(90)
    if(shower == "y"):
        im.show()
    #    rotated.show()
        print(f"Label: {labels[0]}, Letter: {chr(labels[0] + 96)}")


    im = initData1[1][0]
#print(chr(mapping[labels[0]]))
    #rotated = im.rotate(90)
    if(shower == "y"):
        im.show()
    #    rotated.show()
        print(f"Label: {labels[1]}, Letter: {chr(labels[1] + 96)}")


    im = initData1[10000][0]
#print(chr(mapping[labels[10000]]))

    #rotated = im.rotate(90)
    if(shower == "y"):
        im.show()
    #    rotated.show()
        print(chr(labels[10000]+96))


    im = initData1[10001][0]
#print(labels[10001])
#print(chr(mapping[labels[10000]]))
    #rotated = im.rotate(90)
    if(shower == "y"):
        im.show()
        #rotated.show()
        print(chr(labels[10001]+96))

#print(dataset[0][0][0][0][0][1]) Only two levels in dataset

#testset = torchvision.datasets.EMNIST(
#    root = "./data",
#    split = "test",
#    train = False,
#    download = True,
#)


    labelTest = input("Label testing(y/n):")

    if(labelTest == 'y'):
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

    compVision = input("Actual Computer Vision System(y/n):")
    if(compVision == "y"):
        criterion = nn.CrossEntropyLoss() #taken fom https://www.geeksforgeeks.org/deep-learning/computer-vision-with-pytorch/
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #net from model short for network

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
                labels = labels - 1
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            percent = correct/total
            print(f'Accuracy of the network on test images: {percent:.2f}%')


        #My tests for individual letters so I can check if it works
        testiter = iter(testloader)
        images, labels = next(testiter)
        image = images[1]
        label = labels[1] 

        imageInput = image.unsqueeze(0) #This just adds that extra 1 in the list to make it a batch of 1   

        with torch.no_grad():
            outputs = net(imageInput)
            _, predicted = torch.max(outputs, 1)

        real = chr(label+96)
        predicted_letter = chr(predicted.item() + 97) # 97 because I didn't adjust labels
        print(f"Actual Label {real}")
        print(f"Predicted Label {predicted_letter}")

        plt.imshow(image.squeeze(), cmap='gray')
        plt.title("Test Image")
        plt.axis('off')
        plt.show()


        image = images[2]
        label = labels[2] 

        imageInput = image.unsqueeze(0) #This just adds that extra 1 in the list to make it a batch of 1   

        with torch.no_grad():
            outputs = net(imageInput)
            _, predicted = torch.max(outputs, 1)

        real = chr(label+96)
        predicted_letter = chr(predicted.item() + 97) # 97 because I didn't adjust labels
        print(f"Actual Label {real}")
        print(f"Predicted Label {predicted_letter}")

        plt.imshow(image.squeeze(), cmap='gray')
        plt.title("Test Image")
        plt.axis('off')
        plt.show()

        run("imgs\milesFike.png")
        i =     segment_letters("imgs\milesFike.png","imgs\m2.png", output_dir="letters")
        for j in range(i):
            process(f"letters/letter{j}.png")
            image_path = "imgs/m2.png"  # your prepared image
            img = Image.open(image_path)
            img.show()
            img_tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0)  # [1, 1, 28, 28] This is because the network needs to receive a tensor format

            # Running my M  through the network
            device = torch.device("cpu")
            net.to(device)
            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                outputs = net(img_tensor)
                _, predicted = torch.max(outputs, 1)

            predicted_letter = chr(predicted.item() + 97)  #adjustment for labels because ascii
            print(f"Predicted letter: {predicted_letter}")

        