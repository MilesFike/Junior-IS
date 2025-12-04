import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
#Code above ensures datasets are downloaded correctly

from pathlib import Path

BASE_DIR = Path(__file__).parent# ProjectCode/
PARENT_DIR = BASE_DIR.parent # Junior-IS/

IMGS_DIR = BASE_DIR / "imgs"
LETTERS_DIR = BASE_DIR / "letters"
IMGS_DIR.mkdir(exist_ok=True)
LETTERS_DIR.mkdir(exist_ok=True)


import torch
import torchvision
import torch.nn as nn #Provides convolution requirements
import torchvision.transforms
from PIL import Image
import torch.optim as optim
from visionClassAllChars import net
from ChoiceDataset import CHoiCeDataset
import torchvision.transforms.functional as Tform
import matplotlib.pyplot as plt
from lineToLetter import run, segment_letters
from CHoiCeImageProcessing import process


#What EMNIST DATA LOOKS LIKE according to https://www.tensorflow.org/datasets/catalog/emnist:
#FeaturesDict({
#    'image': Image(shape=(28, 28, 1), dtype=uint8), #This part requires pickletools
#    'label': ClassLabel(shape=(), dtype=int64, num_classes=62),
#})

transform = torchvision.transforms.ToTensor()

if __name__ == "__main__":
    def labelCheck(label):
        # EMNIST label mapping: 0-9 (digits), 10-35 (A-Z), 36-61 (a-z)
        if(label < 10):
            return label + 48  # 0-9 → ASCII 48-57
        elif(label < 36):
            return (label - 10) + 65  # 10-35 → ASCII 65-90 (A-Z)
        else:
            return (label - 36) + 97  # 36-61 → ASCII 97-122 (a-z)



    shower = input("Show images(y/n):")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor()
    ])

   #CHoiCe-Dataset is in the ProjectCode directory
    dataset = CHoiCeDataset(root_dir=str(BASE_DIR / "CHoiCe-Dataset" / "V0.3"), transform=transform, exclude_digits=True)

# Splitting into train/test parts
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
#print(chr(mapping[labels[0]]))
    #rotated = im.rotate(90)
    if(shower == "y"):
        print("Showing some images from dataset")
#print(labels[10001])
#print(chr(mapping[labels[10000]]))
    #rotated = im.rotate(90)


#print(dataset[0][0][0][0][0][1]) Only two levels in dataset

#testset = torchvision.datasets.EMNIST(
#    root = "./data",
#    split = "test",
#    train = False,
#    download = True,
#)


    labelTest = input("Label testing(y/n):")

    if(labelTest == 'y'):
        alphabeta = [chr(i) for i in range(48, 58)]
        for i in range(65, 91):
            alphabeta.append(chr(i))
        for i in range(97, 123):
            alphabeta.append(chr(i))
        print(alphabeta)

    compVision = input("Actual Computer Vision System(y/n):")
    if(compVision == "y"):
        criterion = nn.CrossEntropyLoss() #taken fom https://www.geeksforgeeks.org/deep-learning/computer-vision-with-pytorch/
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9) #Increased from 0.001 to 0.01

        # Increased from 3 to 30 epochs for better training on small dataset
        print("Training for 45 epochs")
        for i in range(45): 
            device = torch.device("cpu")
            net.to(device)  #makes it run on cpu
            running_loss = 0

            for j, data in enumerate(trainloader, 0):
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)
            
            #print(f"Input batch shape: {inputs.shape}")
            #print(f"Labels batch shape: {labels.shape}")
                optimizer.zero_grad()
            
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if j % 10 == 9:  #prints every 10 mini-batches (dataset is extra small)
                    print(f'[Epoch {i + 1}, Batch {j + 1}] loss: {running_loss / 10:.3f}')
                    running_loss = 0

        print('This model is trained images will be tested on now. This should be mostly effective.')
        torch.save(net.state_dict(), 'CHOICE.pth')
        print('Model saved!')
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
            print(f'Accuracy of the network on its test images: {percent:.2f}%')


        #My tests for individual letters so I can check if it works
        testiter = iter(testloader)
        images, labels = next(testiter)
        image = images[1]
        label = labels[1] 

        imageInput = image.unsqueeze(0) #This just adds that extra 1 in the list to make it a batch of 1   

        with torch.no_grad():
            outputs = net(imageInput)
            _, predicted = torch.max(outputs, 1)

        real = chr(labelCheck(label.item()))
        predicted_letter = chr(labelCheck(predicted.item())) # 97 because I didn't adjust labels
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

        real = chr(labelCheck(label.item()))
        predicted_letter = chr(labelCheck(predicted.item())) # 97 because I didn't adjust labels
        print(f"Actual Label {real}")
        print(f"Predicted Label {predicted_letter}")

        plt.imshow(image.squeeze(), cmap='gray')
        plt.title("Test Image")
        plt.axis('off')
        plt.show()

        run(str(IMGS_DIR / "milesFike.png"))
        i =     segment_letters(str(IMGS_DIR / "milesFike.png"),str(IMGS_DIR / "m2.png"), output_dir=str(LETTERS_DIR))
        for j in range(i):
            process(str(LETTERS_DIR / f"letter{j}.png"))
            image_path = str(IMGS_DIR / "m2.png")  # your prepared image
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

            predicted_letter = chr(labelCheck(predicted.item()))  #adjustment for labels because ascii
            print(f"Predicted letter: {predicted_letter}")

        