from PIL import Image
from PIL import ImageOps#Helps create original black but textured background with color import
import cv2 #cv2 used for more background conversions
import numpy 
import time
def run(imPathM):
    m2 = cv2.imread(imPathM)
    #converts to gray scale
    gray = cv2.cvtColor(m2, cv2.COLOR_BGR2GRAY)\
    #This extra Gaussian Blur reduces the background noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    #Pixels above threshold become white. These are then replaced with black
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    black_background = numpy.zeros_like(m2)

    result = cv2.bitwise_and(m2, m2, mask=mask) + black_background
    #cv2.imshow("m2", m2)
    #checks for all black pixels

    cv2.imwrite("imgs\m2.png", result)


    img = Image.open("imgs\m2.png")
    #collecting data for iteration
    pixels = img.load()
    width, height, = img.size
    for x in range(width):
        for y in range(height):
            currentColor = pixels[x,y]
            #print(currentColor) Do not uncomment this line
            if currentColor != (0, 0, 0):
                pixels[x,y] = (255, 255, 255)
    #img.show()
    img = img.resize((128, 128))
    #img.show()
    img.save("imgs\m2.png")
    #Transfering back to cv2 for Gaussian Blur
    m2 = cv2.imread("imgs\m2.png")
    m2 = cv2.GaussianBlur(m2, (5, 5), 1)
    cv2.imwrite("imgs\m2.png", m2)
    m3 = Image.open("imgs\m2.png")

    #m3.show
    #cv2.imwrite("imgs\m3.png", m3)

    listOfEdgePixels = []
    m3 = Image.open("imgs\m2.png")
    pixels = m3.load()#Somehow gives attribute size

    width, height, = m3.size
    m3.show() #This is the last show where it works

    
    img = cv2.imread("imgs\m2.png")

    #2 pixel padding
    padded = cv2.copyMakeBorder(
        img,
    top=2, bottom=2, left=2, right=2,
    borderType=cv2.BORDER_CONSTANT,
    value=(0, 0, 0) ) # black color)


    cv2.imwrite("imgs\m2.png", padded)
    m3 = Image.open("imgs\m2.png")
    pixels = m3.load()#Somehow gives attribute size

    width, height, = m3.size
    #m3.show() #This is the last show where it works
    initWhite = [0,0]

    for x in range(width):
        for y in range(height):
            r, g, b = m3.getpixel((x, y))
            if r != 0 or g != 0 or b != 0:
                initWhite = [x,y]
                break
        if(initWhite != [0,0]):
            break   

    listOfEdgePixels.append(initWhite)
    #pixels[initWhite[0], initWhite[1]] = (68,214,44)
    m3.save("imgs\m2.png")
    #print(initWhite)
    m2 = cv2.imread("imgs\m2.png")
    cropped_image = m2[0:height, (initWhite[0])-1:width]
    #print(width)
    #print(height)
    cv2.imwrite("imgs\m2.png", cropped_image)

    m3 = Image.open("imgs\m2.png")
    pixels = m3.load()#Somehow gives attribute size

    width, height, = m3.size
    #m3.show()
    initWhite = [0,0]

    for y in range(height):
        for x in range(width):
            r, g, b = m3.getpixel((x, y))
            if r != 0 or g != 0 or b != 0:
                initWhite = [x,y]
                break
        if(initWhite != [0,0]):
            break   

    listOfEdgePixels.append(initWhite)
    #pixels[initWhite[0], initWhite[1]] = (68,214,44)
    m3.save("imgs\m2.png")
    #print(initWhite)
    m2 = cv2.imread("imgs\m2.png")
    cropped_image = m2[initWhite[1]-1:height, 0:width]
    #print(width)
    #print(height)
    cv2.imwrite("imgs\m2.png", cropped_image)
    m3 = Image.open("imgs\m2.png")
    pixels = m3.load()#Somehow gives attribute size
    #m3.show()
    width, height, = m3.size
    #m3.show()
    initWhite = [0,0]

    for x in range(width-1, -1, -1):
        for y in range(height -1, -1, -1):
            r, g, b = m3.getpixel((x, y))
            if r != 0 or g != 0 or b != 0:
                initWhite = [x,y]
                break
        if(initWhite != [0,0]):
            break   

    listOfEdgePixels.append(initWhite)
    #pixels[initWhite[0], initWhite[1]] = (68,214,44)
    m3.save("imgs\m2.png")
    #print(initWhite)
    m2 = cv2.imread("imgs\m2.png")
    cropped_image = m2[0:height, 0:initWhite[0] + 2]
    #print(width)
    #print(height)
    cv2.imwrite("imgs\m2.png", cropped_image)


    m3 = Image.open("imgs\m2.png")
    pixels = m3.load()#Somehow gives attribute size

    width, height, = m3.size
    #m3.show()
    initWhite = [0, 0]
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            r, g, b = m3.getpixel((x, y))
            if r != 0 or g != 0 or b != 0:
                initWhite = [x, y]
                break
        if initWhite != [0, 0]:
            break

    listOfEdgePixels.append(initWhite)
    #pixels[initWhite[0], initWhite[1]] = (68, 214, 44)
    m3.save("imgs\m2.png")
    #print("Bottom edge pixel:", initWhite)

    m2 = cv2.imread("imgs\m2.png")
    #+1 adds so gap between this and bottom.
    cropped_image = m2[0:initWhite[1]+2, 0:width]
    cv2.imwrite("imgs\m2.png", cropped_image)

    
    m3 = Image.open("imgs\m2.png")
    pixels = m3.load() #gives attribute's size

    width, height, = m3.size
    print(width)
    print(height)
    if(width != height):
        if height < width:
            while (height != width):
                #longer = Image.new("RGB", (width, height + 1), (68, 214, 44))  #neon green row
                longer = Image.new("RGB", (width, height + 1), (0, 0, 0))  #black row
                longer.paste(m3, (0, 0))
                m3 = longer
                height = height + 1
        if width < height:
            while (height != width):
                #wider = Image.new("RGB", (width + 1, height), (68, 214, 44))  #neon green column
                wider = Image.new("RGB", (width + 1, height), (0, 0, 0))  #black column

                wider.paste(m3, (0, 0)) 
                m3 = wider
                width = width + 1
    m3.save("imgs\m2.png")

    pixels = m3.load() #gives attribute's size

    width, height, = m3.size
    print(width)
    print(height)
    m3 = Image.open("imgs\m2.png")
    m3 = m3.resize((28,28),Image.BICUBIC)#Reduces image size to zero considers surrounding 4 pixels to determine new image values.
    m3.save("imgs\m2.png")

    width, height, = m3.size
    print(width)
    print(height)


    m3 = Image.open("imgs\m2.png")
    m3 = m3.transpose(Image.FLIP_LEFT_RIGHT)
    m3.save("imgs\m2.png")
    #m3.show()

    m3 = Image.open("imgs\m2.png")
    m3 = m3.rotate(90)
    m3.save("imgs\m2.png")

    #m2 = cv2.imread("imgs\m2.png")
    #m2 = cv2.GaussianBlur(m2, (5, 5), 1)
    #cv2.imwrite("imgs\m2.png", m2)
    img = Image.open("imgs\m2.png").convert('L')  # L makes gray scale only 2 nums not 3 like RGB
    img.save("imgs\m2.png")

if __name__ == "__main__":

    imPathM = "imgs\m.jpg"
    imPathM = "letters\letter0.png"
    run(imPathM)

