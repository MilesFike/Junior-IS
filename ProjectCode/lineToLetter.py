from PIL import Image
from PIL import ImageOps#Helps create original black but textured background with color import
import cv2 #cv2 used for more background conversions
import numpy 
import time
import os #to store the images
def run(imPathM):
    m2 = cv2.imread(imPathM)
    #converts to gray scale
    gray = cv2.cvtColor(m2, cv2.COLOR_BGR2GRAY)

    #Pixels above threshold become white. These are then replaced with black
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    black_background = numpy.zeros_like(m2)

    result = cv2.bitwise_and(m2, m2, mask=mask) + black_background
    #cv2.imshow("m2", m2)
    #checks for all black pixels

    cv2.imwrite("imgs/m2.png", result)

    img = Image.open("imgs/m2.png")
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
    img.save("imgs/m2.png")

    return cv2.imread("imgs/m2.png", cv2.IMREAD_GRAYSCALE)
    #num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
def segment_letters(orimg, img, output_dir):
    os.makedirs("letters", exist_ok=True)

    img1 = Image.open(img)
    #collecting data for iteration
    pixels = img1.load()
    width, height, = img1.size
    letterList = [] #list for ranges of letters
    storeRange = []

    temp = False

    for x in range(width):
        whiteHere = any(pixels[x,y] != (0,0,0) for y in range(height))
        
        if whiteHere and not temp:
            storeRange.append(x-1)
            temp = True

        if whiteHere and temp:
            continue

        if not whiteHere and temp:
             temp = False
             storeRange.append(x+1)
             letterList.append(storeRange)
             storeRange = []

    i = 0

    img1 = Image.open(orimg)
    #collecting data for iteration
    pixels = img1.load()
    width, height, = img1.size
    for pair in letterList:
        begin = pair[0]
        end = pair[1]
        crop = img1.crop((begin, 0, end, height))  # (left, top, right, bottom)
        if((end - begin) > 10):
            crop.save(os.path.join(output_dir, f"letter{i}.png"))
            i += 1
    print(letterList)
                
    #for i in range(len(letterList)):
        #output_path = os.path.join(output_dir, f"letter_{letter_idx}.png")
        #cv2.imwrite(output_path, letter_crop)
        #letter_idx += 1
    #print(f"Saved {letter_idx} letter images to '{output_dir}'")

    return i

#def makeLetters(imPath):
#    img = cv2.imread('/directorypath/image.bmp')
#    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

if __name__ == "__main__":

    imPathM = "imgs/milesFike.png"
    #imPathM = r"C:\Users\Glast\Desktop\Junior-IS\imgs\milesFike.png"

    im = run(imPathM)
    segment_letters("imgs/milesFike.png","imgs/m2.png", output_dir="letters")

    
