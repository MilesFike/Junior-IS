from PIL import Image
from PIL import ImageOps#Helps create original black but textured background with color import
import cv2 #cv2 used for more background conversions
import numpy 
import time
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
    img.save("imgs\m2.png")


    #num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

if __name__ == "__main__":

    imPathM = "imgs\milesFike.png"
    run(imPathM)

    
