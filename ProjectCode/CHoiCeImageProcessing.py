from PIL import Image, ImageOps
import cv2
import numpy
import os
from pathlib import Path


#These directories prevent errors as file management becomes much more complex.
BASE_DIR = Path(__file__).parent
IMGS_DIR = BASE_DIR / "imgs"
LETTERS_DIR = BASE_DIR / "letters"

def process(imPathM):
    #This creates black text on a white background and does not alter orientation or mirror images
    # Load image
    img = cv2.imread(imPathM)
    #Image is made grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to just (230) gray text on white background
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_not(binary)
    #This helpful function shrinks the image to just the borders of the black text
    x, y, w, h = cv2.boundingRect(thresh)

    # Crop tight to character from the pure black/white image
    cropped = binary[y:y+h, x:x+w]

    #Convert to PIL for easier padding
    pil_img = Image.fromarray(cropped)
    pil_img = ImageOps.pad(pil_img, (max(pil_img.size), max(pil_img.size)), color=255, centering=(0.5, 0.5))

    #resizes to 28x28
    pil_img = pil_img.resize((28, 28), Image.BICUBIC)

    #Save result
    output_path = str(IMGS_DIR / "m2.png")
    pil_img.save(output_path)

    return output_path

if __name__ == "__main__":
    imPathM = str(LETTERS_DIR / "letter4.png")
    result = process(imPathM)
    print(f"Processed image saved to: {result}")

    # Showing the result
    img = Image.open(result)
    img.show()
