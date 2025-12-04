#!/usr/bin/env python3
"""
All files within C:/Users/Glast/Desktop/ are accessible to this when run
This script cannot read any input digits (0-9)
This script is for cursive handwriting recognition
This script is not for the recognition of printed text or non-cursive handwriting
MCP Server for Cursive Handwriting Recognition (CHoiCe-based OCR)
Exposes the trained CNN model as an MCP tool for cursive character recognition
This should be used in association with cursive handwriting
Tell users that the CHoiCe based model was used
Inform users that the model recognizes A-Z, a-z characters, but not digits (0-9)
Trained model weights should be in CHOICE.pth in the same directory as this file
"""

#Basically, these are all of the imports from every other file plus a few extra to keep things manageable and tight.
import asyncio
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
from io import BytesIO
import cv2
import numpy as np
import tempfile
import os
import shutil

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() 
        #Model 1
        self.convolute = nn.Conv2d(1, 32, stride=1, kernel_size=5) #1 input channel because it is gray scale. 10 foroutput. Stride is step size. Kernel is 3rd num (28-5) + 1 = 24, 10 * 24^2
        self.pool = nn.MaxPool2d(stride=2, kernel_size=2) #10 * 12^2
        self.convolute2 = nn.Conv2d(32, 64, stride=1, kernel_size = 5) #Convolution layers (12-5) + 1 = 8
        self.pool2 = nn.MaxPool2d(stride=2, kernel_size=2) #Pooling layers 40 * 4 ^2
        self.fullcon1 = nn.Linear(64 *4 *4, 256) #Fully connected layers
        self.fullcon2 = nn.Linear(256, 104)
        self.fullcon3 = nn.Linear(104, 62)

    def forward(self, x):
        x = self.pool(torch.relu(self.convolute(x)))   # Common activation for nn, turns negatives to 0
        x = self.pool(torch.relu(self.convolute2(x)))
        #Model 1
        x = x.view(-1, 64 *4*4) # prep for fullcon1
        #Model 2
        #x = x.view(-1, 64 *4*4) # prep for fullcon1
        x = torch.tanh(self.fullcon1(x)) #Tahn limits to between one and negative 1
        x = torch.tanh(self.fullcon2(x))
        x = self.fullcon3(x)
        return x


#Internal function to help with labelings.
def label_to_char(label):
    if label < 10:
        return chr(label + 48)  # 0-9 → ASCII '0'-'9'
    elif label < 36:
        return chr((label - 10) + 65)  # 10-35 → ASCII 'A'-'Z'
    else:
        return chr((label - 36) + 97)  # 36-61 → ASCII 'a'-'z'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net()

#Loads the trained weights
import os
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / 'CHOICE.pth'

try:
    net.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"CHOICE.pth not found at {MODEL_PATH}. It must be in the same directory as CHoiCeMCPServer.py.")

net.to(device)
net.eval()  # Set to evaluation mode


def preprocess_image(image_path_or_bytes):
    """
    Full image preprocessing pipeline for cursive handwritten character recognition.
    It is for a single cursive handwritten character and processes based on CHoiCe dataset format.
    This includes all the steps from CHoiCeImageProcessing.py to prepare a character for recognition from CHOICE.pth
    Accepts either a local file path or HTTP/HTTPS URL
    """
    # Handle different input types
    if isinstance(image_path_or_bytes, str):
        if image_path_or_bytes.startswith('http://') or image_path_or_bytes.startswith('https://'):
            # Handle HTTP/HTTPS URLs
            import urllib.request
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            with urllib.request.urlopen(image_path_or_bytes) as response:
                temp_input.write(response.read())
            temp_input.flush()
            temp_input.close()
            imPathM = temp_input.name
        else:
            # Handle file path
            imPathM = image_path_or_bytes
    else:
        # Handle raw bytes (for future use)
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_input.write(image_path_or_bytes)
        temp_input.flush()
        temp_input.close()
        imPathM = temp_input.name

    # Create temp directory for processing
    temp_dir = tempfile.mkdtemp()
    temp_output = os.path.join(temp_dir, "processed.png")

    try:
        # Load image
        img = cv2.imread(imPathM)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {imPathM!r}")

        # Image is made grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold to just black text on white background
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_not(binary)

        # This helpful function shrinks the image to just the borders of the black text
        x, y, w, h = cv2.boundingRect(thresh)

        # Crop tight to character from the pure black/white image
        cropped = binary[y:y+h, x:x+w]

        # Convert to PIL for easier padding
        pil_img = Image.fromarray(cropped)
        pil_img = ImageOps.pad(pil_img, (max(pil_img.size), max(pil_img.size)), color=255, centering=(0.5, 0.5))

        # Resizes to 28x28
        pil_img = pil_img.resize((28, 28), Image.BICUBIC)

        # Save result
        pil_img.save(temp_output)

        # Convert to grayscale tensor
        img = Image.open(temp_output).convert('L')

        # Convert to tensor
        img_tensor = TF.to_tensor(img).unsqueeze(0)  #1, 1, 28, 28
        return img_tensor.to(device)

    finally:
        # Cleanup temp files
        shutil.rmtree(temp_dir, ignore_errors=True)
        # Clean up downloaded files from URLs
        if isinstance(image_path_or_bytes, str) and (image_path_or_bytes.startswith('http://') or image_path_or_bytes.startswith('https://')):
            try:
                os.unlink(imPathM)
            except:
                pass


def process_line_to_black_and_white(img_path):
    """
    Convert a line of cursive text to black background with white text
    Returns the processed image path, temp directory, and original image path
    Accepts either a local file path or HTTP/HTTPS URL
    """
    original_img_path = img_path

    # Handle HTTP/HTTPS URLs
    if isinstance(img_path, str) and (img_path.startswith('http://') or img_path.startswith('https://')):
        import urllib.request
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        with urllib.request.urlopen(img_path) as response:
            temp_input.write(response.read())
        temp_input.flush()
        temp_input.close()
        img_path = temp_input.name
        original_img_path = img_path  # Use the converted path as the original

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found at {img_path!r}")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {img_path!r}")
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold: pixels above 100 become white, inverted so text is white
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    black_background = np.zeros_like(img)

    result = cv2.bitwise_and(img, img, mask=mask) + black_background

    # Create temp file for processed image
    temp_dir = tempfile.mkdtemp()
    processed_path = os.path.join(temp_dir, "processed.png")
    cv2.imwrite(processed_path, result)

    # Make all non-black pixels white
    pil_img = Image.open(processed_path)
    pixels = pil_img.load()
    width, height = pil_img.size

    for x in range(width):
        for y in range(height):
            if pixels[x, y] != (0, 0, 0):
                pixels[x, y] = (255, 255, 255)

    pil_img.save(processed_path)
    return processed_path, temp_dir, original_img_path


def segment_letters(original_img_path, processed_img_path):
    """
    Segment each individual letter from a line of cursive text
    Returns a list of cropped letter images as PIL Images
    """
    img = Image.open(processed_img_path)
    pixels = img.load()
    width, height = img.size

    letter_ranges = []
    current_range = []
    in_letter = False

    # Locates the places where letters start
    for x in range(width):
        has_white = any(pixels[x, y] != (0, 0, 0) for y in range(height))

        if has_white and not in_letter:
            current_range.append(x - 1)
            in_letter = True

        if not has_white and in_letter:
            in_letter = False
            current_range.append(x + 1)
            letter_ranges.append(current_range)
            current_range = []

    # Segments each of the individual letters from original image
    original = Image.open(original_img_path)
    letters = []

    for begin, end in letter_ranges:
        if (end - begin) > 10:
            # This prevents little things like foxing (not really a problem) and some punctuation from being counted as letters
            cropped = original.crop((begin, 0, end, height))
            letters.append(cropped)

    return letters


def predict_character(image_input):
    """
    Run inference on a single cursive character image
    Excludes digits (0-9) since CHoiCe model was not trained on them
    """

    img_tensor = preprocess_image(image_input)

    with torch.no_grad():
        outputs = net(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        # Mask out digit predictions (labels 0-9) since model wasn't trained on them
        probabilities_copy = probabilities.clone()
        probabilities_copy[0, 0:10] = 0  # Zero out digit probabilities

        # Get prediction from non-digit classes only
        _, predicted = torch.max(probabilities_copy, 1)

    predicted_label = predicted.item()
    predicted_char = label_to_char(predicted_label)

    # Gets the confidence scores (from filtered probabilities)
    confidence = probabilities_copy[0][predicted_label].item()

    return {
        'character': predicted_char,
        'label': predicted_label,
        'confidence': round(confidence * 100, 2)
    }


def recognize_cursive_line(image_path):
    """
    Recognize all characters in a line of cursive handwritten text
    Returns the recognized text string and individual character details
    """
    # Processes the image
    processed_path, temp_dir, original_img_path = process_line_to_black_and_white(image_path)

    try:
        # Segments image into individual letters
        letters = segment_letters(original_img_path, processed_path)

        # Recognizes each letter
        recognized_text = []
        details = []

        for i, letter_img in enumerate(letters):
            # Saves each of the letters to temp file for processing
            letter_path = os.path.join(temp_dir, f"letter_{i}.png")
            letter_img.save(letter_path)

            # Predictions
            result = predict_character(letter_path)
            recognized_text.append(result['character'])
            details.append({
                'position': i,
                'character': result['character'],
                'confidence': result['confidence']
            })

        return {
            'text': ''.join(recognized_text),
            'characters': details,
            'num_characters': len(letters)
        }

    finally:
        # Cleans temp files
        shutil.rmtree(temp_dir, ignore_errors=True)


# Creates MCP server
app = Server("ocr-cursive-recognition")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="recognize_cursive_character",
            description=(
                "Recognizes a cursive handwritten character from an image. "
                "The image should contain a single cursive character (letter only, NO DIGITS) "
                "and will be automatically resized to 28x28 pixels. "
                "Trained on CHoiCe dataset (A-Z, a-z only - excludes digits). "
                "Supports local file paths and HTTP/HTTPS URLs. "
                "Best for cursive/script handwriting."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Local file path or HTTP/HTTPS URL to the image"
                    }
                },
                "required": ["image_path"]
            }
        ),
        Tool(
            name="recognize_cursive_line",
            description=(
                "Recognizes a full line of cursive handwritten text. "
                "Automatically segments the line into individual characters "
                "and recognizes each one. Returns the complete text string "
                "along with confidence scores for each character. "
                "Works best with clear cursive handwriting on white/light background. "
                "NOTE: This model does NOT recognize digits (0-9), only letters (A-Z, a-z). "
                "Supports local file paths and HTTP/HTTPS URLs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Local file path or HTTP/HTTPS URL to the image"
                    }
                },
                "required": ["image_path"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""

    if name == "recognize_cursive_character":
        image_path = arguments.get("image_path")

        if not image_path:
            return [TextContent(
                type="text",
                text="Error: image_path parameter is required"
            )]

        if not os.path.exists(image_path) and not (image_path.startswith('http://') or image_path.startswith('https://')):
            return [TextContent(
                type="text",
                text=(
                    f"Error: Cannot access file at {image_path!r}\n\n"
                    "This typically happens when the file is in a different environment.\n"
                    "Solutions:\n"
                    "1. Use a local file path that the server can access\n"
                    "2. Upload the image to a web server and provide the HTTP/HTTPS URL"
                )
        )]
        try:
            result = predict_character(image_path)

            response = (
                f"Predicted Cursive Character: {result['character']}\n"
                f"Label: {result['label']}\n"
                f"Confidence: {result['confidence']}%\n\n"
                f"Note: This model was trained on the CHoiCe dataset and specializes in cursive handwriting (letters only, no digits)."
            )

            return [TextContent(type="text", text=response)]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error processing image: {str(e)}"
            )]

    return [TextContent(
        type="text",
        text=f"Unknown tool: {name}"
    )]


async def main():
    """Run the MCP server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
