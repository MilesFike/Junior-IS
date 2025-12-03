#!/usr/bin/env python3
"""
MCP Server for Handwriting Recognition (EMNIST-based OCR)
Exposes the trained CNN model as an MCP tool for character recognition
"""

#Basically, these are all of the imports from every other file plus a few extra to keep things manageable and tight.
import asyncio
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
import base64
from io import BytesIO
import cv2
import numpy as np
import tempfile
import os
import shutil


# Define the neural network architecture (must match trained model)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # CRITICAL: Must call parent class constructor!
        
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


# Initialize model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net()

# Load trained weights
try:
    net.load_state_dict(torch.load('EMNIST.pth', map_location=device))
    print("Model taken from EMNIST.pth")
except FileNotFoundError:
    print("EMNIST.pth not found. It must be in the same directory.")

net.to(device)
net.eval()  # Set to evaluation mode


def preprocess_image(image_path_or_bytes):
    """
    Full image preprocessing pipeline for handwritten character recognition.
    It is for a single handwritten character and processes based on the constraints of the EMNIST dataset.
    This includes all the steps from imageProcessing.py to prepare a character for recognition from EMNIST.pth
    Accepts either a file path or base64 encoded bytes
    """
    # Handle different input types
    if isinstance(image_path_or_bytes, str):
        if image_path_or_bytes.startswith('data:image'):
            # Handle base64 data URL
            header, encoded = image_path_or_bytes.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_input.write(image_bytes)
            temp_input.close()
            imPathM = temp_input.name
        else:
            # Handle file path
            imPathM = image_path_or_bytes
    else:
        # Handle bytes
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_input.write(image_path_or_bytes)
        temp_input.close()
        imPathM = temp_input.name
    
    # Create temp directory for processing
    temp_dir = tempfile.mkdtemp()
    temp_output = os.path.join(temp_dir, "processed.png")
    
    try:
        m2 = cv2.imread(imPathM)
        #Checks to see if m2 exists
        if m2 is None:
            raise FileNotFoundError(f"Could not read image at {imPathM!r}")
        
        #converts to gray scale
        gray = cv2.cvtColor(m2, cv2.COLOR_BGR2GRAY)
        #This extra Gaussian Blur reduces the background noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        #Pixels above threshold become white. These are then replaced with black
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        black_background = np.zeros_like(m2)

        result = cv2.bitwise_and(m2, m2, mask=mask) + black_background
        #checks for all black pixels

        cv2.imwrite(temp_output, result)

        img = Image.open(temp_output)
        #collecting data for iteration
        pixels = img.load()
        width, height = img.size
        for x in range(width):
            for y in range(height):
                currentColor = pixels[x,y]
                #print(currentColor) Do not uncomment this line
                if currentColor != (0, 0, 0):
                    pixels[x,y] = (255, 255, 255)
        
        width, height = img.size
        img.save(temp_output)

        m3 = Image.open(temp_output)
        pixels = m3.load() #gives attribute's size

        width, height = m3.size
        
        if(width != height):
            if height < width:
                while (height != width):
                    longer = Image.new("RGB", (width, height + 1), (0, 0, 0))  #black row
                    longer.paste(m3, (0, 0))
                    m3 = longer
                    height = height + 1
            if width < height:
                while (height != width):
                    wider = Image.new("RGB", (width + 1, height), (0, 0, 0))  #black column
                    wider.paste(m3, (0, 0)) 
                    m3 = wider
                    width = width + 1

        m3 = m3.resize((128,128))
        m3.save(temp_output)
        
        img = Image.open(temp_output)
        img.save(temp_output)
        #Transfering back to cv2 for Gaussian Blur
        m2 = cv2.imread(temp_output)
        m2 = cv2.GaussianBlur(m2, (5, 5), 1)
        cv2.imwrite(temp_output, m2)
        m3 = Image.open(temp_output)

        listOfEdgePixels = []
        m3 = Image.open(temp_output)
        pixels = m3.load()#Somehow gives attribute size

        width, height = m3.size
        
        img = cv2.imread(temp_output)

        #2 pixel padding
        padded = cv2.copyMakeBorder(img, top=2, bottom=2, left=2, right=2, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)) #It's black this time

        cv2.imwrite(temp_output, padded)
        m3 = Image.open(temp_output)
        pixels = m3.load()#Somehow gives attribute size

        width, height = m3.size
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
        m3.save(temp_output)
        m2 = cv2.imread(temp_output)
        cropped_image = m2[0:height, (initWhite[0])-1:width]
        cv2.imwrite(temp_output, cropped_image)

        m3 = Image.open(temp_output)
        pixels = m3.load()#Somehow gives attribute size

        width, height = m3.size
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
        m3.save(temp_output)
        m2 = cv2.imread(temp_output)
        cropped_image = m2[initWhite[1]-1:height, 0:width]
        cv2.imwrite(temp_output, cropped_image)
        m3 = Image.open(temp_output)
        pixels = m3.load()#Somehow gives attribute size
        
        width, height = m3.size
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
        m3.save(temp_output)
        m2 = cv2.imread(temp_output)
        cropped_image = m2[0:height, 0:initWhite[0] + 2]
        cv2.imwrite(temp_output, cropped_image)

        m3 = Image.open(temp_output)
        pixels = m3.load()#Somehow gives attribute size

        width, height = m3.size
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
        m3.save(temp_output)

        m2 = cv2.imread(temp_output)
        #+1 adds so gap between this and bottom.
        cropped_image = m2[0:initWhite[1]+2, 0:width]
        cv2.imwrite(temp_output, cropped_image)
        m2 = cv2.imread(temp_output)
        gray = cv2.cvtColor(m2, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(thresh)
        cropped = m2[y:y+h, x:x+w]
        cv2.imwrite(temp_output, cropped)
        
        m3 = Image.open(temp_output)
        m3 = m3.convert("L")
        m3 = ImageOps.pad(m3, (max(m3.size), max(m3.size)), color=0, centering=(0.5, 0.5))
        m3 = m3.resize((28, 28), Image.BICUBIC)

        m3.save(temp_output)

        m3 = Image.open(temp_output)
        m3 = m3.transpose(Image.FLIP_LEFT_RIGHT)
        m3.save(temp_output)

        m3 = Image.open(temp_output)
        m3 = m3.rotate(90)
        m3.save(temp_output)

        img = Image.open(temp_output).convert('L')  # L makes gray scale only 2 nums not 3 like RGB
        img.save(temp_output)
        
        # Convert to tensor
        img_tensor = TF.to_tensor(img).unsqueeze(0)  # [1, 1, 28, 28]
        return img_tensor.to(device)
    
    finally:
        # Cleanup temp files
        shutil.rmtree(temp_dir, ignore_errors=True)
        if isinstance(image_path_or_bytes, str) and image_path_or_bytes.startswith('data:image'):
            try:
                os.unlink(temp_input.name)
            except:
                pass


def process_line_to_black_and_white(img_path):
    """
    Convert a line of text to black background with white text
    Returns the processed image path
    """
    img = cv2.imread(img_path)
    
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
    return processed_path, temp_dir


def segment_letters(original_img_path, processed_img_path):
    """
    Segment each individual letter from a line of text
    Returns a list of cropped letter images as PIL Images
    """
    img = Image.open(processed_img_path)
    pixels = img.load()
    width, height = img.size
    
    letter_ranges = []
    current_range = []
    in_letter = False
    
    #Locates the places where letters start
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
            #This prevents little things like foxing and some punctuation from being counted as letters
            cropped = original.crop((begin, 0, end, height))
            letters.append(cropped)
    
    return letters


def predict_character(image_input):
    """Run inference on a single character image"""
    img_tensor = preprocess_image(image_input)
    
    with torch.no_grad():
        outputs = net(img_tensor)
        _, predicted = torch.max(outputs, 1)
    
    predicted_label = predicted.item()
    predicted_char = label_to_char(predicted_label)
    
    # Get confidence scores
    probabilities = torch.softmax(outputs, dim=1)
    confidence = probabilities[0][predicted_label].item()
    
    return {
        'character': predicted_char,
        'label': predicted_label,
        'confidence': round(confidence * 100, 2)
    }


def recognize_line(image_path):
    """
    Recognize all characters in a line of handwritten text
    Returns the recognized text string and individual character details
    """
    # Process the image
    processed_path, temp_dir = process_line_to_black_and_white(image_path)
    
    try:
        # Segment into individual letters
        letters = segment_letters(image_path, processed_path)
        
        # Recognize each letter
        recognized_text = []
        details = []
        
        for i, letter_img in enumerate(letters):
            # Save letter to temp file for processing
            letter_path = os.path.join(temp_dir, f"letter_{i}.png")
            letter_img.save(letter_path)
            
            # Predict
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
        # Cleanup temp files
        shutil.rmtree(temp_dir, ignore_errors=True)


# Create MCP server
app = Server("ocr-handwriting-recognition")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="recognize_character",
            description=(
                "Recognizes a handwritten character from an image. "
                "The image should contain a single character (letter or digit) "
                "and will be automatically resized to 28x28 pixels. "
                "Trained on EMNIST dataset (0-9, A-Z, a-z)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file containing a handwritten character, or base64 data URL"
                    }
                },
                "required": ["image_path"]
            }
        ),
        Tool(
            name="recognize_line",
            description=(
                "Recognizes a full line of handwritten text. "
                "Automatically segments the line into individual characters "
                "and recognizes each one. Returns the complete text string "
                "along with confidence scores for each character. "
                "Works best with clear handwriting on white/light background."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file containing a line of handwritten text"
                    }
                },
                "required": ["image_path"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    
    if name == "recognize_character":
        image_path = arguments.get("image_path")
        
        if not image_path:
            return [TextContent(
                type="text",
                text="Error: image_path parameter is required"
            )]
        
        try:
            result = predict_character(image_path)
            
            response = (
                f"Predicted Character: {result['character']}\n"
                f"Label: {result['label']}\n"
                f"Confidence: {result['confidence']}%"
            )
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error processing image: {str(e)}"
            )]
    
    elif name == "recognize_line":
        image_path = arguments.get("image_path")
        
        if not image_path:
            return [TextContent(
                type="text",
                text="Error: image_path parameter is required"
            )]
        
        try:
            result = recognize_line(image_path)
            
            response = f"Recognized Text: {result['text']}\n\n"
            response += f"Number of Characters: {result['num_characters']}\n\n"
            response += "Character Details:\n"
            
            for char_detail in result['characters']:
                response += (
                    f"  Position {char_detail['position']}: "
                    f"{char_detail['character']} "
                    f"(confidence: {char_detail['confidence']}%)\n"
                )
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error processing line: {str(e)}"
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