from openai import OpenAI
import base64
import os
import time
import random
import bpy

"""
OpenAI Image Generation Component for OpenMats Addon

This module provides utility functions to generate images using:
- DALL·E 3 (returns image URL)
- OpenAI Images API (returns local saved image)

Used internally by the main addon operator to process prompts.

Requires an OpenAI API key.
"""

# Function for DALL-E-3 Integration
def generate_response(user_input, user_api_key, model):
    client = OpenAI(
    api_key=user_api_key
    )

    # Reformat Model Input to be fit needs of Response Call
    if model == 'DALLE3':
        model = 'dall-e-3'
    elif model == 'DALLE2':
         model == 'dall-e-2'

    print("STARTED REQUEST: " + str(user_input))
    print(f"Called OpenAI Via {model}")
    
    response = client.images.generate(
        model = "dall-e-3",
        prompt= (user_input),
        size = "1024x1024",
        quality="standard",
        n=1
    )
        
    image_url = response.data[0].url

    print (image_url)

    return image_url

# Function for Images API Integraion
def generate_response_images(user_input, user_api_key):
    client = OpenAI(
    api_key=user_api_key
    )

    print("STARTED REQUEST: " + str(user_input))
    print("Called OpenAI Via gpt-image-1")

    response = client.images.generate(
        model = "gpt-image-1",
        prompt= (user_input),
        size = "1024x1024",
        quality="high",
        n=1
    )
    
    # Decode Image into base64
    image_base64 = response.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    ####### Download images #########

    # Paths
    folder_dir = os.path.dirname(bpy.data.filepath)
    texture_path = os.path.join(folder_dir, 'textures')
    
    if not (os.path.exists(texture_path)):
        # Create New Folder based on file path in if statement
        os.makedirs(texture_path)

    # Joining Directories
    save_dir = os.path.join(texture_path, str(time.time()*1000 * random.randint(1,34)) + ".png")

    # Save File
    with open(save_dir, "wb") as file:
            file.write(image_bytes)
    
    # Ensure File Path to Image Exists after download
    if os.path.exists(save_dir):
        print(save_dir)    
        return save_dir
    else:
        return None