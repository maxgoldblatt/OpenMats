import os
import requests
import bpy
import random
import time

"""
DALL·E Image Downloader - OpenMats Utility Module

Handles downloading and saving of image files from URLs returned by the
OpenAI DALL·E 3 API. Saves images to a 'textures' directory located relative
to the current .blend file.

Features:
- Verifies that the .blend file is saved before downloading
- Creates 'textures' folder if it doesn't exist
- Generates unique filenames based on timestamp

Used internally by OpenMats after DALL·E 3 responses return a direct URL.
"""

def download_files(url):

    if not bpy.data.filepath:
        print("The file is unsaved. Cannot download textures.")
        return None
    
    # Get File Directory
    folder_dir = os.path.dirname(bpy.data.filepath)
    texture_path = os.path.join(folder_dir, 'textures')
    print (folder_dir)

    if not (os.path.exists(texture_path)):
        # Create New Folder based on file path in if statement
        os.makedirs(texture_path)
    
    # Joining Directories
    save_dir = os.path.join(texture_path, str(time.time()*1000 * random.randint(1,34)) + ".png")
    
    # Send GET request
    response = requests.get(url)

    # Ensure file is loaded correctly
    if response.ok == False:
        print("An error has occured. Failed to download image.")
        return None
    else:
        print("Image Loaded")

        # Save File
        with open(save_dir, "wb") as file:
            file.write(response.content)
        print(f"Image saved as {save_dir}")
    
    # Ensure File Path to Image Exists after download
    if os.path.exists(save_dir):
        print(save_dir)    
        return save_dir
    else:
        return None