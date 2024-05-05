from PIL import Image
import os
import torch
import os
from PIL import Image

import os
from PIL import Image
import torch

def resize_images(source_folder, target_folder):
    """
    Resize images in a source folder and save them to a target folder.

    Args:
        source_folder (str): Path to the source folder containing images.
        target_folder (str): Path to the target folder where resized images will be saved.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Recursive function to resize images in subfolders
    def resize_images_recursive(subfolder_path, target_subfolder):
        """
        Recursive function to resize images in subfolders.

        Args:
            subfolder_path (str): Path to the current subfolder.
            target_subfolder (str): Path to the target subfolder for resized images.
        """
        # Loop through each item in the subfolder
        for item in os.listdir(subfolder_path):
            item_path = os.path.join(subfolder_path, item)

            # Check if the item is a file
            if os.path.isfile(item_path):
                # Open the image using Pillow
                image = Image.open(item_path)

                # Resize the image to 256x256 pixels
                resized_image = image.resize((256, 256))

                # Save the resized image to the target subfolder
                resized_image.save(os.path.join(target_subfolder, item))

                # Close the image
                image.close()

                print('Image', item, 'resized')

            # Check if the item is a subfolder
            elif os.path.isdir(item_path):
                target_subfolder_path = os.path.join(target_subfolder, item)

                # Create the target subfolder if it doesn't exist
                if not os.path.exists(target_subfolder_path):
                    os.makedirs(target_subfolder_path)

                # Recursively call the function for the subfolder
                resize_images_recursive(item_path, target_subfolder_path)

    # Call the recursive function for the source folder
    resize_images_recursive(source_folder, target_folder)





def crop_center(image, crop_size):
    """
    Crop the center of an image to the specified size.

    Args:
        image (PIL.Image): Input image.
        crop_size (tuple): Size of the cropped image (width, height).

    Returns:
        PIL.Image: Cropped image.
    """
    width, height = image.size
    left = (width - crop_size[0]) // 2
    top = (height - crop_size[1]) // 2
    right = (width + crop_size[0]) // 2
    bottom = (height + crop_size[1]) // 2
    return image.crop((left, top, right, bottom))


def process_images_in_folder(folder_path, output_path, crop_size):
    """
    Process images in a folder by cropping them to the specified size and saving them to an output folder.

    Args:
        folder_path (str): Path to the input folder containing images.
        output_path (str): Path to the output folder where cropped images will be saved.
        crop_size (tuple): Size of the cropped images (width, height).
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Traverse through the input folder and its subfolders
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                output_subfolder = os.path.relpath(root, folder_path)
                output_subfolder_path = os.path.join(output_path, output_subfolder)
                if not os.path.exists(output_subfolder_path):
                    os.makedirs(output_subfolder_path)

                # Open the image
                image = Image.open(image_path)

                # Crop the image
                cropped_image = crop_center(image, crop_size)

                # Define the output image path
                output_image_path = os.path.join(output_subfolder_path, file)

                # Save the cropped image
                cropped_image.save(output_image_path)

                # Print the information
                print(f"Cropped {image_path} to {output_image_path}")