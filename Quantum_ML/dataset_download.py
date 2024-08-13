import torch
from torchgeo.datasets import LandCoverAI
import glob
import matplotlib.pyplot  as plt
import cv2
import os
from skimage.transform import resize
import numpy as np

root  = "./Datasets/LandCover/"

## downloading dataset
dataset = LandCoverAI(root = root, download=True)
print("\n Dataset downloaded successfully!..\n")


## Add image cropping and splitting code here later

print("\n Reading original dataset and resizing images...")
## Creating a list of all images
DATA_ROOT = '../Datasets/LandCover/'

images_list = list(glob.glob(os.path.join(DATA_ROOT, "images", "*.tif")))
mask_list = list(glob.glob(os.path.join(DATA_ROOT, "masks", "*.tif")))

sample_set = []
sample_mask = []
for i, im in enumerate(images_list):
    curr_im = cv2.imread(im) / 255
    curr_im = resize(curr_im, (512,512), order=0, anti_aliasing=False, preserve_range=True)

    sample_set.append(curr_im)



for i, im in enumerate(mask_list):
    curr_im = cv2.imread(im)
    curr_im = resize(curr_im, (512,512), order=0, anti_aliasing=False, preserve_range=True)

    sample_mask.append(curr_im)



## Extracting patches
print("\n Extracting Patches....")
def extract_patches(image, mask, patch_size=(64, 64)):
    patches_image = []
    patches_mask = []
    height, width = image.shape[:2]
    patch_height, patch_width = patch_size
    
    for y in range(0, height - patch_height + 1, patch_height):
        for x in range(0, width - patch_width + 1, patch_width):
            patch_image = image[y:y+patch_height, x:x+patch_width]
            patch_mask = mask[y:y+patch_height, x:x+patch_width]
            patches_image.append(patch_image)
            patches_mask.append(patch_mask)
    
    return np.array(patches_image), np.array(patches_mask)

patched_images_X_train = []
patched_images_y_train = []

## loop over each image and append the patches to the array

for im, gt_im in zip(sample_set, sample_mask):
    patches_image, patches_mask = extract_patches(im, gt_im)

    for patch_im, patch_gt in zip(patches_image, patches_mask):

        patched_images_X_train.append(patch_im)
        patched_images_y_train.append(patch_gt)

print("Number of patches:", len(patched_images_X_train))
print("Shape of a patch (image):", patched_images_X_train[0].shape)
print("Shape of a patch (mask):", patched_images_y_train[0].shape)


## Saving Patches to different folders

print("\n Saving Patches...")
def save_images_and_masks(image_list, mask_list, image_dir='images', mask_dir='masks'):
    # Create directories if they don't exist
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for idx, (image, mask) in enumerate(zip(image_list, mask_list)):
        # Ensure the image is a uint8 numpy array
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Ensure the mask is a uint8 numpy array
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        # Ensure the image is in the correct range
        image = np.clip(image, 0, 255)
        mask = np.clip(mask, 0, 255)

        # Convert images to BGR if they are in RGB format
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

        # Define the file paths
        image_path = os.path.join(image_dir, f'image_{idx:04d}.png')
        mask_path = os.path.join(mask_dir, f'image_{idx:04d}.png')

        # Save the image and mask
        cv2.imwrite(image_path, image)
        cv2.imwrite(mask_path, mask)

    print(f'Saved {len(image_list)} images and {len(mask_list)} masks.')


save_images_and_masks(patched_images_X_train, patched_images_y_train)