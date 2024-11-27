import os
import random
import shutil
from collections import defaultdict
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
import numpy as np

# Define paths
dataset_path = r'C:\Users\Brenda\Documents\COS30049 Computing Technology Innovation Project\COS30049_camera_trap_images'  # Change this to your dataset path
output_path = r'C:\Users\Brenda\Documents\COS30049 Computing Technology Innovation Project\output'  # Change this to your desired output path

# Initialize lists to hold image paths
species_images = {}

# Define similar species mapping
similar_species_mapping = {
    'binturong': 'bearcat',
    'blue banded pitta': 'blue banded pitta',
    'blue-banded pita': 'blue banded pitta',
    'bulwer phesant': 'bulwer phesant',
    "bulwer's pheasant": "bulwer pheasant",
    'great argus': 'great argus pheasant',
    'great argus pheasant': 'great argus pheasant',
    'great argus pheasent': 'great argus pheasant',
    'horse squirrel': 'horse tail squirrel',
    'long tailed macaque': 'long-tailed macaque',
    'long tail macaque': 'long-tailed macaque',
    'long tail macque': 'long-tailed macaque',
    'long tailed macque': 'long-tailed macaque',
    'long-tail macque': 'long-tailed macaque',
    'long-tailed macaque': 'long-tailed macaque',
    'long tailed porcupine': 'long-tailed porcupine',
    'Long-tailed porcupine': 'long-tailed porcupine',
    'long tail porcupine': 'long-tailed porcupine',
    'long-tailed porcupine': 'long-tailed porcupine',
    'malayan porcupine': 'malayan porcupine',
    'malay porcupine': 'malayan porcupine',
    'malay civet': 'malayan civet',
    'malayan civet': 'malayan civet',
    'marble cat': 'marbled cat',
    'marbled cat': 'marbled cat',
    'mask palm civet': 'masked palm civet',
    'masked palm civet': 'masked palm civet',
    'masked palm civet': 'masked palm civet',
    'mongoose': 'mongoose',
    'mongoose sp': 'mongoose',
    'mangoose sp': 'mongoose',
    'mousedeer': 'mousedeer',
    'mousedeer s.p': 'mousedeer',
    'mousedeer sp': 'mousedeer',
    'Mousedeer sp': 'mousedeer',
    'mouse sp': 'mice sp',
    'munjac': 'muntjac',
    'muntjac': 'muntjac',
    'muntjac s.p': 'muntjac',
    'muntjac sp': 'muntjac',
    'muntjact sp': 'muntjac',
    'muntjact': 'muntjac',
    'munjact': 'muntjac',
    'munjact sp': 'muntjac',
    'palm civet': 'palm civet',
    'palm civet s.p': 'palm civet',
    'pig tail macque': 'pig-tailed macaque',
    'pig tailed macaque': 'pig-tailed macaque',
    'pig tailed macque': 'pig-tailed macaque',
    'Pig-tailed macaque': 'pig-tailed macaque',
    'rat s.p': 'rat',
    'rat sp': 'rat',
    'roughneck monitor': 'roughneck monitor lizard',
    'roughneck monitor lizard': 'roughneck monitor lizard',
    'roul roul': 'roulroul',
    'crested partridge': 'roulroul',
    'samba deer': 'sambar deer',
    'sambar deer': 'sambar deer',
    'sambar deer s.p': 'sambar deer',
    'small toothed palm civet': 'small-toothed palm civet',
    'small-toothed palm civet': 'small-toothed palm civet',
    'squirrel s.p': 'squirrel',
    'squirrel sp': 'squirrel',
    'sun bear': 'sunbear',
    'sunbear': 'sunbear',
    'thick spined porcupine': 'thick-spined porcupine',
    'thicked spined porcupine': 'thick-spined porcupine',
    'Thick-spine porcupine': 'thick-spined porcupine',
    'thicked spine porcupine': 'thick-spined porcupine',
    'malayan porcupine': 'thick-spined porcupine',
    'treeshew sp': 'treeshrew',
    'treeshrew s.p': 'treeshrew',
    'treeshrew sp': 'treeshrew',
    'treshew sp': 'treeshrew',
    'treshrew sp': 'treeshrew',
    'tuffed squirrel': 'tufted ground squirrel',
    'tufted squirrel': 'tufted ground squirrel',
    'tufted ground squirrel': 'tufted ground squirrel',
    'ground tufted squirrel': 'tufted ground squirrel',
    'yellow throated marteen': 'yellow-throated marten',
    'yellow throated marten': 'yellow-throated marten',
    'yellow throted marten': 'yellow-throated marten',
    'yellow throten marten': 'yellow-throated marten',
    'yellow-throated marten': 'yellow-throated marten',
}

# Augmentation definition
aug_seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # Horizontal flip 50% of the time
    iaa.Flipud(0.2),  # Vertical flip 20% of the time
    iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),  # Add Gaussian noise
    iaa.Multiply((0.8, 1.2)),  # Change brightness
    iaa.Affine(rotate=(-25, 25)),  # Rotate images
])

# Loop through site folders
for site_folder in os.listdir(dataset_path):
    site_path = os.path.join(dataset_path, site_folder)
    
    if os.path.isdir(site_path):
        print(f"Processing site folder: {site_folder}")  # Print site folder name
        filter_folder = os.path.join(site_path, 'Filter')
        
        if os.path.exists(filter_folder):
            for species_folder in os.listdir(filter_folder):
                # Check for unidentified species
                if any(term in species_folder.lower() for term in ['undentified', 'unidentified', 'unidentify sp', 'unidentifed', 'mix']):
                    print(f"Skipping unidentified species folder: {species_folder}")
                    continue  # Skip this species folder
                
                print(f"Found species folder: {species_folder}")  # Add this line
                species_path = os.path.join(filter_folder, species_folder)
                
                if os.path.isdir(species_path):
                    image_list = [os.path.join(species_path, img) for img in os.listdir(species_path) if img.lower().endswith(('jpg', 'jpeg', 'png'))]
                    
                    # Normalize species name based on the mapping
                    normalized_species = similar_species_mapping.get(species_folder.lower(), species_folder.lower())
                    species_images[normalized_species] = species_images.get(normalized_species, []) + image_list
                    print(f"Added {len(image_list)} images for {normalized_species}.")

# Prepare to collect images ensuring we have a max of 100 images per species
selected_images = set()  # Use a set to avoid duplicates

# Total images needed per species
max_images_per_species = 50

# Ensure at least one image from each species
for species, images in species_images.items():
    if len(images) > 0:
        if len(images) < max_images_per_species:
            # Collect all images for species with less than 100 images
            selected_images.update(images)
        else:
            # Select a maximum of 100 images for each species
            selected_from_species = random.sample(images, max_images_per_species)
            selected_images.update(selected_from_species)

# Create a list to hold images for output
output_images = list(selected_images)

# Shuffle output images to randomize distribution
random.shuffle(output_images)

# Create output directories for each species
for species in species_images.keys():
    os.makedirs(os.path.join(output_path, species), exist_ok=True)

# Move images to respective folders
for img_path in output_images:
    species_name = next((species for species, images in species_images.items() if img_path in images), None)
    if species_name:
        # Move original image to respective species folder
        shutil.copy(img_path, os.path.join(output_path, species_name, os.path.basename(img_path)))

print("Image collection complete.")
