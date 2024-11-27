import os
import cv2
import albumentations as A
import glob

# Define the paths to your class folders
base_dir = r'C:\Users\Brenda\Documents\COS30049 Computing Technology Innovation Project\output'
augmented_base_dir = r'C:\Users\Brenda\Documents\COS30049 Computing Technology Innovation Project\augmented_images2'  # New directory for augmented images

classes_to_augment = [
    "monitor lizard", "asian black hornbill", "small-toothed palm civet",
    "white fronted langur", "teledu", "three-striped ground squirrel",
    "blue banded pitta", "clouded leopard", "banded linsang",
    "roughneck monitor lizard", "bird sp", "orangutan", "porcupine sp",
    "babbler s.p", "bornean crested fireback", "ground squirrel",
    "kinabalu squirrel", "banded langur", "dog", "pigeon sp",
    "slow lorris", "dove s.p", "otter civet", "sunda pangolin",
    "white-rumped shama", "malayan weasel"
]

# Define the augmentation pipeline
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.5)
])

# Function to augment images in a specific class
def augment_class_images(class_name):
    class_dir = os.path.join(base_dir, class_name)
    output_dir = os.path.join(augmented_base_dir, class_name)  # Save augmented images in a new folder structure
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all images in the class directory
    image_paths = glob.glob(os.path.join(class_dir, '*.jpg'))  # Change the extension if needed

    for img_path in image_paths:
        # Read the image
        image = cv2.imread(img_path)

        # Generate 2 augmented images for each original image
        for i in range(2):
            # Perform augmentation
            augmented_image = augmentation_pipeline(image=image)['image']
            
            # Save the augmented image with a unique name
            img_name = os.path.splitext(os.path.basename(img_path))[0]  # Get the base name without extension
            output_path = os.path.join(output_dir, f'{img_name}_aug_{i+1}.jpg')
            cv2.imwrite(output_path, augmented_image)

    print(f'Augmented images for {class_name} saved in {output_dir}')

# Loop through each class and augment images
for class_name in classes_to_augment:
    augment_class_images(class_name)
