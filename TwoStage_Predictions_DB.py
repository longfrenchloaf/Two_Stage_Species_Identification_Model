import os
import cv2
import pytesseract
import mysql.connector
from datetime import datetime
import re
from ultralytics import YOLO  # Assuming you're using the YOLOv8 library
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn  # Ensure nn is imported for Linear layer
from PIL import Image
import time

# Define a dictionary to map class IDs to class names
class_mapping = {
    0: 'asian black hornbill',
    1: 'babbler',
    2: 'banded civet',
    3: 'banded langur',
    4: 'banded linsang',
    5: 'bay cat',
    6: 'bearcat',
    7: 'bearded pig',
    8: 'bird',
    9: 'blue banded pitta',
    10: 'bornean crested fireback',
    11: 'bornean ground cuckoo',
    12: 'bornean porcupine',
    13: 'bornean yellow muntjac',
    14: 'bulwer pheasant',
    15: 'clouded leopard',
    16: 'crested serpent eagle',
    17: 'dog',
    18: 'dove',
    19: 'emerald dove',
    20: 'great argus pheasant',
    21: 'ground squirrel',
    22: 'horse tail squirrel',
    23: 'human',
    24: 'kinabalu squirrel',
    25: 'leopard cat',
    26: 'long-tailed macaque',
    27: 'long-tailed porcupine',
    28: 'malayan civet',
    29: 'malayan weasel',
    30: 'marbled cat',
    31: 'maroon langur',
    32: 'masked palm civet',
    33: 'mice',
    34: 'mongoose',
    35: 'monitor lizard',
    36: 'moonrat',
    37: 'mousedeer',
    38: 'muntjac',
    39: 'orangutan',
    40: 'otter civet',
    41: 'palm civet',
    42: 'pangolin',
    43: 'pig-tailed macaque',
    44: 'pigeon',
    45: 'porcupine',
    46: 'rat',
    47: 'red muntjac',
    48: 'roughneck monitor lizard',
    49: 'roulroul',
    50: 'sambar deer',
    51: 'slow lorris',
    52: 'small-toothed palm civet',
    53: 'squirrel',
    54: 'sunbear',
    55: 'sunda pangolin',
    56: 'teledu',
    57: 'thick-spined porcupine',
    58: 'three-striped ground squirrel',
    59: 'treeshrew',
    60: 'tufted ground squirrel',
    61: 'white fronted langur',
    62: 'white-rumped shama',
    63: 'yellow-throated marten',
}

# Paths to models
yolov8_model_path = "C:/Users/Brenda/Documents/COS30049 Computing Technology Innovation Project/yolov8_resnet/best.pt"
resnet_model_path = "C:/Users/Brenda/Documents/COS30049 Computing Technology Innovation Project/yolov8_resnet/resnet50_model.pth"

# Dataset path
dataset_path = "C:/Users/Brenda/Documents/COS30049 Computing Technology Innovation Project/yolov8_resnet/COS30049_camera_trap_images"
cropped_image_save_path = "C:/Users/Brenda/Documents/COS30049 Computing Technology Innovation Project/yolov8_resnet/cropped_images"

# Load YOLOv8 model
yolo_model = YOLO(yolov8_model_path)

# Load the ResNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Determine device
resnet_model = models.resnet50(weights='IMAGENET1K_V1')  # Load ResNet50 with ImageNet weights
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 64)  # Ensure this matches your model's output
resnet_model.load_state_dict(torch.load(resnet_model_path, map_location='cpu'))
resnet_model.to(device)  # Move to the appropriate device
resnet_model.eval()  # Set the model to evaluation mode

# Define preprocessing for ResNet
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Connect to MySQL database
def connect_to_database():
    for attempt in range(3):
        try:
            connection = mysql.connector.connect(
                host='localhost',
                user='root',
                password='',  # Change this to your MySQL password
                database='semenggoh',
                autocommit=True  # Enable auto-reconnect
            )
            return connection
        except mysql.connector.Error as err:
            print(f"Attempt {attempt + 1} failed: {err}")
            if err.errno == mysql.connector.errorcode.ER_ACCESS_DENIED_ERROR:
                print("Access denied: Check your username and password.")
            elif err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist.")
            else:
                print("Database connection error.")
    return None  # If all attempts fail

# Initialize database connection and ensure connection is available
connection = connect_to_database()
if connection is None:
    raise Exception("Failed to connect to the database after multiple attempts.")

cursor = connection.cursor()

# Function to get species name from index
def get_species_name(class_index):
    return class_mapping.get(class_index, "Unknown")

# Function to preprocess the image for better OCR results
def preprocess_image_for_ocr(image_path):
    image = cv2.imread(image_path)
    # Resize the image if necessary
    scale_percent = 200  # Resize image to 200%
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # Denoising
    denoised = cv2.GaussianBlur(thresh, (5, 5), 0)
    return denoised

# Extract date, time, and temperature; return None if not found
def extract_metadata(image_path):
    processed_image = preprocess_image_for_ocr(image_path)
    
    custom_config = r'--oem 3 --psm 6 -l eng'
    text = pytesseract.image_to_string(processed_image, config=custom_config)

    # Clean the text
    text = text.replace('°', '° ')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^0-9:\- /°Cc]', ' ', text)
    text = re.sub(r'\s*:\s*', ':', text)
    text = re.sub(r'\s*-\s*', '-', text)

    date = None
    time = None
    temperature = None

    # Extract date
    date_match = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', text) or re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})', text)
    if date_match:
        date = date_match.group(1)
    else:
        print("Failed to extract date; will be stored as NULL.")
        date = None

    # Extract time
    time_match = re.search(r'(\d{1,2}:\d{2}:\d{2})', text)
    if time_match:
        time = time_match.group(1)
    else:
        print("Failed to extract time; will be stored as NULL.")
        time = None

    # Extract temperature
    temperature_match = re.search(r'(\d+)\s*°\s*[Cc]', text)
    if temperature_match:
        temperature = f"{temperature_match.group(1)}°C"
    else:
        print("Failed to extract temperature; will be stored as NULL.")
        temperature = None

    return date, time, temperature

# Function to get or create site ID based on the site name
def get_or_create_site_id(site_name):
    cursor.execute("SELECT site_id FROM sites WHERE site_name = %s", (site_name,))
    result = cursor.fetchone()
    
    if result:
        return result[0]  # Site already exists, return the site_id
    else:
        # Insert new site and return the new site_id
        insert_query = "INSERT INTO sites (site_name) VALUES (%s)"
        cursor.execute(insert_query, (site_name,))
        connection.commit()  # Commit to save changes
        return cursor.lastrowid  # Return the newly created site_id

def get_or_create_species_id(species_name):
    # Get the species ID from the mapping
    species_id = next((id for id, name in class_mapping.items() if name == species_name), None)

    if species_id is not None:
        # Check if the species already exists in the database
        cursor.execute("SELECT species_id FROM species WHERE species_id = %s", (species_id,))
        result = cursor.fetchone()

        if result:
            # Species exists, return the existing species_id
            return result[0]
        else:
            # Species does not exist, insert new species
            insert_query = "INSERT INTO species (species_id, species_name) VALUES (%s, %s)"
            cursor.execute(insert_query, (species_id, species_name))
            connection.commit()  # Commit to save changes
            return species_id  # Return the newly created species_id
    else:
        print(f"Species name '{species_name}' not found in mapping.")
        return None  # Handle it appropriately, e.g., return None or raise an error

def process_and_save_images():
    global connection, cursor
    # Ensure the cropped images directory exists
    if not os.path.exists(cropped_image_save_path):
        os.makedirs(cropped_image_save_path)

    for site_folder in os.listdir(dataset_path):
        site_path = os.path.join(dataset_path, site_folder)

        if os.path.isdir(site_path):
            print(f"Processing site folder: {site_folder}")
            site_id = get_or_create_site_id(site_folder)
            filter_folder = os.path.join(site_path, 'Filter')

            if os.path.exists(filter_folder):
                for species_folder in os.listdir(filter_folder):
                    if any(term in species_folder.lower() for term in ['undentified', 'unidentified', 'mix']):
                        print(f"Skipping unidentified species folder: {species_folder}")
                        continue

                    species_path = os.path.join(filter_folder, species_folder)
                    if os.path.isdir(species_path):
                        image_list = [os.path.join(species_path, img) for img in os.listdir(species_path) if img.lower().endswith(('jpg', 'jpeg', 'png'))]

                        for image_path in image_list:
                            print(f"Processing image: {image_path}")

                            try:
                                # Run YOLO model for object detection
                                pred = yolo_model.predict(source=image_path, conf=0.4)
                                image = cv2.imread(image_path)

                                if pred and hasattr(pred[0], 'boxes'):
                                    boxes = pred[0].boxes
                                    print(f"Detected {boxes.data.size(0)} objects in {image_path}")

                                    capture_date, capture_time, temperature = None, None, None
                                    if boxes.data.size(0) > 0:
                                        # Extract metadata like date, time, and temperature from the image using OCR
                                        capture_date, capture_time, temperature = extract_metadata(image_path)

                                    for i, box in enumerate(boxes.data):
                                        if box.shape[0] == 6:
                                            x1, y1, x2, y2, conf, class_id = box.tolist()
                                            # Crop the detected object
                                            cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
                                            if cropped_image.size == 0:
                                                print(f"Cropped image is empty for box {i}. Skipping.")
                                                continue

                                            cropped_image_path = os.path.join(cropped_image_save_path, f"{os.path.basename(image_path)}_crop_{i}.jpg")
                                            cv2.imwrite(cropped_image_path, cropped_image)
                                            print(f"Cropped image saved to: {cropped_image_path} (Confidence: {conf}, Class ID: {class_id})")
                                            cropped_image = image[int(y1):int(y2), int(x1):int(x2)]

                                            # Convert cropped image from NumPy array to PyTorch tensor
                                            try:
                                                # Ensure the cropped image is properly shaped
                                                if len(cropped_image.shape) == 3:  # Check if the image has 3 channels
                                                    pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                                                    cropped_image_tensor = preprocess(pil_image).unsqueeze(0)  # Add batch dimension
                                                    cropped_image_tensor = cropped_image_tensor.to(device)
                                                    
                                                    # Debugging: Check the type and shape of the tensor
                                                    print(f"Cropped image tensor shape: {cropped_image_tensor.shape}, type: {type(cropped_image_tensor)}")
                                                    
                                                    # Make prediction using ResNet
                                                    with torch.no_grad():
                                                        outputs = resnet_model(cropped_image_tensor)
                                                        _, predicted = torch.max(outputs, 1)
                                                        species = get_species_name(predicted.item())
                                                        print(f"Predicted species by ResNet: {species}")
                                                        species_id = get_or_create_species_id(species)
                                                else:
                                                    print(f"Cropped image has unexpected shape: {cropped_image.shape}. Skipping.")
                                            except Exception as e:
                                                print(f"Error converting cropped image to tensor: {e}")
                                                continue

                                            if species_id is not None:
                                                # Ensure database connection
                                                if not connection.is_connected():
                                                    connection = connect_to_database()
                                                    if connection is None:
                                                        print("Failed to reconnect to the database. Skipping this image.")
                                                        continue
                                                    cursor = connection.cursor()

                                                try:
                                                    with open(cropped_image_path, 'rb') as f:
                                                        image_data = f.read()

                                                    insert_image_query = """
                                                        INSERT INTO images (image_name, image_data, capture_date, capture_time, temperature, site_id)
                                                        VALUES (%s, %s, %s, %s, %s, %s)
                                                    """
                                                    cursor.execute(insert_image_query, (cropped_image_path, image_data, capture_date, capture_time, temperature, site_id))
                                                    connection.commit()
                                                    image_id = cursor.lastrowid
                                                    print(f"Inserted image record with ID: {image_id}")

                                                    insert_prediction_query = """
                                                        INSERT INTO predictions (image_id, species_id, confidence_score)
                                                        VALUES (%s, %s, %s)
                                                    """
                                                    cursor.execute(insert_prediction_query, (image_id, species_id, conf))
                                                    connection.commit()
                                                    print(f"Inserted prediction for image ID {image_id} with species ID {species_id}")
                                                except mysql.connector.Error as db_error:
                                                    print(f"Database error: {db_error}. Trying to reconnect.")
                                                    connection = connect_to_database()
                                                    if connection:
                                                        cursor = connection.cursor()
                                                    else:
                                                        print("Failed to reconnect. Skipping further database operations.")
                            except Exception as e:
                                print(f"Error processing image {image_path}: {e}")
# Closing the database connection
def close_database_connection():
    cursor.close()
    connection.close()
    print("Database connection closed.")

# Run the function to process and save images
process_and_save_images()

# Close the database connection after processing
close_database_connection()