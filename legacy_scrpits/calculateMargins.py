from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import cv2
import numpy as np
import json
import os
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.environ['USERPROFILE']  





#This script is intended for use in cut.py



def calculate_bounding_box_distances(imagepath, model_name='ZhengPeng7/BiRefNet', device='cuda'):
    """
    Calculate the distances from the borders of the bounding box of the detected object 
    to the borders of the image.

    Parameters:
        imagepath (str): Path to the input image file.
        model_name (str): The model name for image segmentation.
        device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
        str: A JSON string containing the distances from the bounding box to the image borders.
    """

    # Load the pre-trained BiRefNet model
    birefnet = AutoModelForImageSegmentation.from_pretrained(model_name, trust_remote_code=True)
    torch.set_float32_matmul_precision('high')
    birefnet.to(device)
    birefnet.eval()

    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load and transform the image
    image = Image.open(imagepath).convert('RGB')
    input_images = transform_image(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_np = pred.numpy()

    # Convert the prediction mask to binary format
    mask = (pred_np > 0.5).astype(np.uint8) * 255

    # Convert the PIL image to an OpenCV image
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Ensure the mask has the same size as the original image
    mask_resized = cv2.resize(mask, (image_cv.shape[1], image_cv.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the distances from the bounding box to the image borders
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box coordinates
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate distances from the bounding box to the image borders
        distance_left = x
        distance_right = image_cv.shape[1] - (x + w)
        distance_top = y
        distance_bottom = image_cv.shape[0] - (y + h)

        # Create a dictionary with the distances
        distances = {
            "distance_left": distance_left,
            "distance_right": distance_right,
            "distance_top": distance_top,
            "distance_bottom": distance_bottom
        }

        # Return the distances as a JSON string
        return json.dumps(distances, indent=4)

    else:
        # If no contours are found, return None
        return json.dumps({"error": "No object detected"})
    
pass

