from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import os
import argparse
import cv2
import numpy as np

# Set HOME environment variable if it doesn't exist
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.environ['USERPROFILE']  # or use another appropriate directory

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Imagenumber")
args = vars(parser.parse_args())

imagepath ='Datenset\\picture'+ args["image"] + '.jpg'
SAVETO = 'outputCV\\'
birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to('cuda')
birefnet.eval()
di = args["image"]

def extract_object(birefnet, imagepath):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load and transform the image
    image = Image.open(imagepath).convert('RGB')
    input_images = transform_image(image).unsqueeze(0).to('cuda')

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

    # Draw a bounding box around the largest contour on the original image
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box coordinates
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw the bounding box on the original image
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box with thickness 2

    return image_cv  # Return the original image with the bounding box

# Extract the object and get the image with bounding box
result_image = extract_object(birefnet, imagepath)

# Visualization using OpenCV
cv2.imshow('Result', result_image)
birefnetpath = SAVETO + "/{}_image_birefBound.png".format(di)
cv2.imwrite(birefnetpath, result_image)
cv2.waitKey(0)  # Wait until a key is pressed to close the window
cv2.destroyAllWindows()

