from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import os
import cv2
import numpy as np
import json


#This is the image segmentation approach it processes all images in the Datenset folder 



image_folder_path = 'Datenset'
response_path = 'BirefNet_Output\\birefnet_output.json'
birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to('cuda')
birefnet.eval()



def extract_bbox(imagepath):
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(imagepath).convert('RGB')
    input_images = transform_image(image).unsqueeze(0).to('cuda')

    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_np = pred.numpy()

    mask = (pred_np > 0.5).astype(np.uint8) * 255

    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    mask_resized = cv2.resize(mask, (image_cv.shape[1], image_cv.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox = [0,0,0,0]
    # Draw a bounding box around the largest contour on the original image
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box coordinates
        x,y,w,h = cv2.boundingRect(largest_contour)
        bbox = [x,y,x+w,y+h]

    print("image: ", imagepath, bbox)
    return bbox


results = {}
for image_name in os.listdir(image_folder_path):
    image_path = image_folder_path + "\\" + image_name
    original_image = Image.open(image_path)
    results[image_name] = extract_bbox(image_path)

print(results)

with open(response_path, "w") as json_file:
    json.dump(results, json_file, indent=4)
