from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import os
import argparse
import cv2

# Set HOME environment variable if it doesn't exist
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.environ['USERPROFILE']  # or use another appropriate directory

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Image number")
args = vars(parser.parse_args())

# Define image path
imagepath = 'Datenset\\picture' + args["image"] + '.jpg'
SAVETO = 'Freisteller_Output'
# Load BiRefNet model
birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to('cuda')
birefnet.eval()

def extract_object(birefnet, imagepath):
    
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load and transform the image
    image = Image.open(imagepath).convert("RGB")
    input_images = transform_image(image).unsqueeze(0).to('cuda')

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()

    binary_mask = (pred > 0.5).float()
    pred_pil = transforms.ToPILImage()(binary_mask)
    mask_resized = pred_pil.resize(image.size, Image.LANCZOS)
    
    # Convert to RGBA (transparency support) and apply mask
    image_rgba = image.convert("RGBA")
    data = image_rgba.getdata()

    new_data = []
    for i, item in enumerate(data):
        if mask_resized.getdata()[i] == 0: 
            new_data.append((255, 255, 255, 0)) 
        else:  # Foreground (where mask is 1)
            new_data.append(item)
    image_rgba.putdata(new_data)
    return image_rgba, mask_resized

# Extract object and visualize
result_image = extract_object(birefnet, imagepath)[0]
freisteller_path = SAVETO + "/{}_image_freisteller.png".format(args["image"])
result_image.save(freisteller_path, "PNG")
# Visualization
plt.axis("off")
plt.imshow(result_image)
plt.show()
