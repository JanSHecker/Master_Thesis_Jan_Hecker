# Imports
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
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Imagenumber")
args = vars(parser.parse_args())
imagepath = r'C:\Users\janhe\Desktop\Masterarbeit\image_saliency_opencv-master\image_saliency_opencv-master\images\zeitung\picture'+ args["image"] + '.jpg'

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

    image = Image.open(imagepath)
    input_images = transform_image(image).unsqueeze(0).to('cuda')

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    return image, mask
resultimage = extract_object(birefnet, imagepath)[0]
# Visualization
plt.axis("off")
plt.imshow(resultimage)
plt.show()

