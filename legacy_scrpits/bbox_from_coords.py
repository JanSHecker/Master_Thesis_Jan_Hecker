from PIL import Image, ImageDraw
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Imagenumber")
args = vars(parser.parse_args())
di = args["image"]
imagepath = r'C:\Users\janhe\Desktop\Masterarbeit\image_saliency_opencv-master\image_saliency_opencv-master\images\zeitung\picture'+ args["image"] + '.jpg'
SAVETO = r'C:\Users\janhe\Desktop\Masterarbeit\BirefnetHuggingface\outputCV'
# Load the image

image = Image.open(imagepath)

# Define the bounding box coordinates
x1, y1 = 1200, 200
x2, y2 = 2400, 800

# Draw the bounding box
draw = ImageDraw.Draw(image)
draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

# Display the image with the bounding box
image.show()

# Optionally, save the image with the bounding box
output_path = SAVETO + "/{}_image_box_from_coords.png".format(di)
image.save(output_path)
print(f"Image saved with bounding box at {output_path}")
