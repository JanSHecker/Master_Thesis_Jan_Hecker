from PIL import Image, ImageDraw
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Imagenumber")
args = vars(parser.parse_args())
di = args["image"]
imagepath = 'Datenset\\picture'+ args["image"] + '.jpg'
SAVETO = 'outputCV\\'

image = Image.open(imagepath)


x1, y1 = 0, 0
x2, y2 = 100, 100


draw = ImageDraw.Draw(image)
draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)


image.show()


output_path = SAVETO + "{}_image_box_from_coords.png".format(di)
image.save(output_path)

