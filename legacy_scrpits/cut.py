import argparse
import json
import cv2
import os
from calculateMargins import calculate_bounding_box_distances




#The purpose of this script was to apply a cut an image based on the amount of columns and rows desired for the image as they are defined for the newspaper (Spaltigkeit) it takes the arguments -i image -c columns and -r rows
#The

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Image")
parser.add_argument("-c", "--columns", required=True, help="Columns")
parser.add_argument("-r", "--rows", required=True, help="Rows")
args = vars(parser.parse_args())



di = args["image"]

image_path = 'Datenset\\picture'+ args["image"] + '.jpg'
SAVETO = 'outputCV\\picture'

image = cv2.imread(image_path)

intendedColums = int(args["columns"])
intendedRows = int(args["rows"])

intendedWidth = 50000 * intendedColums + 5000 * (intendedColums - 1)
intendedHeight = 3845 * intendedRows

intendedAspectRatio = intendedWidth/intendedHeight
print(f"the intended Aspect Ratio is {intendedAspectRatio}")




def cut(image, intendedRatio):
    
    height, width = image.shape[:2]
    originalRatio = width / height
    cut_top = 0
    cut_bottom = 0
    cut_left = 0
    cut_right = 0
    margins = calculate_bounding_box_distances(image_path)
    margins = json.loads(margins)
    print(margins)
    if originalRatio < intendedRatio:
        print("image cut horizontally")
        targetheight = width / intendedRatio
        horizontalCut = round(height - targetheight)
        cut_top = min(int(margins["distance_top"]),horizontalCut)
        cut_bottom = min(int(margins["distance_bottom"]),horizontalCut-cut_top)
        if horizontalCut > cut_top + cut_bottom:
            print("Image content doesnt allow for this aspect Ratio!")
    
    if originalRatio > intendedRatio:
        print("image cut vertically")
        targetwidth = height * intendedRatio
        verticalCut = round(width - targetwidth)
        cut_left = min(int(margins["distance_left"]),verticalCut)
        cut_right = min(int(margins["distance_right"]),verticalCut-cut_left)
        if verticalCut > cut_left + cut_right:
            print("Image content doesnt allow for this aspect Ratio!")

    cutImage = image [cut_top:height-cut_bottom,cut_left:width-cut_right]
    return cutImage  


cutImage = cut(image, intendedAspectRatio)

cv2.imshow('Result', cutImage)
cutImagePath = SAVETO + "_image_cut.png".format(di)
cv2.imwrite(cutImagePath, cutImage)
cv2.waitKey(0)  # Wait until a key is pressed to close the window
cv2.destroyAllWindows()
   