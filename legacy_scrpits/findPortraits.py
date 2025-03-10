
import json
import shutil
import os


json_file_path = 'C:\\Users\\janhe\\Desktop\\Masterarbeit\\image_saliency_opencv-master\\image_saliency_opencv-master\\output\\detection_results.json'
with open(json_file_path, 'r') as f:
        image_dict = json.load(f)

paths = {}
paths["portraits"] = 'Input\\Portraits'
paths["group"] = 'Input\\Groups'
paths["notPerson"] = 'Input\\notPerson'

for path in paths:
    if os.path.exists(paths[path]):
        shutil.rmtree(paths[path])
    os.makedirs(paths[path])


for image in image_dict:
    input_path = "C:\\Users\\janhe\\Desktop\\Masterarbeit\\image_saliency_opencv-master\\image_saliency_opencv-master\\images\\zeitung" + "\\" + image 
    portrait = False
    person = False
    for detection in image_dict[image]:
        if detection["category"] == "person":
            if detection["score"] > 0.5:
                person = True
                if detection["bboxrelative"] >= 0.2:
                    portrait = True
      
            
    if person:
        if portrait:
            destination_path = paths["portraits"] + "\\" + image       
        else: 
            destination_path = paths["group"] + "\\" + image
    else:
        destination_path = paths["notPerson"] + "\\"  + image            
    shutil.copy(input_path, destination_path)            
               