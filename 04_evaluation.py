import json
import pandas as pd

control_output_path = 'Image Cropping Detection.v1i.coco\\train\\_annotations.coco.json'
gpt_output_path_1 = 'GPT_Output\\prompt_1_output.json'
gpt_output_path_2 = 'GPT_Output\\prompt_2_output.json'
birefnet_output_path = 'BirefNet_Output\\birefnet_output.json'
object_detection_output_path_1 = 'Object_Detection_output\\0.3constructedbbox.json'
object_detection_output_path_2 = 'Object_Detection_output\\0.6constructedbbox.json'
evaluation_output_path = 'Evaluation\\'
conversion_key_path = ''
meta_image_path = 'dataset_analysis.json'
image_categories_path = 'dataset_categories_defined.json'


with open(control_output_path, 'r') as file:
    control_data = json.load(file)

with open(gpt_output_path_1, 'r') as file:
    gpt_data_1 = json.load(file)

with open(gpt_output_path_2, 'r') as file:
    gpt_data_2 = json.load(file)

with open(object_detection_output_path_1, 'r') as file:
    object_detection_data_1 = json.load(file)

with open(object_detection_output_path_2, 'r') as file:
    object_detection_data_2 = json.load(file)

with open(birefnet_output_path, 'r') as file:
    birefnet_data = json.load(file)

# Extract the annotations
annotations = control_data["annotations"]

# Map image_id to bbox
control_bbox = {annotation["image_id"]: annotation["bbox"] for annotation in annotations}
for image_id in control_bbox:
    control_bbox[image_id] = [control_bbox[image_id][0], control_bbox[image_id][1], control_bbox[image_id][0]+ control_bbox[image_id][2], control_bbox[image_id][1] + control_bbox[image_id][3]]

# Print the resulting dictionary
print(control_bbox)


def get_intersection_area(bbox1,bbox2):

    if bbox1 == [] or bbox2 == []:
        return 0

    x1_min, y1_min, x1_max, y1_max = map(int, bbox1)
    x2_min, y2_min, x2_max, y2_max = map(int, bbox2)

    # Calculate intersection coordinates
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    # Calculate intersection area
    inter_width = max(0, x_inter_max - x_inter_min)
    inter_height = max(0, y_inter_max - y_inter_min)
    inter_area = inter_width * inter_height

    return inter_area

def calculate_bbox_area(bbox):
    if bbox == []:
        return 0
    x_min, y_min, x_max, y_max = map(int,bbox)
    width = x_max - x_min
    height = y_max - y_min




    return width * height

def compare_dict_keys(dict1, dict2):
    # Get the keys of both dictionaries
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    # Find common keys
    common_keys = keys1.intersection(keys2)

    # Find keys unique to each dictionary
    unique_to_dict1 = keys1 - keys2
    unique_to_dict2 = keys2 - keys1

    # Print results
    print("Common keys:", common_keys)
    print("Keys unique to dict1:", unique_to_dict1)
    print("Keys unique to dict2:", unique_to_dict2)


image_data = control_data["images"]
conversion_key = {image["file_name"].split(".")[0].replace("_","."):image["id"] for image in image_data}
print(conversion_key)

compare_dict_keys(gpt_data_1, conversion_key)

gpt_converted_1 = {conversion_key[filename]:gpt_data_1[filename] for filename in gpt_data_1}
gpt_converted_2 = {conversion_key[filename]:gpt_data_2[filename] for filename in gpt_data_2}
birefnet_converted = {conversion_key[filename]:birefnet_data[filename] for filename in birefnet_data}
object_detection_converted_1 = {conversion_key[filename]:object_detection_data_1[filename] for filename in object_detection_data_1}
object_detection_converted_2 = {conversion_key[filename]:object_detection_data_2[filename] for filename in object_detection_data_2}

def evaluate_predictions(prediction_bboxes):

    true_positive_area = {key: get_intersection_area(prediction_bboxes[key],control_bbox[key]) for key in control_bbox}
    false_positive_area = {key: calculate_bbox_area(prediction_bboxes[key]) - true_positive_area[key] for key in control_bbox}
    false_negative_area = {key: calculate_bbox_area(control_bbox[key]) - true_positive_area[key] for key in control_bbox}
    true_negative_area = {key: image_data[key]["width"] * image_data[key]["height"] - calculate_bbox_area(prediction_bboxes[key]) - calculate_bbox_area(control_bbox[key]) + true_positive_area[key] for key in control_bbox}


    metrics = {}

    for key in control_bbox:
        TP = true_positive_area[key]  # True Positives
        TN = true_negative_area[key]  # True Negatives
        FP = false_positive_area[key]  # False Positives
        FN = false_negative_area[key]  # False Negatives
        
        if key == 93:
            print("93 :", prediction_bboxes[key],control_bbox[key])
        # Accuracy
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        # Precision
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        # Recall
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # F1-Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Intersection over Union (IoU)
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

        metrics[key] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "iou": iou,
        }
        # metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
        # average_metrics = metrics_df.mean()
        # max_metrics = metrics_df.max()
        # min_metrics = metrics_df.min()
        # metrics["00_average"] = average_metrics.to_dict()
        # metrics["00_max"] = max_metrics.to_dict()
        # metrics["00_min"] = min_metrics.to_dict()

    return metrics


gpt_metrics_1 = evaluate_predictions(gpt_converted_1)
gpt_metrics_2 = evaluate_predictions(gpt_converted_2)
birefnet_metrics = evaluate_predictions(birefnet_converted)
object_detection_metrics_1 = evaluate_predictions(object_detection_converted_1)
object_detection_metrics_2 = evaluate_predictions(object_detection_converted_2)

with open(evaluation_output_path + "gpt_evaluation_1.json", "w") as json_file:
    json.dump(gpt_metrics_1, json_file, indent=4)

with open(evaluation_output_path + "gpt_evaluation_2.json", "w") as json_file:
    json.dump(gpt_metrics_2, json_file, indent=4)

with open(evaluation_output_path + "birefnet_evaluation.json", "w") as json_file:
    json.dump(birefnet_metrics, json_file, indent=4)

with open(evaluation_output_path + "object_detection_1.json", "w") as json_file:
    json.dump(object_detection_metrics_1, json_file, indent=4)

with open(evaluation_output_path + "object_detection_2.json", "w") as json_file:
    json.dump(object_detection_metrics_2, json_file, indent=4)

with open(conversion_key_path + "conversion_key.json", "w") as json_file:
    json.dump(conversion_key, json_file, indent=4)






with open(meta_image_path, 'r') as file:
    image_meta_data = json.load(file)

with open(image_categories_path, 'r') as file:
    image_categories = json.load(file)



meta_data_converted = {conversion_key[filename]:image_meta_data[filename] for filename in image_meta_data}
image_categories_converted = {conversion_key[filename]:image_categories[filename] for filename in image_categories}


all_data = {}

for image in gpt_converted_1:
    all_data[image] = {
        "accuracy od1": object_detection_metrics_1[image]["accuracy"],
        "precision od1": object_detection_metrics_1[image]["precision"],
        "recall od1": object_detection_metrics_1[image]["recall"],
        "f1_score od1": object_detection_metrics_1[image]["f1_score"],
        "iou od1": object_detection_metrics_1[image]["iou"],
        "accuracy od2": object_detection_metrics_2[image]["accuracy"],
        "precision od2": object_detection_metrics_2[image]["precision"],
        "recall od2": object_detection_metrics_2[image]["recall"],
        "f1_score od2": object_detection_metrics_2[image]["f1_score"],
        "iou od2": object_detection_metrics_2[image]["iou"],
        "accuracy is": birefnet_metrics[image]["accuracy"],
        "precision is": birefnet_metrics[image]["precision"],
        "recall is": birefnet_metrics[image]["recall"],
        "f1_score is": birefnet_metrics[image]["f1_score"],
        "iou is": birefnet_metrics[image]["iou"],
        "accuracy gpt1": gpt_metrics_1[image]["accuracy"],
        "precision gpt1": gpt_metrics_1[image]["precision"],
        "recall gpt1": gpt_metrics_1[image]["recall"],
        "f1_score gpt1": gpt_metrics_1[image]["f1_score"],
        "iou gpt1": gpt_metrics_1[image]["iou"],
        "accuracy gpt2": gpt_metrics_2[image]["accuracy"],
        "precision gpt2": gpt_metrics_2[image]["precision"],
        "recall gpt2": gpt_metrics_2[image]["recall"],
        "f1_score gpt2": gpt_metrics_2[image]["f1_score"],
        "iou gpt2": gpt_metrics_2[image]["iou"],
        "category": image_categories_converted[image],
        "size10000": meta_data_converted[image]["size10000"],
        "aspect ratio": meta_data_converted[image]["aspect_ratio"]
    }

with open(evaluation_output_path + "full_evaluation.json", "w") as json_file:
    json.dump(all_data, json_file, indent=4)