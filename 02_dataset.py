import json
import pandas as pd
import os
from PIL import Image
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt

image_folder_path = 'Datenset'


data =  {}

for image_name in os.listdir(image_folder_path):
        image_path = image_folder_path + "\\" + image_name
        original_image = Image.open(image_path)
        width, height = original_image.size


        data[image_name] = {
                "width": width,
                "height": height,
                "size10000": width * height / 10000,
                "aspect_ratio": width/height
        }


# data_df = pd.DataFrame.from_dict(data, orient="index")
# average_metrics = data_df.mean()
# max_metrics = data_df.max()
# min_metrics = data_df.min()
# data["00_average"] = average_metrics.to_dict()
# data["00_max"] = max_metrics.to_dict()
# data["00_min"] = min_metrics.to_dict()

with open("dataset_analysis.json", "w") as json_file:
    json.dump(data, json_file, indent=4)



with open('Object_Detection_output\\0.6raw_detections.json', 'r') as file:
    detection_data = json.load(file)

category_data = {}
for image in detection_data:
    categories = []
    for detection in detection_data[image]:
        categories.append(detection["categories"][0]["category_name"])
    category_data[image] = categories



def count_categories_per_image(data):
    result = {}
    for image, categories in data.items():
        # Use Counter to count occurrences of each category
        result[image] = dict(Counter(categories))
    return result

# Get the result
categories_per_image = count_categories_per_image(category_data)

with open("dataset_category_counts_image.json", "w") as json_file:
    json.dump(categories_per_image, json_file, indent=4)

category_count_in_set = defaultdict(int)
for image in categories_per_image:
     for category in categories_per_image[image]:
            if category == "person" and categories_per_image[image]["person"] == 1:
               category_count_in_set["single person"] += 1
            if category == "person" and categories_per_image[image]["person"] == 2:
               category_count_in_set["pair"] += 1
            if category == "person" and categories_per_image[image]["person"] >= 3:
                category_count_in_set["group"] += 1
            category_count_in_set[category] += 1

with open("dataset_category_counts_set.json", "w") as json_file:
    json.dump(category_count_in_set, json_file, indent=4)


refined_categories = {}
for image in categories_per_image:
    refined_categories[image] = "none"
    for category in categories_per_image[image]:
        if category ==  "person" and categories_per_image[image]["person"] == 1:
            refined_categories[image] = "single-person"
        elif category ==  "person" and categories_per_image[image]["person"] == 2:
            refined_categories[image] = "pair"
        elif category ==  "person" and categories_per_image[image]["person"] >= 3:
            refined_categories[image] = "group"
        elif category in ["dog","cat", "horse","elephant"]:
            refined_categories[image] = "animal"
        else:
            refined_categories[image] = "other"


with open("dataset_categories_defined.json", "w") as json_file:
    json.dump(refined_categories, json_file, indent=4)


category_counts = Counter(refined_categories.values())
print(category_counts)

# Extract categories and their counts
categories = list(category_counts.keys())
counts = list(category_counts.values())

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(categories, counts, color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.title('Distribution of Picture Categories')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

sizes = [info["size10000"] for info in data.values()]
aspect_ratios = [info["aspect_ratio"] for info in data.values()]

# Plotting
plt.figure(figsize=(10, 6))

# Scatter plot: Size vs Aspect Ratio
plt.scatter(aspect_ratios, sizes, alpha=0.7)
plt.title("Image Size vs Aspect Ratio")
plt.xlabel("Aspect Ratio (width / height)")
plt.ylabel("Size (width * height)")
# Optional: Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.5)

# Optional: Highlight specific points or add annotations
# Example: Highlight the largest image
plt.legend()

plt.tight_layout()
plt.show()