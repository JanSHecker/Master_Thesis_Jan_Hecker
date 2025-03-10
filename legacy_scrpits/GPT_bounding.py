from openai import OpenAI
import base64
import argparse
from PIL import Image, ImageDraw
import json
import os



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Image number")
args = vars(parser.parse_args())

# Load the image
image_path = r'C:\Users\janhe\Desktop\Masterarbeit\image_saliency_opencv-master\image_saliency_opencv-master\images\zeitung\picture' + args["image"] + '.jpg'
response_path = r'C:\Users\janhe\Desktop\Masterarbeit\BirefnetHuggingface\GPT_Output\response' + args["image"] + '.json'

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')





base64_image = encode_image(image_path)

client = OpenAI(api_key="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")

def ask_gpt(prompt):
    if os.path.exists(response_path):
        print("Response already exists. No new request")
        with open(response_path) as json_file:
            return json.load(json_file)
    print("Making new request to Open AI")   
    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": prompt,
        },
        {
          "type": "image_url",
          "image_url": {
            "url":  f"data:image/jpeg;base64,{base64_image}",
            "detail": "high",
          },
        },
      ],
    }
  ],
  max_tokens = 300
)
    response = response.choices[0].message
    serializable_data = {
    "role": response.role,
    "content": response.content
}
    return serializable_data


# Get image dimensions
original_image = Image.open(image_path)
original_image.show()
width, height = original_image.size
print(f"Image dimensions: {width}x{height}")

# Step 1: Ask GPT for bounding box coordinates based on dimensions
prompt = f"""
Given an image with dimensions {width}x{height}, please provide estimated bounding box coordinates
for the most important motive of the picture, so that nothing important is left out (x1, y1) for the top-left corner and (x2, y2) for the bottom-right corner.
"""
bounding_box_response = ask_gpt(prompt)
# Example: If bounding_box_response contains custom objects


# Save as JSON
with open(response_path, "w") as json_file:
    json.dump(bounding_box_response, json_file, indent=4)


print("GPT-4 Response on Bounding Box:", bounding_box_response["content"])

# Parse GPT's response for bounding box coordinates
# Assuming GPT returns coordinates in the format:
# "Top-left corner: (x1, y1) : (1200, 200), Bottom-right corner: (x2, y2) : (2400, 800)"
coords = {}
for line in bounding_box_response["content"].splitlines():
    if "Top-left" in line:
        coords['x1'], coords['y1'] = map(int, line.split(":")[-1].strip(" ()").split(","))
    elif "Bottom-right" in line:
        coords['x2'], coords['y2'] = map(int, line.split(":")[-1].strip(" ()").split(","))

# Draw the bounding box based on GPT's response
if 'x1' in coords and 'y1' in coords and 'x2' in coords and 'y2' in coords:
    draw = ImageDraw.Draw(original_image)
    draw.rectangle([(coords['x1'], coords['y1']), (coords['x2'], coords['y2'])], outline="red", width=3)
    original_image.show()

    # Optionally, save the image with the bounding box
    output_path = "GPT_Output\gpt_picture" + args["image"] + ".jpg"
    original_image.save(output_path,  quality = 95)
    print(f"Image saved with bounding box at {output_path}")
else:
    print("Could not parse bounding box coordinates from GPT response.")
