from openai import OpenAI
import base64
from PIL import Image, ImageDraw
import json
import os
import re


image_folder_path = 'Datenset'
response_path = 'GPT_Output\\gpt_output.json'
extracted_path = 'GPT_Output\\'

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')







client = OpenAI(api_key="key")

def ask_gpt(prompt,base64_image): 
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


def evaluate_dataset():
    if os.path.exists(response_path):
        print("Response already exists. No new request")
        with open(response_path) as json_file:
            return json.load(json_file)
    results = {}
    for image_name in os.listdir(image_folder_path):
        image_path = image_folder_path + "\\" + image_name
        original_image = Image.open(image_path)
        width, height = original_image.size
        base64_image = encode_image(image_path)

        prompt_1 = f"""
Given an image with dimensions {width}x{height}, provide estimated bounding box coordinates
for the most important motive of the picture, so that nothing important is left out (x1, y1) for the top-left corner and (x2, y2) for the bottom-right corner.
"""

        prompt_2 = f"""
Given a newspaper image with dimensions {width}x{height}, estimate bounding box coordinates (x1, y1) for the top-left corner and (x2, y2) for the bottom-right corner for a crop that ensures no key elements (e.g., faces, logos, or important visuals) are cut off. The crop should focus on the most important part of the image while preserving context
"""

        prompt_3 = f"""
Given a newspaper image with dimensions {width}x{height}, estimate bounding box coordinates (x1, y1) for the top-left corner and (x2, y2) for the bottom-right corner for a crop that minimizes the loss of contextual information. Ensure the crop retains the most important visual and textual elements while removing unnecessary background.
        """
        print('making request for: ' + image_path)
        bounding_box_response_1 = ask_gpt(prompt_1,base64_image)
        bounding_box_response_2 = ask_gpt(prompt_2,base64_image)
        bounding_box_response_3 = ask_gpt(prompt_3,base64_image)
        
        results[image_name] = {
          "prompt_1": bounding_box_response_1,
          "prompt_2": bounding_box_response_2,
          "prompt_3": bounding_box_response_3,
        }
    # Save as JSON
    with open(response_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    return results
  
def extract_coords(gpt_output):
    for prompt in gpt_output[list(gpt_output.keys())[0]]:
      extracted_bboxes = {}
      for key in gpt_output:
          coords = []
          print(gpt_output[key])
          response = gpt_output[key][prompt]
          content = response["content"]
          print("response: ", response)
          pattern = re.compile(r'\(?\d{1,5},\s*\d{1,5}\)?')
          matches = pattern.findall(content)
        
          if len(matches) >= 2:
              coords.extend(matches[0].strip('()').split(','))
              coords.extend(matches[1].strip('()').split(','))
      
          extracted_bboxes[key.split("\\")[-1]] = coords
      with open(extracted_path + prompt + "_output.json", "w") as json_file:
        json.dump(extracted_bboxes, json_file, indent=4)

#execute methods

gpt_output = evaluate_dataset()

extract_coords(gpt_output)