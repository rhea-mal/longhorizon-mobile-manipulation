# from vlm_utils import analyze_imageraw_with_gpt4v
import cv2
from PIL import Image
# pip install google-genai
from google import genai
import re
import json
import os
import yaml


def extract_action_plan_from_gemini_response(response):
    try:
        text = response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        raise ValueError(f"Could not extract text from response: {e}")

    # Try to extract JSON inside ```json ... ```
    match = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", text)
    if match:
        json_str = match.group(1)
    else:
        # fallback: check if the whole response is a JSON array string
        if text.startswith("[") and text.endswith("]"):
            json_str = text
        else:
            raise ValueError("No JSON array found in the response.")

    try:
        actions = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

    if not isinstance(actions, list):
        raise ValueError("Parsed JSON is not a list.")

    return actions

### GEMINI
def gemini_prompt(prompt, image_path=None, mode="basic"):
    api_key="AIzaSyAfs_FIMNuV4DWjUN47rc2IzZ7uQEvhqg0"
    client = genai.Client(api_key=api_key)
    # genai.configure(api_key=api_key)

    if mode=="basic":
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

    elif mode =="with_context":
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction="Return an array of actions"),
            contents=prompt
        )
    
    elif mode=="image":
        image = Image.open(image_path)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, prompt]
        )
    return response

### CHAT GPT 40
# def chatgpt4o_prompt(prompt, image_path):
#     image_data = cv2.imread(jpg_path)
#     ## replace this with lab key when jeannette back
#     api_key = "yourapikey"
#     vlm_response = analyze_imageraw_with_gpt4v(
#                 image_data,
#                 prompt,
#                 api_key=api_key,
#                 model='gpt-4o-2024-08-06',
#                 show_timing=True
#             ) 
#     return vlm_response

def format_config(plan_array, library):
    config = []
    for name in plan_array:
        config.append(library[name])
    config_path = "planner/longhorizon_plan.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print(f"✅ Config written to {config_path}")


def list_of_policies(DIR):
    policies = {}  # dict of policy name → full path
    for base_dir in [DIR]:
        if not os.path.exists(base_dir):
            print(base_dir, " doesn't exist")
            continue  
        for name in os.listdir(base_dir):
            path = os.path.join(base_dir, name)
            if os.path.isdir(path):
                policies[name] = path

    return policies

def make_plan(policy_library_dir, task):
    library = list_of_policies(policy_library_dir)
    prompt = (
        "You are a helpful robot planner. Only return a JSON array of actions. "
        "Do not include any explanation, markdown, or formatting.\n"
        "Here is an example format:\n"
        "[\"pick_red_cube\", \"place_red_cube\", \"pick_blue_cube\", \"place_blue_cube\"]\n\n"
        f"Now generate a plan using only the following available actions:\n"
        f"{', '.join(library.keys())}\n\n"
        f"The task is: {task}.\n\n"
        "Return only the JSON array."
    )

    raw_output = gemini_prompt(prompt)
    print(raw_output)
    plan_array = extract_action_plan_from_gemini_response(raw_output)
    format_config(plan_array, library)

def main():
    policy_library_dir = "exps/waypoint/longhorizon"
    task = "pick and place the cube twice"
    make_plan(policy_library_dir, task)

if __name__ == '__main__':
    main()
