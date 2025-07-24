from openai import OpenAI
import cv2
import numpy as np
from PIL import Image
import base64
import re
import json
from fastdtw import fastdtw
import cv2
from PIL import Image, ImageTk
import pandas as pd
import textwrap
import time
import tkinter as tk
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt


def analyze_imageraw_with_gpt4v(image_data, prompt, api_key, model="gpt-4o", show_raw_response=False, show_timing=False):
    """
    Analyzes raw image data using the GPT-4 Vision model.

    Args:
        image_data (numpy.ndarray): The image data as a numpy array in BGR format.
        prompt (str): The text prompt to guide the analysis.
        api_key (str): The API key for accessing the OpenAI service.
        model (str, optional): The model to use for analysis. Defaults to "gpt-4o".
        show_raw_response (bool, optional): If True, prints the raw response from the API. Defaults to False.
        show_timing (bool, optional): If True, prints the time taken for the analysis. Defaults to False.

    Returns:
        str: The content of the response message from the GPT-4 Vision model.
    """
    if show_timing:
        start_time = time.time()

    # Convert numpy array to base64
    _, buffer = cv2.imencode('.png', image_data)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    client = OpenAI(
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        temperature=0.2,
    )

    if show_raw_response:
        print("Raw response:")
        print(response)
    if show_timing:
        end_time = time.time()
        print(f"Time for VLM: {end_time - start_time:.2f} seconds")
    return response.choices[0].message.content


def analyze_image_with_gpt4v(image_path, prompt, api_key, model="gpt-4o", show_raw_response=False, show_timing=False):
    """
    Analyzes an image using the GPT-4 Vision model.

    Args:
        image_path (str): The file path to the image to be analyzed.
        prompt (str): The text prompt to guide the analysis.
        api_key (str): The API key for accessing the OpenAI service.
        model (str, optional): The model to use for analysis. Defaults to "gpt-4o".
        show_raw_response (bool, optional): If True, prints the raw response from the API. Defaults to False.
        show_timing (bool, optional): If True, prints the time taken for the analysis. Defaults to False.

    Returns:
        str: The content of the response message from the GPT-4 Vision model.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image at path: {image_path}")

    # Call the base function with the image data
    return analyze_imageraw_with_gpt4v(image, prompt, api_key, model, show_raw_response, show_timing)


def verify_image_with_gpt4v(image_path, prior_result_image_path, prompt, api_key, model="gpt-4o", show_raw_response=False, show_timing=False):
    """
    Re-query an image using the GPT-4 Vision model.

    Args:
        image_path (str): The file path to the image to be analyzed.
        prompt (str): The text prompt to guide the analysis.
        api_key (str): The API key for accessing the OpenAI service.
        model (str, optional): The model to use for analysis. Defaults to "gpt-4o".
        show_raw_response (bool, optional): If True, prints the raw response from the API. Defaults to False.

    Returns:
        str: The content of the response message from the GPT-4 Vision model.
    """
    if show_timing:
        start_time = time.time()
    base64_image = encode_image_to_base64(image_path)
    prior_result_base64_image = encode_image_to_base64(prior_result_image_path)
    client = OpenAI(
        api_key=api_key,  # This is the default and can be omitted
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{prior_result_base64_image}"}}
                ]
            }
        ],
    )
    # TODO: change temperature
    if show_raw_response:
        print("Raw response:")
        print(response)
    if show_timing:
        end_time = time.time()
        print(f"Time for VLM: {end_time - start_time:.2f} seconds")
    return response.choices[0].message.content
