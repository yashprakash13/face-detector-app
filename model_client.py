# import requests

# response = requests.post("http://127.0.0.1:8000/detect-face", params={"img_path": "./test_images/multiple.jpg"})

# print(response.text)


import requests
import numpy as np
import cv2
import base64

# Load an image using OpenCV
image_path = "./test_images/multiple.jpg"
image = cv2.imread(image_path)

# Convert the image to a NumPy ndarray
image_np = np.array(image)

# Convert the NumPy array to a base64-encoded string
image_base64 = base64.b64encode(image_np.tobytes()).decode('utf-8')

# Define the API endpoint
api_endpoint = "http://localhost:8000/detect-face"

# Prepare the payload
payload = {"img_path": image_base64}

# Send the cURL request using the requests library
response = requests.post(api_endpoint, json=payload)

# Print the response
print(response.text)