import requests

image_path = "https://raw.githubusercontent.com/yashprakash13/face-detector-app/master/test_images/single.jpg"

api_endpoint = "http://localhost:8000/detect-face"
params = (
    ('img_path', image_path),
)
response = requests.post(api_endpoint, params=params)

print(response.text)
