from io import BytesIO
import logging
from typing import Dict, List

from deepface import DeepFace
import numpy as np
from PIL import Image
import requests


logger = logging.getLogger("ray.serve")

def perform_detection_and_return(img_path: str) -> List[Dict]:
    response = requests.get(img_path, stream=True)
    if response.status_code == 200:
        image_bytes = BytesIO(response.content)
        image_pil = Image.open(image_bytes)
        img_np = np.array(image_pil)
        response = DeepFace.extract_faces(img_path=img_np, detector_backend="mtcnn")        
        logger.info("Output done.")
        output = []
        for index, face in enumerate(response):
            output.append(response[index]['facial_area'])
        return output
    else:
        logger.error(response.text)
