from io import BytesIO
from typing import Union

from fastapi.responses import JSONResponse
from deepface import DeepFace
from fastapi import FastAPI
import numpy as np
from PIL import Image
from ray import serve
import requests

from logger import Logger


app = FastAPI()

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.5, "num_gpus": 0})
@serve.ingress(app)
class FaceDetector:
    def __init__(self):
        # Load model
        self.logger = Logger(module="FaceDetector")
        self.logger.info("Initialized model.")

    def _process_image(img_path: Union[np.ndarray, str]) -> np.ndarray:
        if isinstance(img_path, str):
            # If img_path is a string, assume it's a file path and download the image
            response = requests.get(img_path)
            img = Image.open(BytesIO(response.content))
            img_np = np.array(img)
            return img_np
        elif isinstance(img_path, np.ndarray):
            # If img_path is a NumPy array, use it directly
            img_np = img_path
            return img_np
        else:
            # Handle other cases or raise an error if needed
            raise ValueError("Unsupported type for img_path. Must be str or np.ndarray.")

    @app.post("/detect-face")
    def detect_face(self, img_path: Union[np.ndarray, str]) -> dict:
        # Run inference
        img = self._process_image(img_path)
        response = DeepFace.extract_faces(img_path=img, detector_backend="mtcnn")        
        self.logger.info("Output done.")
        output = []
        for index, face in enumerate(response):
            output.append(response[index]['facial_area'])

        return JSONResponse(output)
    
facedetector_app = FaceDetector.bind()
