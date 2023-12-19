from io import BytesIO
from typing import List, Union

from fastapi.responses import JSONResponse
from deepface import DeepFace
from fastapi import FastAPI
import numpy as np
from PIL import Image
from pydantic import BaseModel
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

    @app.post("/detect-face")
    def detect_face(self, img_path: str) -> JSONResponse:
        # Run inference
        response = requests.get(img_path)
        img = Image.open(BytesIO(response.content))
        img_np = np.array(img)
        response = DeepFace.extract_faces(img_path=img_np, detector_backend="mtcnn")        
        self.logger.info("Output done.")
        output = []
        for index, face in enumerate(response):
            output.append(response[index]['facial_area'])

        return JSONResponse(output)
    
facedetector_app = FaceDetector.bind()
