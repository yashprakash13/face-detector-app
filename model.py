from io import BytesIO

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

    @app.post("/detect-face")
    def detect_face(self, img_path: str) -> JSONResponse:
        # Run inference
        response = requests.get(img_path, stream=True)
        if response.status_code == 200:
            image_bytes = BytesIO(response.content)
            image_pil = Image.open(image_bytes)
            img_np = np.array(image_pil)
            response = DeepFace.extract_faces(img_path=img_np, detector_backend="mtcnn")        
            self.logger.info("Output done.")
            output = []
            for index, face in enumerate(response):
                output.append(response[index]['facial_area'])

            return JSONResponse(output)
        return JSONResponse({"Error": "Internal Server Error 500"})
    
facedetector_app = FaceDetector.bind()
