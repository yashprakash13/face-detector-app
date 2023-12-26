import logging

from fastapi.responses import JSONResponse
from fastapi import FastAPI
from ray import serve

from model import perform_detection_and_return


app = FastAPI()

logger = logging.getLogger("ray.serve")


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.5, "num_gpus": 0})
@serve.ingress(app)
class FaceDetector:
    def __init__(self):
        # Load model
        logger.info("Initialized model.")

    @app.post("/detect-face")
    def detect_face(self, img_path: str) -> JSONResponse:
        # Run inference
        logger.info(f"Img path: {img_path}")
        output = perform_detection_and_return(img_path=img_path)
        if output:
            return JSONResponse(output)
        else:
            return JSONResponse({"Error": "Internal Server Error 500"})
    
facedetector_app = FaceDetector.bind()
