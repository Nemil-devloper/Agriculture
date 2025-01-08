from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from starlette.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from typing import Type

app = FastAPI()
MiddlewareType: Type[CORSMiddleware] = CORSMiddleware

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    MiddlewareType,
    allow_origins=origins,  # Allowed origins
    allow_credentials=True,  # Allow credentials (cookies, etc.)
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

MODEL = tf.keras.models.load_model("../models/1.keras")

CLASS_NAME = ["Early Blight","Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class=CLASS_NAME[np.argmax(predictions[0])]
    confidence=np.max(predictions[0])
    return{
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
