from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf 
import cv2
import os


app = FastAPI()

MODEL1 = tf.keras.models.load_model("2")
ResNet50Model  = tf.keras.models.load_model("Resnet200PT")
CLASS_NAMES = ["Potato Early Blight","Potato Late Blight","Healthy"]

@app.get("/ping")
async def ping():
    return "Connection Estabilished"

def read_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File()
):
    image = read_as_image(await file.read())

    image = tf.image.resize(image, (256, 256))
    image_reshape = np.expand_dims(image,0)

    
    predicted = MODEL1.predict(image_reshape)
    predicted2 = ResNet50Model.predict(image_reshape)
    result_class1 = CLASS_NAMES[np.argmax(predicted[0])]
    result_class2 = CLASS_NAMES[np.argmax(predicted2[0])]

    result_confidence1 = (float(np.max(predicted[0]))*100)
    rounded1 = f"{result_confidence1:.2f}"
    result_confidence2 = (float(np.max(predicted2[0]))*100)
    rounded2 = f"{result_confidence2:.2f}"

    print("Model 8L : "+result_class1)
    print("Model 8L : "+rounded1)
