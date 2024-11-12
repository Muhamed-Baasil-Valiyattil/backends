from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf 


app = FastAPI()

MODEL = tf.keras.models.load_model("modelv1.2")
CLASS_NAMES = ["Potato Early Blight","Potato Late Blight","Healthy"]

@app.get("/ping")
async def ping():
    return "Working..."

def read_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File()
):
    image = read_as_image(await file.read())
    image_reshape = np.expand_dims(image,0)
    
    predicted = MODEL.predict(image_reshape)
    result_class = CLASS_NAMES[np.argmax(predicted[0])]
    result_confidence = float(np.max(predicted[0]))
    print(result_class)
    print(result_confidence)
    return {"Predicted Class":result_class,"Confidence":result_confidence}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port = 8000)
