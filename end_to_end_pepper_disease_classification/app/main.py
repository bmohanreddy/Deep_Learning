from fastapi import FastAPI, Request, File, UploadFile
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from io import BytesIO
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
MODEL = load_model('C:/Users/mohan/Desktop/Deep-Learning/end_to_end_pepper_disease_classification/saved_models/pepper_model_v1.h5')
CLASS_NAMES = ['Bacterial Spot','Healthy']



@app.get("/")
async def home(request:Request):
    return templates.TemplateResponse("index1.html", {"request":request})

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
@app.post("/predict")
async def predict(request:Request, file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image,0)
    prediction = MODEL.predict(image_batch)
    predict_class =CLASS_NAMES[ np.argmax(prediction[0])]
    confidance = np.max(prediction[0])
    result = {"class_name":predict_class,"confidance":confidance}
    return templates.TemplateResponse("index1.html", {"request":request,"result":result})


if __name__ == '__main__':
    uvicorn.run(app, host ='localhost', port=8000)
