from fastapi import FastAPI, UploadFile
import tensorflow as tf
import cv2
import numpy as np
import os


app = FastAPI()

#load the model
model_path = os.path.join('model', 'vgg16_culture_customized.h5')
model = tf.keras.models.load_model(model_path)


@app.get("/")
def read_root():
    return {"Hello": "World"}


#this POST returns the class of the image uploaded
@app.post("/predict/")
async def predict(file: UploadFile):
    image_size = (224, 224) #specific for VGG16
    content = await file.read()
    image_path = np.frombuffer(content, np.uint8)
    image_to_predict = cv2.imdecode(image_path,cv2.IMREAD_COLOR)
    img_to_predict = np.expand_dims(cv2.resize(image_to_predict, image_size), axis=0) 

    #prediction of the class by the model
    probabilities = np.squeeze(model.predict(img_to_predict))
    res = np.argmax(probabilities)
    class_names = ['British', 'Chinese', 'French',
                   'German', 'Greek', 'Iran',
                   'Italian', 'Japanese', 'Spanish']
    prediction = class_names[res] #to have the name of the class 

    return {"class_label": prediction}


   