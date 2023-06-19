import streamlit as st
import requests
from PIL import Image
import os
import numpy as np

st.title('Predictions of Artworks in Metropolitan Museum of New-York City')
uploaded_image = st.file_uploader("Choose a picture", accept_multiple_files=False)

if uploaded_image is not None:
    # Charger l'image avec PIL
    image = Image.open(uploaded_image)
    
    # Afficher l'image dans Streamlit
    col1, col2, col3 = st.columns(3)
    col2.image(image, caption='Uploaded Image')
    
    if col2.button('Predict the culture'):
        res = requests.post("http://127.0.0.1:8000/predict/", files={"file":uploaded_image.getvalue()})
        if res.status_code == 200:
            json_data = res.json()
            culture = json_data['class_label']
            col2.write(f"This artwork is {culture} !")

            list_flags = os.listdir('Applications/flags/')
            goodflag = list_flags[np.argmax([culture in x for x in list_flags])]
            flag = Image.open("Applications/flags/"+goodflag)
            col2.image(flag)

        else :
            raise ValueError('Status code is not 200, something went wrong')
