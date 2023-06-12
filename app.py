import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import streamlit as st
from PIL import Image

st.title("Dog Cat Classifier")

# Step1: Load the Model
clf=load_model('cats_dogs_small_3.h5')

# Step2: Upload an Image for classification
uploaded_file=st.file_uploader('Choose a Database', accept_multiple_files=False)
if uploaded_file is not None:
    file=uploaded_file
else:
    file='image1.jpg'
if st.checkbox("View Image", False):
    image=Image.open(file)
    st.image(image)
# Step3: Preprocessing the loaded image
image=load_img(uploaded_file, target_size=(150, 150))
image=img_to_array(image)
image=np.expand_dims(image, axis=0)



# get the prediction and print the result
prediction=int(clf.predict(image)[0][0])

if st.button('Predict'):
    if prediction==1:
        st.subheader('The uploded image is a Dog 🐶')
    if prediction==0:
        st.subheader('The uploaded image is CAT 🐱')
