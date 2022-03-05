import keras
from PIL import Image, ImageOps
import numpy as np
import streamlit as st


st.title("Image Classification using CNN Algorithm")
st.header("Solar Panel Classification Example")
st.text("Upload a Solar Panel Image for image classification as defective or non-defective")

def load_model():
    model = keras.models.load_model('best_model_improved.h5')
    return model
model = load_model()
    
def image_classification(img, model):
    
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 64, 64, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (64, 64)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    
    return prediction

uploaded_file = st.file_uploader("Choose a Solar Panel Image ...", type=["jpg",'png','jpeg'])
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        prediction = image_classification(image,model)
        class_names = ['Defective','Non-Defective']
        string = "This image is :"+class_names[np.argmax(prediction)]
        st.success(string)