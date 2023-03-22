import streamlit as st
import pandas as pd
from io import StringIO
import cv2 
import numpy as np
from keras.applications import ResNet50
from keras.models import Model, Sequential, load_model
import time
from Inference import *

model = create_model()
model.load_weights("final_model.h5")
vocab = np.load("vocab.npy", allow_pickle=True).item()

def resnet():
    resnet_model = ResNet50(include_top=True)
    resnet_model = Model(inputs=resnet_model.input, outputs=resnet_model.layers[-2].output)
    return resnet_model

st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "webp"])

if uploaded_file is not None:
    # Read the image data from the uploader
    image_data = uploaded_file.read()

    # Convert the binary image data to a NumPy array
    image_np = np.frombuffer(image_data, np.uint8)

    # Decode the image data and convert it to a NumPy array
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    print("hello")
    print(image)

    # Convert the color space from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to 224x224
    image = cv2.resize(image, (224, 224))

    # Reshape the image to a 4D tensor with a single sample
    image = image.reshape(1, 224, 224, 3)

    # Display the processed image to the user
    st.image(image_data, caption='Processed Image', use_column_width=True)

col1, col2 = st.columns(2)

options = ["Greedy Search", "Beam Search", "Nucleus Sampling"]
selected_option = col1.radio("Choose an decoding strategy:", options)

b = col1.button("Generate Caption")

if b:
    try:
        with st.spinner('Loading model...'):
            encoder = resnet()
            feature_vector = encoder.predict(image, verbose=0).reshape(2048,)
            # decoder = load_model("image_caption_gen_epoch_20.h5")
        if selected_option == "Greedy Search":
            with st.spinner('Generating Caption'):
                result = decode_caption(generate_caption(np.array([feature_vector]), model, vocab), vocab)
                col2.write(result)
        elif selected_option == "Beam Search":
            with st.spinner('Generating Caption'):
                fv = np.array([feature_vector])
                complete_captions = beam_search(fv, model, 5, 5, vocab)
                sorted_list = sorted(complete_captions, key=lambda x: x[1])
                result = decode_caption(sorted_list[-1][0], vocab)
                col2.write(result)
        elif selected_option == "Nucleus Sampling":
            with st.spinner('Generating Caption'):
                fv = np.array([feature_vector])
                result = decode_caption(generate_caption_nucleus_sampling(fv, model, vocab), vocab)
                col2.write(result)
    except NameError:
        st.warning("Please Upload Image First")