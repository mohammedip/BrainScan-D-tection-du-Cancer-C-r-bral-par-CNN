import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS for redesign
def local_css():
    st.markdown("""
        <style>
        body {
            background-color: #f4f4f4;
        }
        .main-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 20px;
        }
        .upload-box {
            border: 2px dashed #2980b9;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            background-color: #ecf0f1;
        }
        .result-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model/best_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model()

st.markdown('<h1 class="main-title">Brain Tumor MRI Classifier 🧠</h1>', unsafe_allow_html=True)
st.write(
    "Upload an MRI scan of a brain, and this application will predict the type of tumor or if no tumor is present."
)

st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"]
)
st.markdown('</div>', unsafe_allow_html=True)

if model is not None and uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(
        image,
        caption='Uploaded MRI Scan',
        use_container_width=True
    )

    if st.button('Classify Image'):
        with st.spinner('Analyzing the image...'):
            target_size = (224, 224)
            resized_image = image.resize(target_size)

            img_array = np.array(resized_image) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_batch)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence_score = np.max(prediction) * 100

            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.success(f"Prediction: {predicted_class_name}")
            st.info(f"Confidence: {confidence_score:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

elif model is None:
    st.warning("The model could not be loaded. Please check the model file path and integrity.")