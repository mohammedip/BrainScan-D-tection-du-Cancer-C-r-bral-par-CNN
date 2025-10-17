import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('best_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model()

st.title("Brain Tumor MRI Classifier 🧠")
st.write(
    "Upload an MRI scan of a brain, and this application will predict the "
    "type of tumor or if no tumor is present."
)

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"]
)

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

            # Prédiction
            prediction = model.predict(img_batch)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence_score = np.max(prediction) * 100

            st.success(f"Prediction: {predicted_class_name}")
            st.info(f"Confidence: {confidence_score:.2f}%")

elif model is None:
    st.warning("The model could not be loaded. Please check the model file path and integrity.")