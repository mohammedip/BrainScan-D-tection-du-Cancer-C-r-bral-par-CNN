import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model('model/best_model.h5')

class_names = ['glioma', 'meningioma', 'notumor' , "pituitary"]  

def preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess an image."""
    img = image.load_img(img_path, target_size=target_size)  
    img_array = image.img_to_array(img) 
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def predict_image(img_path):
    img = preprocess_image(img_path, target_size=(224, 224)) 
    preds = model.predict(img)
    pred_index = np.argmax(preds, axis=1)[0]
    print(pred_index)
    pred_class = class_names[pred_index]
    pred_prob = preds[0][pred_index]
    print(f"Predicted class: {pred_class} with probability {pred_prob:.2f}")
    return pred_class, pred_prob

predict_image('Pituitary-macroadenoma-CT-01.jpg')
predict_image('WhatsApp Image 2025-10-17 à 15.15.23_536c78fb.jpg')
