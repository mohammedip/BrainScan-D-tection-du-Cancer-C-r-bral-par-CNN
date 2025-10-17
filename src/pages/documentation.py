import streamlit as st

st.set_page_config(page_title="📖 Documentation", page_icon="📘")

st.title("📖 Project Documentation")
st.write("Welcome to the **Brain Tumor MRI Classifier** documentation page!")

st.markdown("""
## 🧠 Overview
This project is a **deep learning-based system** that classifies brain MRI images into four categories:
- Glioma Tumor  
- Meningioma Tumor  
- Pituitary Tumor  
- No Tumor  

It uses a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras and trained on a dataset of labeled MRI scans.

---

## ⚙️ Workflow
1. **Data Loading & Preprocessing**
   - Images are read from folders (`Data/` directory) corresponding to each tumor class.
   - They are resized to **224×224 pixels** and normalized.
   - Classes are balanced using **data augmentation** (rotation, zoom, flips, shifts).
   - Labels are one-hot encoded for training.

2. **Model Architecture**
   - 3 convolutional blocks with ReLU activation and MaxPooling.
   - Dropout layers for regularization.
   - Fully connected dense layer (256 units) and softmax output layer.
   - Optimizer: **Adam**, Loss: **categorical_crossentropy**.

3. **Training**
   - 35 epochs, batch size 64.
   - Validation split: 20%.
   - Uses `ModelCheckpoint` to save the best model based on validation accuracy.

4. **Evaluation**
   - Generates accuracy/loss curves.
   - Confusion matrix and classification report.
   - Visualizes correct and incorrect predictions.

5. **Deployment**
   - A Streamlit web interface allows users to upload an MRI image.
   - The trained model predicts the tumor type and displays the result with confidence score.

---

## 🧩 Technologies Used
| Component | Technology |
|------------|-------------|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Data Augmentation | ImageDataGenerator |
| Web App | Streamlit |
| Visualization | Matplotlib, Seaborn |
| Image Processing | OpenCV, PIL |
| Data Handling | NumPy, Pandas |

---

## 📊 Results
- High validation accuracy on the test dataset.
- Model generalizes well across the four classes.
- Real-time prediction via Streamlit app.

---

## 🧱 File Structure
📁 project_root/
│
├── preprocess.py # Data loading, augmentation, splitting
├── train_model.py # Model creation and training
├── model/
│ └── best_model.h5 # Saved trained model
├── Home.py # Streamlit main interface
└── pages/
└── 1_Documentation.py # Documentation page
            """)