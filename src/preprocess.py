import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

imgs = []
labels = []

base_dir = 'Data'  
img_size = (244,244)

for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)

    for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(folder_path, filename)
                    try:
                        img = cv2.imread(img_path) 
                        img = cv2.resize(img , img_size)
                        imgs.append(img)
                        labels.append(folder_name)
                    except Exception as e:
                          print(f"Error : {e}")    


print(labels)