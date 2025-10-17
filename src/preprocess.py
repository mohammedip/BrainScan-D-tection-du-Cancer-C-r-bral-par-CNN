import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image


imgs = []
labels = []

base_dir = 'Data'  
img_size = (224,224)

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



# plt.hist(labels, bins=len(set(labels))+3, color='blue', edgecolor='black')
# plt.title("Data Histogramme")
# plt.xlabel("class")
# plt.ylabel("Fréquence")
# plt.show()

# for i in set(labels):
#     index = labels.index(i)  
#     cv2.imshow("échantillon d'image", imgs[index])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    


datagen = ImageDataGenerator(
    rotation_range=20,      
    width_shift_range=0.1,  
    height_shift_range=0.1, 
    zoom_range=0.1,         
    horizontal_flip=True    
)

class_counts = pd.Series(labels).value_counts()
max_count = class_counts.max()

img_balanced = []
label_balanced = []

save_dir = 'augmented_images'
os.makedirs(save_dir, exist_ok=True)

for cls in class_counts.index:
    img_cls = np.array(imgs)[np.array(labels) == cls].astype('float32') / 255.0
    
    if img_cls.ndim == 3:
        img_cls = np.expand_dims(img_cls, 0)
    
    label_cls = [cls] * len(img_cls)
    img_balanced.extend(img_cls)
    label_balanced.extend(label_cls)
    
    n_to_generate = max_count - len(img_cls)
    
    if n_to_generate > 0:
        gen = datagen.flow(img_cls, batch_size=1, shuffle=True)
        
        for i in range(n_to_generate):
            img_aug = next(gen)[0]
            
            img_to_save = (img_aug * 255).astype(np.uint8)

            img_aug = img_aug.astype('float32') / 255.0
            img_balanced.append(img_aug)
            label_balanced.append(cls)
            
            
            # cv2.imshow("Augmented", img_to_save)
            # cv2.waitKey(100)
            # cv2.destroyAllWindows()
            
            # filename = f"{cls}_aug_{i}.png"
            # filepath = os.path.join(save_dir, filename)
            # Image.fromarray(img_to_save).save(filepath)


img_balanced = np.array(img_balanced)
label_balanced = np.array(label_balanced)

encoder = LabelEncoder()
label_encoded = encoder.fit_transform(label_balanced)       

# print(label_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    img_balanced, label_encoded, test_size=0.2, random_state=42
)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
