
from preprocess import np , os , X_train , X_test, y_train , y_test
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time


learning_rate = 0.001
epochs = 35
batch_size = 64
input_shape = (224, 224, 3)
num_classes = 4 


print("X_train shape:", X_train.shape)
print("X_test  shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test  shape:", y_test.shape)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape, padding='same'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    

    layers.Flatten(),  
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])


optimizer = optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
try:
    plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
    print("Saved model architecture plot to model_architecture.png")
except Exception as e:
    print("plot_model failed (graphviz/pydot missing?). Exception:", e)

checkpoint_path = 'model/best_model.h5'
checkpoint_cb = ModelCheckpoint(filepath=checkpoint_path,
                                monitor='val_accuracy',
                                save_best_only=True,
                                verbose=1)

start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[checkpoint_cb],
    verbose=1
)
end_time = time.time()
print(f"\nTraining time: {(end_time - start_time) / 60:.2f} minutes")

best_model = tf.keras.models.load_model(checkpoint_path)

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest loss: {test_loss:.4f}  |  Test accuracy: {test_acc:.4f}")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png') 
plt.show()

y_pred_probs = best_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig('confusion_matrix.png')
plt.show()

def show_examples(X, y_true, y_pred, correct=True, samples=8):
    """Affiche 'samples' images correctes ou incorrectes."""
    indices = []
    for i in range(len(y_true)):
        if correct and (y_true[i] == y_pred[i]):
            indices.append(i)
        if (not correct) and (y_true[i] != y_pred[i]):
            indices.append(i)
        if len(indices) >= samples:
            break

    if len(indices) == 0:
        print("No examples found for correct=" + str(correct))
        return

    cols = min(4, samples)
    rows = (len(indices) + cols - 1) // cols
    plt.figure(figsize=(cols*3, rows*3))
    for idx, i in enumerate(indices):
        ax = plt.subplot(rows, cols, idx+1)
        img = X[i]
        if img.shape[-1] == 3:
            img_disp = img[..., ::-1]  
        else:
            img_disp = img.squeeze()
        plt.imshow(img_disp)
        plt.title(f"T:{y_true[i]} P:{y_pred[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("\n--- Examples of correct predictions ---")
show_examples(X_test, y_true, y_pred, correct=True, samples=8)

print("\n--- Examples of incorrect predictions ---")
show_examples(X_test, y_true, y_pred, correct=False, samples=8)
