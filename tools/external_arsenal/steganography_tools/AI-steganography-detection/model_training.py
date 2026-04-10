import os
import numpy as np
import tensorflow as tf
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report


# ================== CONFIGURATION ==================
BASE_DIR = "<PATH/TO/>/IStego100K-master"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
MODEL_PATH = os.path.join(BASE_DIR, "steganography_detector.h5")

IMG_SIZE = (256, 256)
BATCH_SIZE = 32        
EPOCHS = 10            
LEARNING_RATE = 0.0001


# ================== DATA LOADING ==================
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)


# ================== MODEL ARCHITECTURE ==================
def build_model():
    """Defines the CNN model architecture."""
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


model = build_model()

# ================== CALLBACKS ==================
checkpoint = ModelCheckpoint(
    MODEL_PATH, 
    save_best_only=True, 
    save_weights_only=False, 
    monitor="val_accuracy", 
    mode="max", 
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss", 
    patience=3, 
    restore_best_weights=True
)


# ================== TRAINING ==================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

print(f"✅ Model saved as '{MODEL_PATH}'")


# ================== CLEANUP UNUSED FILES ==================
def delete_unused_files(directory):
    """Removes unnecessary files from dataset folders."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith(('.jpg', '.png', '.jpeg', '.h5')):  
                os.remove(os.path.join(root, file))

delete_unused_files(BASE_DIR)
print("✅ Unused files deleted to save space!")


# ================== MODEL EVALUATION ==================
val_images, val_labels = next(val_generator)
predictions = model.predict(val_images)
pred_labels = (predictions > 0.5).astype("int32")

print("\n📊 Classification Report:")
print(classification_report(val_labels, pred_labels))

# ================== PLOTTING RESULTS ==================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.show()
