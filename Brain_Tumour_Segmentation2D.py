import os
import json
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# Configure MPS (Metal) for Mac M1/M2
print("TensorFlow version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())

try:
    # Check for MPS
    if len(tf.config.list_physical_devices('mps')) > 0:
        print("MPS (Metal) device found. Using Apple Silicon GPU.")
        mps_device = tf.device('/device:mps:0')
    else:
        print("No MPS device found. Using CPU.")
        mps_device = tf.device('/cpu:0')
except:
    print("Error configuring MPS. Falling back to CPU.")
    mps_device = tf.device('/cpu:0')

class BrainTumorDataset:
    def __init__(self, base_path, batch_size=2):
        self.base_path = base_path
        self.batch_size = batch_size
        self.dataset_json = self._load_dataset_json()
        self.train_files = self._get_training_files()
        
    def _load_dataset_json(self):
        json_path = os.path.join(self.base_path, 'ML_Decathlon_Dataset/Task01_BrainTumour/dataset.json')
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def _get_training_files(self):
        return self.dataset_json['training']
    
    def load_volume(self, file_path):
        full_path = os.path.join(self.base_path, 'ML_Decathlon_Dataset/Task01_BrainTumour', file_path.replace('./', ''))
        return nib.load(full_path).get_fdata()
    
    def preprocess_volume(self, volume):
        # Normalize each modality separately
        preprocessed = np.zeros_like(volume, dtype=np.float32)
        for i in range(volume.shape[-1]):
            modality = volume[..., i]
            nonzero_mask = modality != 0
            if np.any(nonzero_mask):
                mean = np.mean(modality[nonzero_mask])
                std = np.std(modality[nonzero_mask])
                if std != 0:
                    preprocessed[..., i] = (modality - mean) / std
        return preprocessed
    
    def prepare_data(self, num_samples=None):
        data = []
        labels = []
        
        train_files = self.train_files[:num_samples] if num_samples else self.train_files
        
        for idx, file_info in enumerate(train_files):
            print(f"Processing sample {idx+1}/{len(train_files)}")
            
            # Load image and label
            image = self.load_volume(file_info['image'])
            label = self.load_volume(file_info['label'])
            
            # Get middle slices from each axis for 2D training
            mid_z = image.shape[2] // 2
            
            # Extract middle slices and all modalities
            image_slice = image[:, :, mid_z, :]
            label_slice = label[:, :, mid_z]
            
            # Preprocess image
            image_slice = self.preprocess_volume(image_slice)
            
            # Convert label to one-hot encoding
            label_one_hot = tf.keras.utils.to_categorical(label_slice, num_classes=4)
            
            data.append(image_slice)
            labels.append(label_one_hot)
            
            # Free up memory
            del image, label
            
        return np.array(data), np.array(labels)

def create_unet_model(input_shape, num_classes):
    # Reduce model complexity for M1/M2 memory constraints
    base_filters = 32  # Reduced from 64
    
    inputs = layers.Input(input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(base_filters, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(base_filters, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(base_filters*2, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(base_filters*2, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(base_filters*4, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(base_filters*4, 3, activation='relu', padding='same')(conv3)
    
    # Decoder
    up2 = layers.Conv2DTranspose(base_filters*2, 2, strides=(2, 2), padding='same')(conv3)
    up2 = layers.concatenate([conv2, up2], axis=3)
    conv4 = layers.Conv2D(base_filters*2, 3, activation='relu', padding='same')(up2)
    conv4 = layers.Conv2D(base_filters*2, 3, activation='relu', padding='same')(conv4)
    
    up1 = layers.Conv2DTranspose(base_filters, 2, strides=(2, 2), padding='same')(conv4)
    up1 = layers.concatenate([conv1, up1], axis=3)
    conv5 = layers.Conv2D(base_filters, 3, activation='relu', padding='same')(up1)
    conv5 = layers.Conv2D(base_filters, 3, activation='relu', padding='same')(conv5)
    
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv5)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def dice_coefficient(y_true, y_pred):
    smooth = 1e-5
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = tf.reduce_mean((2. * intersection + smooth) / (union + smooth))
    return dice

def train_model():
    # Initialize dataset
    dataset = BrainTumorDataset(base_path='.')
    
    # Prepare data with smaller batch for M1
    print("Starting data preparation...")
    X, y = dataset.prepare_data(num_samples=30)  # Reduced from 50 for M1/M2
    print(f"Data loaded. Shape of X: {X.shape}, Shape of y: {y.shape}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Create and compile model
    input_shape = (240, 240, 4)
    with mps_device:
        model = create_unet_model(input_shape, num_classes=4)
    
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', dice_coefficient]
        )
    
        # Train model with smaller batch size
        history = model.fit(
            X_train, y_train,
            batch_size=1,  # Reduced from 2 for M1/M2
            epochs=30,     # Reduced from 50
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=5, 
                    restore_best_weights=True,
                    monitor='val_dice_coefficient',
                    mode='max'
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model.keras',  # Changed from .h5 to .keras
                    save_best_only=True,
                    monitor='val_dice_coefficient',
                    mode='max'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    patience=3,
                    factor=0.5,
                    monitor='val_loss',
                    mode='min'
                )
            ]
        )
    
    return model, history

if __name__ == "__main__":
    # Import layers and models here to ensure MPS is configured first
    from tensorflow.keras import layers, models
    
    print("Starting training...")
    model, history = train_model()
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coefficient'], label='Training Dice')
    plt.plot(history.history['val_dice_coefficient'], label='Validation Dice')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Save final model
    model.save('final_model.keras')