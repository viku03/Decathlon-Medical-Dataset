import os
import json
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
from scipy.ndimage import rotate
from tqdm import tqdm
import gc

class BrainTumor3DDataset:
    def __init__(self, base_path):
        self.base_path = base_path
        self.dataset_json = self._load_dataset_json()
        self.train_files = self._get_training_files()

    def _load_dataset_json(self):
        json_path = os.path.join(self.base_path, 'ML_Decathlon_Dataset/Task01_BrainTumour/dataset.json')
        with open(json_path, 'r') as f:
            return json.load(f)

    def _get_training_files(self):
        return self.dataset_json['training']

    def load_volume(self, file_path):
        full_path = os.path.join(self.base_path, 'ML_Decathlon_Dataset/Task01_BrainTumour', 
                                file_path.replace('./', ''))
        return nib.load(full_path).get_fdata()

    def preprocess_volume(self, volume):
        """Optimized preprocessing using vectorized operations"""
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
        images = []
        labels = []
        
        train_files = self.train_files[:num_samples] if num_samples else self.train_files
        
        for idx, file_info in tqdm(enumerate(train_files), desc="Loading data", total=len(train_files)):
            try:
                # Load and preprocess in chunks to save memory
                image = self.load_volume(file_info['image'])
                label = self.load_volume(file_info['label'])
                
                image = self.preprocess_volume(image)
                
                images.append(image)
                labels.append(label)
                
                # Force garbage collection after each sample
                gc.collect()
                
            except Exception as e:
                print(f"Error processing file {idx}: {e}")
                continue
            
        return images, labels

def train_model():
    # Initialize dataset
    dataset = BrainTumor3DDataset(base_path='.')
    
    print("Starting data preparation...")
    images, labels = dataset.prepare_data(num_samples=10)  # Reduced samples for testing
    
    print("Creating train/val split...")
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Clear memory
    del images, labels
    gc.collect()
    
    # Create data generators
    patch_size = (64, 64, 64)
    train_generator = DataGenerator3D(X_train, y_train, batch_size=1, patch_size=patch_size,
                                    augment=True)
    val_generator = DataGenerator3D(X_val, y_val, batch_size=1, patch_size=patch_size,
                                  augment=False)
    
    # Create and compile model
    input_shape = (*patch_size, 4)
    model = create_3d_unet(input_shape, n_classes=4)
    
    # Use a more memory-efficient optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)
    
    model.compile(optimizer=optimizer,
                 loss=dice_loss,
                 metrics=['accuracy'])
    
    # Memory-efficient callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_3d_model.keras',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode='min'
        ),
        # Memory cleanup callback
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: gc.collect()
        )
    ]
    
    # Train with simplified configuration for modern TensorFlow
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        callbacks=callbacks
    )
    
    return model, history

def create_3d_unet(input_shape, n_classes):
    """Memory-efficient U-Net implementation"""
    def conv_block(input_tensor, n_filters, kernel_size=(3, 3, 3)):
        x = layers.Conv3D(n_filters, kernel_size, padding='same', 
                         kernel_initializer='he_normal')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv3D(n_filters, kernel_size, padding='same',
                         kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    # Input layer
    inputs = layers.Input(input_shape)
    
    # Encoder path with skip connections
    conv1 = conv_block(inputs, 16)
    pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    conv2 = conv_block(pool1, 32)
    pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    # Bridge
    conv3 = conv_block(pool2, 64)
    
    # Decoder path
    up2 = layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv3)
    concat2 = layers.Concatenate()([up2, conv2])
    conv4 = conv_block(concat2, 32)
    
    up1 = layers.Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4)
    concat1 = layers.Concatenate()([up1, conv1])
    conv5 = conv_block(concat1, 16)
    
    # Output layer
    outputs = layers.Conv3D(n_classes, (1, 1, 1), activation='softmax')(conv5)
    
    return models.Model(inputs=inputs, outputs=outputs)

class DataGenerator3D(Sequence):
    def __init__(self, image_list, label_list, batch_size=1, patch_size=(64, 64, 64),
                 n_channels=4, n_classes=4, shuffle=True, augment=False):
        self.image_list = image_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        
        # Initialize valid_patches before calling on_epoch_end
        self.valid_patches = []
        self._calculate_valid_patches()
        self.on_epoch_end()

    def _calculate_valid_patches(self):
        valid_patches = []
        stride = [p // 2 for p in self.patch_size]
        
        for idx, image in enumerate(self.image_list):
            x_coords = range(0, image.shape[0] - self.patch_size[0], stride[0])
            y_coords = range(0, image.shape[1] - self.patch_size[1], stride[1])
            z_coords = range(0, image.shape[2] - self.patch_size[2], stride[2])
            
            for x in x_coords:
                for y in y_coords:
                    for z in z_coords:
                        patch = image[x:x + self.patch_size[0],
                                    y:y + self.patch_size[1],
                                    z:z + self.patch_size[2]]
                        if np.any(patch):
                            valid_patches.append((idx, x, y, z))
        
        self.valid_patches = valid_patches
        print(f"Total valid patches: {len(self.valid_patches)}")

    def __len__(self):
        return int(np.ceil(len(self.valid_patches) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.valid_patches))
        batch_patches = self.valid_patches[start_idx:end_idx]
        
        batch_size = len(batch_patches)
        X = np.zeros((batch_size, *self.patch_size, self.n_channels), dtype=np.float32)
        y = np.zeros((batch_size, *self.patch_size, self.n_classes), dtype=np.float32)

        for i, (img_idx, x, y_coord, z) in enumerate(batch_patches):
            X[i], y[i] = self._extract_patch(img_idx, x, y_coord, z)

        return X, y

    def _extract_patch(self, img_idx, x, y, z):
        image = self.image_list[img_idx]
        label = self.label_list[img_idx]

        patch_x = image[x:x + self.patch_size[0],
                       y:y + self.patch_size[1],
                       z:z + self.patch_size[2]].astype(np.float32)

        patch_y = np.zeros((*self.patch_size, self.n_classes), dtype=np.float32)
        for c in range(self.n_classes):
            patch_y[..., c] = (label[x:x + self.patch_size[0],
                                    y:y + self.patch_size[1],
                                    z:z + self.patch_size[2]] == c)

        if self.augment:
            patch_x, patch_y = self._augment_data(patch_x, patch_y)

        return patch_x, patch_y

    @staticmethod
    def _augment_data(image, label):
        if np.random.random() > 0.5:
            angle = np.random.uniform(-20, 20)
            image = np.stack([rotate(image[..., c], angle, axes=(0, 1), reshape=False)
                            for c in range(image.shape[-1])], axis=-1)
            label = np.stack([rotate(label[..., c], angle, axes=(0, 1), reshape=False)
                            for c in range(label.shape[-1])], axis=-1)
        return image, label

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.valid_patches)

# Update dice loss function for better numerical stability
def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    smooth = 1e-6  # Increased smoothing factor
    
    # Flatten the arrays
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

if __name__ == "__main__":
    print("Starting 3D training...")
    model, history = train_model()
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_3d.png')
    plt.close()
    
    model.save('final_3d_model.keras')