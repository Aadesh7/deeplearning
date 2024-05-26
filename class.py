import os
from PIL import Image, ImageOps
import numpy as np
from scipy import ndimage
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2


def load_images(directory, target_size):
    images = []
    labels = []
    class_names = os.listdir(directory) # class names as per the folder name
    # creation of dictionary for enumeration
    class_to_index = {class_name: i for i, class_name in enumerate(class_names)}
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name + "/train")
        class_label = class_to_index[class_name]
        for filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, filename)
            image = Image.open(image_path).convert("RGB")
            
            padded_image = ImageOps.pad(image, target_size, method=Image.Resampling.LANCZOS)

            images.append(np.array(padded_image))
            labels.append(np.array(class_label))

    return images, labels

def augment_images(images, labels):
    augmented_images = []
    augmented_labels = []

    for i, image in enumerate(images):
        
        augmented_images.append(image)
        augmented_labels.append(labels[i])

        # Random horizontal flip
        if np.random.rand() < 0.9:
            flipped_image = np.fliplr(image)
            augmented_images.append(flipped_image)
            augmented_labels.append(labels[i])

        # Random rotation
        if np.random.rand() < 0.9:
            angle = np.random.randint(-15, 15)  # Rotate between -15 and +15 degrees
            rotated_image = ndimage.rotate(image, angle, reshape=False, mode='nearest')
            augmented_images.append(rotated_image)
            augmented_labels.append(labels[i])

        # add gaussian noise
        if np.random.rand() < 0.9:
            noise = np.random.normal(0, 0.1, image.shape)
            noisy_image = image + noise
            noisy_image = np.clip(noisy_image, 0, 1)
            augmented_images.append(noisy_image)
            augmented_labels.append(labels[i])

    return augmented_images, augmented_labels

def cnn_model(num_classes):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:], kernel_regularizer=L2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=L2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=L2(0.001)))
    # model.add(layers.BatchNormalization())
    # model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu', kernel_regularizer=L2(0.001)))
    model.add(layers.Dropout(0.5))

    # Fully connected layers
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=L2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=L2(0.001)))

    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

data_dir = 'C:/Users/Aadesh/Desktop/py/animal_data'
target_size = (160,160)

images, labels = load_images(data_dir, target_size)
images, labels = augment_images(images, labels)

print(len(images))

images = np.array(images)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# normalize
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# labels to one-hot encoding
num_classes = len(np.unique(labels))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

batch_size = 32

model = cnn_model(num_classes) #created a CNN model
print(model.summary())

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=40, batch_size=batch_size, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test accuracy:", test_acc)

