import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import autokeras as ak
from keras.callbacks import EarlyStopping, LearningRateScheduler

# Directory containing the extracted real faces
real_faces_dir = r'D:\Bachelor\Implementation\data\real_and_fake_face\Extracted_faces_real'

# Directory containing the extracted fake faces
fake_faces_dir = r'D:\Bachelor\Implementation\data\real_and_fake_face\Extracted_faces_synthesis'

# Function to load and preprocess images from a directory
def load_and_preprocess_images_from_dir(faces_dir, label, max_images=4000, batch_size=16):
    images = []
    labels = []
    total_files = sum([1 for root, dirs, files in os.walk(faces_dir) for file in files if file.endswith('.jpg')])  # Counting files
    processed_files = 0
    loaded_images = 0

    for root, dirs, files in os.walk(faces_dir):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                image = cv2.imread(file_path)
                image = cv2.resize(image, (128, 128))  # Resize to match model input size
                image = image / 255.0  # Normalize pixel values
                images.append(image)
                labels.append(label)  # Assign label based on input parameter
                processed_files += 1
                loaded_images += 1
                print(f"Processed {processed_files}/{total_files} files", end='\r')
                
                # Check if the maximum number of images has been loaded
                if loaded_images >= max_images:
                    break
        else:
            continue  # Continue to the next iteration of the outer loop if the inner loop wasn't terminated by break
        break  # Break out of the outer loop if the maximum number of images has been loaded

    print("Images loaded successfully.")
    return images, labels


# Load and preprocess real images
real_images, real_labels = load_and_preprocess_images_from_dir(real_faces_dir, label=1)

# Load and preprocess fake images
fake_images, fake_labels = load_and_preprocess_images_from_dir(fake_faces_dir, label=0)
fake_images = fake_images[:3494]
fake_labels = fake_labels[:3494]

images = np.array(real_images + fake_images)
labels = np.array(real_labels + fake_labels)

X_train, X_val, y_train, y_val = train_test_split(images,labels, test_size=0.2, random_state=42)
 
print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)

# Check GPU availability
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPUs Available: ", len(physical_devices))

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


#Initialize the ImageClassifier
clf = ak.ImageClassifier(
    overwrite=True,  
    max_trials=10,
    seed=42,
    tuner='greedy'
)

# Define early stopping criteria
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=3,           # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore the best weights observed during training
)

# Define learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_schedule = LearningRateScheduler(lr_scheduler)
# Fit the model
history = clf.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_val, y_val),
    batch_size=8
    #, callbacks=[early_stopping]  # Pass the early stopping callback
    #,callbacks=[lr_schedule]  # Pass LR scheduling callback
)

# Evaluate the model on the validation set
evaluation = clf.evaluate(X_val, y_val)
print("Validation Loss:", evaluation[0])
print("Validation Accuracy:", evaluation[1])

# Export the model
model = clf.export_model()

model.summary()
model.save('model_autokeras_bayesian', save_format='tf')
