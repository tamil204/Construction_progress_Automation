# %%
#Import neccessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import imgaug.augmenters as iaa
import os
from PIL import Image
import numpy as np
import imagehash
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
#Load the dataset
dataset_path = r'C:\Users\Tamilselvi.R\OneDrive\Desktop\construction-sih'

datagen = ImageDataGenerator(
    rescale=1./255,  
    validation_split=0.2  
)

# Create training data generator
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),  
    batch_size=32,          
    class_mode='categorical', 
    subset='training'         
)

# Create validation data generator
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),  
    batch_size=32,
    class_mode='categorical',
    subset='validation'    
)  
print(train_generator.class_indices)


# %%
# Paths to directories
extracted_directory = r'C:\Users\Tamilselvi.R\OneDrive\Desktop\construction-sih'
processed_directory = r'C:\Users\Tamilselvi.R\OneDrive\Desktop\preprocessing\resize_normailze_dup_remov'
os.makedirs(processed_directory, exist_ok=True)

# Define target size for resizing
target_size = (800, 800)  
def normalize_image(image):
    """Normalize image to [0, 1] range and convert back to 8-bit image."""
    img_array = np.array(image)  
    normalized_array = img_array / 255.0  
    return Image.fromarray((normalized_array * 255).astype(np.uint8))  

def resize_image(image, size):
    """Resize image to the specified size."""
    return image.resize(size, Image.LANCZOS)  

def get_image_hash(image):
    """Compute the hash of an image to identify duplicates."""
    return imagehash.average_hash(image)

def process_images():
    seen_hashes = set()
    for root, dirs, files in os.walk(extracted_directory):
        relative_path = os.path.relpath(root, extracted_directory)
        save_path = os.path.join(processed_directory, relative_path)
        os.makedirs(save_path, exist_ok=True)

        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    # Normalize the image
                    normalized_img = normalize_image(img)
                    # Resize the normalized image
                    resized_img = resize_image(normalized_img, target_size)
                    # Compute the hash of the image
                    img_hash = get_image_hash(resized_img)
                    
                    if img_hash not in seen_hashes:
                        seen_hashes.add(img_hash)
                        processed_image_path = os.path.join(save_path, file)
                        resized_img.save(processed_image_path)
                        print(f"Processed and saved: {processed_image_path}")
                    else:
                        print(f"Duplicate image skipped: {file_path}")
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")
process_images()
print("Image preprocessing complete!")



# %%
# Define augmentation sequence
seq = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),  
    iaa.Multiply((0.8, 1.2)),  
    iaa.ContrastNormalization((0.75, 1.5)), 
])

# %%
# Paths to directories
processed_directory = r'C:\Users\Tamilselvi.R\OneDrive\Desktop\preprocessing\resize_normailze_dup_remov'
augmented_directory = r'C:\Users\Tamilselvi.R\OneDrive\Desktop\preprocessing\augmentation'
os.makedirs(augmented_directory, exist_ok=True)
seq = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),  
    iaa.Multiply((0.8, 1.2)),  
    iaa.ContrastNormalization((0.75, 1.5)),  
])

def augment_images():
    for root, dirs, files in os.walk(processed_directory):
        relative_path = os.path.relpath(root, processed_directory)
        save_path = os.path.join(augmented_directory, relative_path)
        os.makedirs(save_path, exist_ok=True)

        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img_array = np.array(img)
                    for i in range(4):
                        augmented_img_array = seq.augment_image(img_array)
                        augmented_image = Image.fromarray(augmented_img_array)
                        augmented_image_path = os.path.join(save_path, f"{os.path.splitext(file)[0]}_aug_{i}.jpg")
                        augmented_image.save(augmented_image_path)
                        print(f"Augmented and saved: {augmented_image_path}")
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")
augment_images()
print("Image augmentation complete!")


# %%
# Paths to directories
augmented_directory = r'C:\Users\Tamilselvi.R\OneDrive\Desktop\preprocessing\augmentation'
split_directory = r'C:\Users\Tamilselvi.R\OneDrive\Desktop\model'
train_directory = os.path.join(split_directory, 'train')
test_directory = os.path.join(split_directory, 'test')
os.makedirs(train_directory, exist_ok=True)
os.makedirs(test_directory, exist_ok=True)

def split_data():
    for subdir in os.listdir(augmented_directory):
        subdir_path = os.path.join(augmented_directory, subdir)
        if os.path.isdir(subdir_path):
            # Create corresponding directories for train and test
            train_subdir = os.path.join(train_directory, subdir)
            test_subdir = os.path.join(test_directory, subdir)
            os.makedirs(train_subdir, exist_ok=True)
            os.makedirs(test_subdir, exist_ok=True)
            files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
            
            # Split the files into training and testing
            train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
            
            for file in train_files:
                shutil.copy(os.path.join(subdir_path, file), os.path.join(train_subdir, file))
            for file in test_files:
                shutil.copy(os.path.join(subdir_path, file), os.path.join(test_subdir, file))
            
            print(f"Split data for {subdir} into train and test sets")
split_data()
print("Data split into training and testing sets complete!")


# %%
# Paths to directories
train_directory = r'C:\Users\Tamilselvi.R\OneDrive\Desktop\model\train'
test_directory = r'C:\Users\Tamilselvi.R\OneDrive\Desktop\model\test'
train_datagen = ImageDataGenerator(
    rescale=1./255,           
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(150, 150),    
    batch_size=32,
    class_mode='categorical'  
)
test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)


# %%
# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  
])


# %%
# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# %%
# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)


# %%
# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')


# %%
# Save the model
model.save('cnn_classification_model.h5')
print("Model training complete and saved as cnn_classification_model.h5")

# %%
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# %%
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load your trained model
model = tf.keras.models.load_model('cnn_classification_model.h5')
def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    return img_array
def predict_class(img_path, model):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class
def get_class_names(train_generator):
    class_labels = train_generator.class_indices
    return list(class_labels.keys())
# Path to your test image
img_path = r'C:\Users\Tamilselvi.R\Downloads\verify30.jpg'  

train_directory = r'C:\Users\Tamilselvi.R\OneDrive\Desktop\model\train'
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(150, 150),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)
class_names = get_class_names(train_generator)
# Predict the class of the test image
predicted_class_index = predict_class(img_path, model)
predicted_class_name = class_names[predicted_class_index]
print(f'Predicted class: {predicted_class_name}')
# Display the test image
def show_image(img_path):
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted: {predicted_class_name}')
    plt.show()

show_image(img_path)




