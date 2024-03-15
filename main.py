import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd

# Dataset paths
TRAIN_DIR = 'Garbage/processed_images' 
TEST_DIR = 'Garbage/original_images'

# Model hyperparameters
IMG_SIZE = (32, 32)
BATCH_SIZE = 32
EPOCHS = 20

# Load and preprocess data
train_datagen = ImageDataGenerator(rescale=1./255) 
train_data = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE)

test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE) 

# Define model
model = Sequential([
  Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
  MaxPooling2D(),
  
  Conv2D(64, 3, activation='relu'),
  MaxPooling2D(),

  Conv2D(128, 3, activation='relu'),
  MaxPooling2D(),

  Flatten(),
  Dense(512, activation='relu'),
  Dense(5, activation='softmax') 
])

# Compile and train model
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy']
)

model.fit(
  train_data,
  validation_data=test_data,
  epochs=EPOCHS
)

# Evaluate model on test set  
test_loss, test_acc = model.evaluate(test_data)
print('Test accuracy:', test_acc)

# Generate predictions and confusion matrix
predictions = model.predict(test_data)
cm = confusion_matrix(test_data.classes, predictions.argmax(axis=1))

model.save('garbage_classification.h5')

test_img = 'Garbage/processed_images/cardboard/cardboard_321.jpg'
img = ku.load_img(test_img, target_size = (32,32))
img = ku.img_to_array(img, dtype=np.uint8)
img = np.array(img)/255.0
prediction = model.predict(img[np.newaxis, ...])

#print("Predicted shape",p.shape)
print("Probability:",np.max(prediction[0], axis=-1))
predicted_class = number_to_class[np.argmax(prediction[0], axis=-1)]
print("Classified:",predicted_class,'\n' , number_to_class[np.argmax(test_img[0])])

plt.axis('off')
plt.imshow(img.squeeze())
plt.imshow(img)
plt.title("Loaded Image")

# Plot confusion matrix
matrix = pd.DataFrame(cm, range(5), range(5))
sns.heatmap(matrix, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class') 
plt.ylabel('True Class')
plt.show()



print(classification_report(test_data.classes, predictions.argmax(axis=1)))