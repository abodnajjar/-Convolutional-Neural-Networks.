
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "C:/Users/abdal/OneDrive/Desktop/CIFAR-10-images-master/train"
test_dir ="C:/Users/abdal/OneDrive/Desktop/CIFAR-10-images-master/test"


selected_classes = ['airplane', 'automobile', 'bird', 'cat', 'dog']

# Use ImageDataGenerator to load images and rescale pixel values
train_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Normalize the images to [0, 1]
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load the images from the directory, but filter by selected classes
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),  # CIFAR-10 images are 32x32 pixels
    batch_size=32,         # Number of images per batch
    class_mode='sparse',   # For multi-class classification, use 'sparse' (integer labels)
    shuffle=True,          # Shuffle the images for training
    classes=selected_classes  # Only load these classes
)
train_size = train_generator.samples
print(f"Total number of training images: {train_size}")

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(32, 32),  # CIFAR-10 images are 32x32 pixels
    batch_size=32,
    class_mode='sparse',   # 'sparse' means labels are integers
    shuffle=False,         # Don't shuffle test data, so we can evaluate
    classes=selected_classes  # Only load these classes
)
test_size = test_generator.samples
print(f"Total number of test images: {test_size}")

# Check the class labels
print("Class labels for training data:", train_generator.class_indices)

# Define the model

model = tf.keras.models.Sequential([    
    tf.keras. layers.Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras. layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      #tf.keras. layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu', input_shape=(32, 32, 3)),
    #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # 5 output units for the 5 classes
])
model.summary()
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using the image data generators
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy}")
########################################################################### B ################################################################################
model_simplified = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=2, activation='relu', input_shape=(32, 32, 3)),  # Non-overlapping filters
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),  # Downsample using average pooling
    tf.keras.layers.Flatten(),  # Flatten the feature map
    tf.keras.layers.Dense(64, activation='relu'),  # Fully connected hidden layer
    tf.keras.layers.Dense(5, activation='softmax')  # Output layer for classification
])

# Compile the simplified model
model_simplified.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

# Train the simplified model
history_simplified = model_simplified.fit(train_generator, epochs=10, validation_data=test_generator)

# Evaluate the simplified model
test_loss_simplified, test_accuracy_simplified = model_simplified.evaluate(test_generator)
print(f"Simplified Model Test Accuracy: {test_accuracy_simplified}")

######################################################################### C ###########################################################################################
y_true = test_generator.classes  # Actual labels

# Predictions for first model
y_pred_1 = model.predict(test_generator)
y_pred_labels_1 = np.argmax(y_pred_1, axis=1)

# Predictions for second (simplified) model
y_pred_2 = model_simplified.predict(test_generator)
y_pred_labels_2 = np.argmax(y_pred_2, axis=1)

# Calculate Accuracy
accuracy_1 = np.mean(y_pred_labels_1 == y_true)
accuracy_2 = np.mean(y_pred_labels_2 == y_true)

# Calculate Precision, Recall, and F1-Score for both models
precision_1 = precision_score(y_true, y_pred_labels_1, average='weighted')
recall_1 = recall_score(y_true, y_pred_labels_1, average='weighted')
f1_1 = f1_score(y_true, y_pred_labels_1, average='weighted')

precision_2 = precision_score(y_true, y_pred_labels_2, average='weighted')
recall_2 = recall_score(y_true, y_pred_labels_2, average='weighted')
f1_2 = f1_score(y_true, y_pred_labels_2, average='weighted')

# Print metrics for both models
print(f"Model 1 - Accuracy: {accuracy_1}")
print(f"Model 1 - Precision: {precision_1}")
print(f"Model 1 - Recall: {recall_1}")
print(f"Model 1 - F1-Score: {f1_1}")

print(f"Model 2 - Accuracy: {accuracy_2}")
print(f"Model 2 - Precision: {precision_2}")
print(f"Model 2 - Recall: {recall_2}")
print(f"Model 2 - F1-Score: {f1_2}")

###################################################################### D ##########################################################################################
model1_predictions = model.predict(test_generator, verbose=1)
model1_pred_classes = np.argmax(model1_predictions, axis=1)

# Get true labels for Model 1
model1_true_labels = test_generator.classes

# Generate confusion matrix for Model 1
conf_matrix1 = confusion_matrix(model1_true_labels, model1_pred_classes)

# Plot confusion matrix for Model 1
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix1, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.title('Confusion Matrix - Model 1')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

# Get predictions for Model 2
model2_predictions = model_simplified.predict(test_generator, verbose=1)
model2_pred_classes = np.argmax(model2_predictions, axis=1)

# Get true labels for Model 2
model2_true_labels = test_generator.classes

# Generate confusion matrix for Model 2
conf_matrix2 = confusion_matrix(model2_true_labels, model2_pred_classes)

# Plot confusion matrix for Model 2
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix2, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.title('Confusion Matrix - Model 2')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()