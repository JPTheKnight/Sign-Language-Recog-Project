import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report

tf.__version__ 

print(tf.config.experimental.list_physical_devices("GPU"))

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255, 
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

training_set = train_datagen.flow_from_directory("D:/sign_language/our-sign-language/dataSet/trainingData",                             
                                                 target_size=(128, 128),
                                                 batch_size=10,
                                                 color_mode='grayscale',                                
                                                 class_mode='categorical')

    
test_set = test_datagen.flow_from_directory("D:/sign_language/our-sign-language/dataSet/testingData",
                                            target_size=(128, 128),                                  
                                            batch_size=10,        
                                            color_mode='grayscale',
                                            class_mode='categorical')

classifier = tf.keras.models.Sequential()

classifier.add(tf.keras.layers.Flatten(input_shape=[128, 128, 1]))

classifier.add(tf.keras.layers.Dense(units=256, activation='relu'))

classifier.add(tf.keras.layers.Dropout(0.50))

classifier.add(tf.keras.layers.Dense(units=128, activation='relu'))

classifier.add(tf.keras.layers.Dropout(0.50))

classifier.add(tf.keras.layers.Dense(units=64, activation='relu'))

classifier.add(tf.keras.layers.Dense(units=27, activation='softmax'))

classifier.compile(optimizer='adam', 
                   loss='categorical_crossentropy', 
                   metrics=['accuracy'])

classifier.fit(training_set,
               epochs=5,
               validation_data=test_set)

print(classifier.summary())

# Make predictions on the test set
y_pred = classifier.predict(test_set)
y_pred = np.argmax(y_pred, axis=1)  # Convert predictions to class labels

# Get the ground truth labels
y_true = test_set.classes

# Get the class labels
class_labels = list(test_set.class_indices.keys())

# Generate the classification report
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights('model.h5')
print('Weights saved')