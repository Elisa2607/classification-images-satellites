# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 08:46:32 2026

@author: ML
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.models import model_from_json

# Lecture des données 

train_val = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,

)

train_test = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)


train_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    rotation_range=5,

)



train_generator_train = train_train.flow_from_directory(
    directory=r"C:\Users\ML\Desktop\G1_SIA2_images_spatiales\output\train",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

train_generator_test = train_test.flow_from_directory(
    directory=r"C:\Users\ML\Desktop\G1_SIA2_images_spatiales\output\test",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

train_generator_val = train_val.flow_from_directory(
    directory=r"C:\Users\ML\Desktop\G1_SIA2_images_spatiales\output\val",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)


# Train modèle

NB_CLASSES = 30
#build the model
# Tunning

base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

base_model.summary()

for layer in base_model.layers[:17]:
    layer.trainable = False
# Pour voir quelles sont les couches entraînables
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.trainable)

model = models.Sequential([
    base_model,
    layers.Flatten(), # ou layers.GlobalAveragePooling2D() suivant le cas
    layers.Dense(30, activation='softmax') 
    ])
#model = models.Sequential()
# CONV => RELU => POOL
#model.add(layers.Convolution2D(32, (3, 3), padding= 'same', input_shape=(200, 200, 3), activation='relu'))
#model.add(layers.BatchNormalization())
#model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 2 eme couche ajouté
#model.add(layers.Convolution2D(50, (3, 3), padding= 'same', input_shape=(200, 200, 3), activation='relu'))
#model.add(layers.BatchNormalization())
#model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 3 eme couvhe
#model.add(layers.Convolution2D(80, (3, 3), padding= 'same', input_shape=(200, 200, 3), activation='relu'))
#model.add(layers.BatchNormalization())
#model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


# Flatten
#model.add(layers.Flatten()) 
# softmax dense classifier
#model.add(layers.Dense(200, activation="relu"))
#model.add(layers.Dropout(0.5))

#
###############
# summary of the model
model.summary()



# compiling the model
model.compile(optimizer='SGD', loss='categorical_crossentropy',
              metrics=['accuracy'])
###############
#training the model
EPOCHS = 30
BATCH_SIZE = 32
VERBOSE = 1

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3,restore_best_weights=True)

model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

callbacks = [es, model_checkpoint]



history = model.fit(train_generator_train, epochs=EPOCHS,
		  verbose=VERBOSE, validation_data=train_generator_val, callbacks=callbacks)
################
#evaluate the model
test_loss, test_acc = model.evaluate(train_generator_test)
print('Test accuracy:', test_acc)

plt.plot(history.history['val_loss']) 
plt.plot(history.history['loss'])
plt.show()  # affichage fonction loss pour voir si on est pas bloque ds min local

#save du model
model_json = model.to_json()
with open ('model.json','w') as json_file:
    json_file.write(model_json)
    model.save_weights('model.h5')



# Rapport classification
np.max(history.history['val_accuracy'])
Y_pred = model.predict(train_generator_test)
yy_pred = np.argmax(Y_pred,1)
yy_test = np.argmax(train_generator_test,1)
mat = confusion_matrix(yy_test,yy_pred)
sns.heatmap(mat.T,square=True,annot=True,cbar=False,cmap=plt.cm.Blues,fmt='.0f')
plt.xlabel('valeur prédite')
plt.ylabel('valeurs réelles')
plt.show()



model_architecture = 'model.json'
model_weights = 'best_model.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

