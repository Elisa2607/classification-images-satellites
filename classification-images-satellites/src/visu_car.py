# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 08:18:38 2026

@author: ML
"""

from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Model
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.applications.vgg19 import VGG19

train_test = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

train_generator_test = train_test.flow_from_directory(
    directory=r"C:\Users\ML\Desktop\G1_SIA2_images_spatiales\output\test",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
    seed=42
)

model_architecture = 'model.json'
model_weights = 'best_model.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)
model.summary()

base_model = model.layers[0]
base_model.summary()

# Récupérer les filtres et les poids de la 1ère couche
filters, biases, *is_anything_else_being_returned = model.layers[0].get_weights()
# normaliser les filtres sur [0 , 1]
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Prendre le deuxième filtre de cette couche (qui contient 3 sous-filtres pour les 3 canaux R,
# V et et B) et visualiser ces 3 sous-filtres
plt.figure()
f=filters[:,:,:,1]
plt.subplot(1,3,1)
plt.imshow(f[:, :, 0], cmap='gray')
plt.subplot(1,3,2)
plt.imshow(f[:, :, 1], cmap='gray')
plt.subplot(1,3,3)
plt.imshow(f[:, :, 2], cmap='gray')
plt.close()


fichier = r"C:\Users\ML\Desktop\G1_SIA2_images_spatiales\AID\Stadium\stadium_83.jpg"
image = Image.open(fichier)
# #im_array = np.array(image)
# #im_fl = np.float64(im_array)
# #new_shape = (224,224,3)
# im_res=np.resize(im_fl, new_shape)/255.0
# im_fin = np.reshape(im_res, (1,224,224,3))
image = np.asarray(image)
image=image.astype('float32')
image=cv2.resize(image,(224,224))
image /= 255
image = image.reshape([-1,224,224,3])


# Définir un modèle intermédiaire contenant les 2 premières couches du modèle initial

inter_model = Model(inputs=base_model.inputs, outputs=base_model.layers[1].output)
inter_model.summary()
feature_maps = inter_model.predict(image)
plt.figure()
plt.imshow(feature_maps[0,:,:,0])


# Analyse des résultats 


y_pred_prob=model.predict(train_generator_test)
y_pred=np.argmax(y_pred_prob, 1)
labels = train_generator_test.classes



print(confusion_matrix(labels, y_pred))
print(classification_report(labels, y_pred))
im = confusion_matrix(labels, y_pred)
sns.heatmap(im.T,square=True,annot=False,cbar=True,cmap=plt.cm.Reds,fmt='.0f')

# labels=labels.astype('float32')

# print(np.where(y_pred != labels))
# plt.imshow(train_generator_test[1007,:,:])
# print(y_pred(1007))
# print(labels(1007))


classifier = VGG19(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=30,
    classifier_activation='softmax'
)

classifier.save("image_spatiale.hdf5")


 