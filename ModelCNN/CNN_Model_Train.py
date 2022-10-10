"""
#########################################################
Importar Librerás básicas de procesamiento información
#########################################################
"""
import os
import cv2
import numpy as np

#Preprocesamiento de imagenes a entrenar
from tensorflow.keras.preprocessing import image

#Crear un excel de información
from tensorflow.keras.callbacks import CSVLogger

#Librería tratamiento arreglos
from keras.utils import np_utils

#Correlacionar matrices
from sklearn.utils import shuffle

"""
#########################################################
Importar Librerás de entrenamiento con TL para una CNN
#########################################################
"""
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.metrics import *

#Data Augmentation para incremento de datos 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

"""
#########################################################
Parametrización datos de entrenamiento
#########################################################
"""

epochs = 30
batch_size = 20
num_classes = 2
width_shape = 224
height_shape = 224

"""
#################################################################
Dirección local carpetas con datos de entrenamiento y validación
#################################################################
"""

train_path = 'E:/ParkingManagerCNN/DataSet/DosClases/train'  
validation_path = 'E:/ParkingManagerCNN/DataSet/DosClases/validation'

"""
#################################################################
Data Augmentation
#################################################################
"""
train_DA = ImageDataGenerator(  
    horizontal_flip=True,
    vertical_flip=True)

validation_DA = ImageDataGenerator(    
    horizontal_flip=True,
    vertical_flip=True)

train_generator = train_DA.flow_from_directory(  
    train_path,
    target_size=(width_shape, height_shape),
    #save_to_dir=r'/DA',
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical')

validation_generator = validation_DA.flow_from_directory(  
    validation_path,
    target_size=(width_shape, height_shape),
    #save_to_dir=r'/DA',
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical')


"""
#################################################################
Datos de entrenamiento
#################################################################
"""

#Cantidad total de datos para entrrenamiento y validación
n_samples_train= 58166
n_samples_validation = 14542

#Validación y corrección imágenes que ingresan al entrenamiento (Dimensión, canales)
image_input = Input(shape=(width_shape, height_shape, 3))

#Métricas de validación entrenamiento
metrics = [
    Precision(name="precision"),
    Recall(name="recall"),
    Accuracy(name="accuracy"),
    AUC(name="auc")
]

#Optimizer con taza de entrenamiento
Optimizer=Adam(learning_rate=1e-6)

"""
#################################################################
Modelo de entrenamiento con Transfer Learning
#################################################################
"""
Model_TL = MobileNetV2(input_tensor=image_input, include_top=False,weights='imagenet')
Model_TL.summary()

"""
#################################################################
Creación de capas de acuerdo a las necesidades del proyecto
#################################################################
"""
last_layer = Model_TL.layers[-1].output
x= Flatten(name='flattenbase')(last_layer)
x = Dense(128, activation='relu', name='fc1base')(x)
x=Dropout(0.35)(x)
x = Dense(128, activation='relu', name='fc2base')(x)
x=Dropout(0.45)(x)
out = Dense(num_classes, activation='softmax', name='outputbase')(x)

#Adhesión de capas al modelo
custom_model = Model(image_input, out)
custom_model.summary()

#Compilación del modelo
custom_model.compile(loss='binary_crossentropy',optimizer=Optimizer,metrics=metrics)

#Creación tabla de excel para almacenar los datos de evaluación generados por epoca de entrenamiento
data_metrics_train= CSVLogger('51MobileNetV2_DS_DosClases.csv', separator=',', append=True)

#Entrenamiento del modelo
model_history = custom_model.fit_generator(  
    train_generator,
    callbacks=[data_metrics_train],
    epochs=epochs,
    validation_data=validation_generator,
    steps_per_epoch=n_samples_train//batch_size,
    validation_steps=n_samples_validation//batch_size)

custom_model.save("E:/ParkingManagerCNN/51MobileNetV2_DS_DosClases.csv")
