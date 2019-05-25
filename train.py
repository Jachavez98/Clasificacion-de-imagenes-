# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
#Cierra keras antes de iniciar
K.clear_session()
#Direcciones de las imagenes para entrenamiento
data_entrenamiento = 'C:/Users/JOHAN/Downloads/AMP-Tech-master/AMP-Tech-master/CNN desde cero/data/entrenamiento'
data_validacion = 'C:/Users/JOHAN/Downloads/AMP-Tech-master/AMP-Tech-master/CNN desde cero/data/validacion'
"""
Parametros para la CNN
"""
epocas=20 #Numero de iteraciones
longitud, altura = 100, 100 #Tamaño de procesamiento de imagenes
batch_size = 32 #Numero de imagenes por cada numero de pasos
pasos = 1000 #Numero de veces que se procesa la informacion
validation_steps = 300 #Al final de cada epoca corre validacion
filtrosConv1 = 32 #Filtro de convolucion 1
filtrosConv2 = 64 #Filtro de convolucion 2
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2) #Tamaño de filtro de maxpooling
clases = 3 #Se declaran 3 clases para gato gorila y perro
lr = 0.0005 # Learning rate, tamaño de ajustes para solucion optima
##Preparamos nuestras imagenes para procesarlas
entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255, #Reescalado de imagenes
    shear_range=0.3, #
    zoom_range=0.3,  #Zoom a imagenes
    horizontal_flip=True) #Toma una imagen y la invierte 

test_datagen = ImageDataGenerator(rescale=1. / 255) #Imagenes solo reescaladas para validacion

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento, #Direccion de las imagenes
    target_size=(altura, longitud), 
    batch_size=batch_size,
    class_mode='categorical') #Modo de clasificacion categorica

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

cnn = Sequential() #Red neuronal sequencial
#Primera capa de convolucion
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
#Capa de Maxpooling
cnn.add(MaxPooling2D(pool_size=tamano_pool))
#Siguiente capa convolucional
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
#Capa de Maxpooling
cnn.add(MaxPooling2D(pool_size=tamano_pool))
#Hacer la imagen plana para la red neuronal
cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
#Activar 50 por ciento de neuronas aleatorias para crear caminos de solucion diferentes 
cnn.add(Dropout(0.5))
#Agregar capa densa con activacion softmax
cnn.add(Dense(clases, activation='softmax'))

cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])
cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps)

target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')