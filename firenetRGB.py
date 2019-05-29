from tflearn.metrics import *
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression
import random as rand

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time


def firenet(x, y, training, x_train, y_train, x_test, y_test):

    # Construcción de la red neuronal. Cada linea supone una capa diferente, y cada capa se compone de un numero determinado de neuronas.
    # {Convolution layer}:
    #    Son las más usuales en las redes neuronales ya que son las capas que simulan el funcionamiento de las neuronas.
    #
    # {Max pool layer}:
    #   Se encargan de reducir las dimensiones de los datos que le entran a través de la combinación de salidas de clusters de neuronas.
    #
    # {Local response normalization layer}: 
    #   Se encargan de normalizar los resultados generados por neuronas con activación ReLU.
    #
    # {Fully connected layer}:
    #    Estas capas se encargan de conectar cada neurona de una capa con cada neurona de la siguiente capa.
    #
    # {Dropout layer}:
    #   Se encargan de evitar el overfitting de una red neuronal, ya que se seleccionan unas neuronas de forma aleatoria y se ignoran mientras se está entrenando.
    #

    network = tflearn.input_data(shape=[None, y, x, 3], dtype=tf.float32, name='input')
    network = conv_2d(network, 64, 5, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 128, 4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 1, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')


    # En caso de querer entrenar la red, con el bool 'training' en true se ejecutan distintas ordenes:
    #   -Se aplica la regresion donde se define el tipo de optimizador, la funcionn de loss que se utilizara, el learning rate
    #    y el nombre de los resultados del placeholder.
    #
    #   -Se genera la Deep Neural Network para ejecutar, es decir, el modelo.
    #
    #   -Se entrena definiendo el input y los targets del train, indicando un numero de epcochs y por ultimo el conjunto de test.
    #
    # En caso de que el booleano 'training' sea negativo, simplemente se genera el modelo de la red con la función DNN.
    #

    if(training):
        network = regression(network, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.0001, name='targets')
        model = tflearn.DNN(network, tensorboard_dir='log', tensorboard_verbose=0)
        #model.load("model/fire/firemodel.tfl", weights_only=True)      #Para seguir entrenando los mismos weights
        model.fit({'input': x_train}, {'targets': y_train}, n_epoch=10, validation_set=({'input': x_test}, {'targets': y_test}), snapshot_step=500, show_metric=True, run_id='firenet-model')
    else:
        model = tflearn.DNN(network, checkpoint_path='firenet', max_checkpoints=1, tensorboard_verbose=2)

    return model
    

if __name__ == '__main__':

    # El parametro booleano 'train' se utilizara para poder variar entre la ejecución del codigo para entrenar o para predecir.
    # A continuacion se especificara el tamaño de las imagenes, aunque es recomendable no modificarlo.

    train = False
    img_sz = 160 #224
    img_sz2 = 160

    # En caso de que nos dispongamos a entrenar:
    if train:
        # Cargamos los archivos .npy que contienen los datos de las imagenes con sus respectivos labels, tanto train como test.
        train = np.load('DataSet/train.npy')
        test = np.load('DataSet/test.npy')

        # Por si hay que tratar el dataset, para que tengan el mismo tamaño el train que el test
        # Cortamos todos los datos de más que tenga el conjunto mas grande
        trainl = len(train)
        testl = len(test)

        if (trainl < testl):
            minimo = trainl
        else:
            minimo = testl

        train = train[:-minimo]
        test = test[-minimo:]
        
        # Mezclamos los datos para que no esten seguidos los labels de fire y normal.
        rand.shuffle(train)
        rand.shuffle(test)

        # Preparamos las variables x (las imagenes) e y (los labels de las imagenes) para enviarlos a la red neuronal.
        # Lo haremos para el conjunto de train y para el de test.
        x_train = np.array([i['images'] for i in train]).reshape(-1, img_sz, img_sz2, 3)
        y_train = [i['labels'] for i in train]
        x_test = np.array([i['images'] for i in test]).reshape(-1, img_sz, img_sz2, 3)
        y_test = [i['labels'] for i in test]

        # Llamamos a la función que genera el modelo y guardamos en un archivo .tfl el contenido de los pesos qu se han generado.
        model = firenet(img_sz, img_sz2, True, x_train, y_train, x_test, y_test)
        model.save("model/fire/firemodel.tfl")

    else:
    # En caso de querer obtener clasificaciones a partir de una imagen

        # .............Pruebas de imagenes para clasificarlas...........
        #
        #img = cv2.imread('Fire-Detection-Image-Dataset-master/Fire images/burn-wall-street.jpg')
        #img = cv2.imread('Fire-Detection-Image-Dataset-master/normal/normal-18.jpg')
        #img = cv2.imread('Fire-Detection-Image-Dataset-master/fire/fire-16.jpg')
        img = cv2.imread('fire-45.jpg')
        #
        # ..............................................................
        
        # Se cambia el tamaño de las imagenes al que necesita la entrada de la red neuronal
        im = cv2.resize(img, (img_sz,img_sz2), interpolation=cv2.INTER_CUBIC)

        # Como que la red neuronal esta preparada para obtener una shape de (?, size_X, size_Y, 3),
        # donde '?' es el numero de la imagen dentro del dataset que se esta utilizando, hay que generar 
        # una matriz con esa dimensión extra.
        image = np.ones((1,img_sz2,img_sz,3), dtype=np.uint8)*255
      
        # Ahora copiamos la imagen con la shape correspondiente a la nueva matriz con el numero de dimensiones correcto.
        image[0,:,:,:]=im[:,:,:]


        # Construimos el modelo sin pasarle valores de x e, ni para train ni para test.
        # Solo le pasamos el tamaño de las imagenes y el booleano de train en 'false'.
        model = firenet(img_sz, img_sz2, False, 0, 0, 0, 0)
        print("Red construida")

        model.load("../NeuralNet/model/fire/firemodel.tfl", weights_only=True)  # Modelo propio de pesos para RPi
        #model.load("model/fire/firemodel.tfl", weights_only=True)  # Modelo propio de pesos
        #model.load("model/fire/firenet", weights_only=True)         # Modelo preentrenado de pesos
        print("Pesos cargados")
            
        # Se muestra la imagen que se ha obtenido para pasar por la NN
        cv2.imshow('image', image[0,:,:,:])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Para calcular el tiempo que transcurre en la prediccion de la NN
        start_time = time.time()
        output = model.predict(image)[0]
        print(output)

        # El que tenga el resultado más proximo a 1 será la clasificacion
        if np.argmax(output) == 1: 
            fire=False
        else:
            fire=True

        # Imprime el tiempo transcurrido para la prediccion
        print("--- %s seconds ---" % (time.time() - start_time)) 

        # Imprime el resultado de la prediccion
        print('Fire: ', fire)
