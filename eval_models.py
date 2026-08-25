# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 13:40:11 2025

@author: hecto
"""

'''
Entrenamos los modelos con los mejores hiperarametros obtenidos de optuna
'''

#importamos las librerias
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, SimpleRNN, Conv1D, MaxPooling1D, Flatten, Dense
#from tensorflow.keras.optimizers import Adam
#import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import keras.optimizers
from keras.callbacks import EarlyStopping
#import optuna

#definimos la ruta de donde se van a leer los archivos
ruta = 'C:/Users/hecto/OneDrive/Documentos/ITA/3er Congreso internacional multidisciplinario de la divulgación científica/'

#definimos el numero de veces que se va a entrenar cada modelo
repes = range(6)

#cargamos la base de datos
df = pd.read_csv(ruta + 'base_datos_mod.csv')

#definimos el salto en el tiempo
#time_step = 3

#serparamos los datos
lat_lon = np.array(df.drop_duplicates(subset=['latitud','longitud'])[['latitud','longitud']])

#aplicamos el separador de datos
X = np.array(df[['PM25', 'TMP', 'latitud', 'longitud', 
              'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
              'sin_doy', 'cos_doy', 'WSy', 'WSx', 'den_po']])

#escalamos los datos de que se usan para el escalamiento
#generamos las instancias
escalador_x = MinMaxScaler(feature_range=(-1,1))
escalador_y = MinMaxScaler(feature_range=(-1,1))
escalador_lat_lon = MinMaxScaler(feature_range=(-1,1))

#le pasamos los datos a las instancias (maximos y minimos)
escalador_x.fit(X)
escalador_y.fit(X[:,0].reshape(-1,1))
escalador_lat_lon.fit(lat_lon)

#ahora su escalamos nuestros datos
X = escalador_x.transform(X)
lat_lon = escalador_lat_lon.transform(lat_lon)

# 1. Filtrar las estaciones por coordenadas (latitud, longitud)
def filtrar_estaciones(X, lat, lon):
    # Filtrar las estaciones que están dentro del rango de latitud y longitud
    X_filtrado = X[(X[:,2] == lat) & (X[:,3] == lon)]
    return X_filtrado

# 2. Crear las secuencias manualmente
def crear_secuencias(X_df, target_column, time_step, forecast_horizon=1):
    """
    Crea las secuencias de tiempo de manera manual, con un time_step dado y una predicción hacia el futuro.
    """
    X, y = [], []
    
    #Ordenar el DataFrame por fecha para asegurar que los datos estén en orden temporal
    #df = df.sort_values('fecha')

    # Crear las secuencias de datos
    for i in range(time_step, len(X_df) - forecast_horizon):
        # Definir la ventana temporal
        X.append(X_df[i-time_step:i,:])  # Últimos `time_step` pasos
        y.append(X_df[i + forecast_horizon-1,target_column])  # Predecir `forecast_horizon` pasos hacia adelante
        
    return np.array(X), np.array(y)

# 3. Función para preparar los datos (filtrado y creación de secuencias)
def preparar_datos(df, lat_lon, target_column, time_step, forecast_horizon=1):
    
    #aplicamos un ciclo para ir filtrando por estacion de monitoreo
    for i in range(len(lat_lon)):
        # Filtrar el DataFrame por coordenadas
        df_filtrado = filtrar_estaciones(df, lat_lon[i,0], lat_lon[i,1])
        
        #para la 1ra vuelta
        if i == 0:
            #generamos la secuencia de esos datos
            X, y = crear_secuencias(df_filtrado, target_column, time_step)
        else:
            #generamos las secuencias de esos datos
            X_con, y_con = crear_secuencias(df_filtrado, target_column, time_step)
            #concatenamos con el original
            X = np.concatenate((X, X_con), axis = 0)
            y = np.concatenate((y, y_con), axis = 0)
    
    return X, y

#generamos un df vacio donde iremos guardando toda la info
df_mse = pd.DataFrame()

###############################################################################
#                                 GRU                                         #
###############################################################################
'''
En esta hubo varios modelos que llegaron practicamente al mismo valor asi que
se tomo como referencia el menor numero de time_step y unidades (neuronas),
A continuación de declara la lista de parametros
'''

neuronas = 231
activacion = 'tanh'
batches = 913
aprendizaje = 0.005
loss = 'mae'
optimizador = 'Adam'
time_step = 10
X_op, y_op = preparar_datos(X,lat_lon, 0, time_step)

#generamos una lista varia donde iremos guardando la info
metrica = []

#repetimos el entrenamiento n veces para obtener varias muestras de la mse
for i in repes:
    
    #construimos el modelo
    red = Sequential()
    # for i in range(num_layers):
    #     if i == 0:
    #         red.add(Dense(num_neurons, input_dim = X_train.shape[1], activation=activation))
    #     else:
    #         red.add(Dense(num_neurons, activation=activation))
    red.add(GRU(neuronas, activation = activacion, input_shape = (time_step,X_op.shape[2])))
    #Capa de salida
    red.add(Dense(1, activation='linear'))
    #definimos el optmizador
    optimizer = getattr(keras.optimizers, optimizador)(learning_rate = aprendizaje)
    #compilamos el modelo
    #definimos la función de perdida que sera utilizada
    if loss == 'mse':
        red.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mse'])
    if loss == 'mae':
        red.compile(optimizer=optimizer,
                    loss='mean_absolute_error',
                    metrics=['mse'])
    if loss == 'mape':
        red.compile(optimizer=optimizer,
                    loss='mean_absolute_percentage_error',
                    metrics=['mse'])
    
    #definimos el detenimiento temprano para que el modelo se detenga despues de que
    #no haya variación en la función de perdida
    early_stopping = EarlyStopping(monitor='loss', patience = 20)
    
    #entrenamiento del modelo
    perdida = red.fit(X_op, y_op, epochs = 100, validation_split = 0.2, shuffle = True, batch_size = batches, callbacks=[early_stopping])
    
    #despues del entrenamiento evaluamos con los datos de prueba y seleccionamos solamente la metrica
    #metrica = red.evaluate(X_test,y_test, verbose = 0)[1]
    metrica.append(perdida.history['val_mse'][-1])
    
#gurdamos la info en el df
df_mse = df_mse.assign(GRU = np.array(metrica))

###############################################################################
#                              LSTM                                           #
###############################################################################
'''
En esta hubo varios modelos que llegaron practicamente al mismo valor asi que
se tomo como referencia el menor numero de time_step y unidades (neuronas),
A continuación de declara la lista de parametros
'''

neuronas = 65
activacion = 'tanh'
batches = 721
aprendizaje = 0.009
loss = 'mse'
optimizador = 'Adam'
time_step = 8
X_op, y_op = preparar_datos(X,lat_lon, 0, time_step)

#generamos una lista varia donde iremos guardando la info
metrica = []

for i in repes:
    #construimos el modelo
    red = Sequential()
    # for i in range(num_layers):
    #     if i == 0:
    #         red.add(Dense(num_neurons, input_dim = X_train.shape[1], activation=activation))
    #     else:
    #         red.add(Dense(num_neurons, activation=activation))
    red.add(LSTM(neuronas, activation = activacion, input_shape = (time_step,X_op.shape[2])))
    #Capa de salida
    red.add(Dense(1, activation='linear'))
    #definimos el optmizador
    optimizer = getattr(keras.optimizers, optimizador)(learning_rate=aprendizaje)
    #compilamos el modelo
    #definimos la función de perdida que sera utilizada
    if loss == 'mse':
        red.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mse'])
    if loss == 'mae':
        red.compile(optimizer=optimizer,
                    loss='mean_absolute_error',
                    metrics=['mse'])
    if loss == 'mape':
        red.compile(optimizer=optimizer,
                    loss='mean_absolute_percentage_error',
                    metrics=['mse'])
    
    #definimos el detenimiento temprano para que el modelo se detenga despues de que
    #no haya variación en la función de perdida
    early_stopping = EarlyStopping(monitor='loss', patience = 20)
    
    #entrenamiento del modelo
    perdida = red.fit(X_op, y_op, epochs = 100, validation_split = 0.2, shuffle = True, batch_size = batches, callbacks=[early_stopping])
    
    #despues del entrenamiento evaluamos con los datos de prueba y seleccionamos solamente la metrica
    #metrica = red.evaluate(X_test,y_test, verbose = 0)[1]
    metrica.append(perdida.history['val_mse'][-1])

#gurdamos la info en el df
df_mse = df_mse.assign(LSTM = np.array(metrica))

###############################################################################
#                                  RNN                                        #
###############################################################################

'''
En esta hubo varios modelos que llegaron practicamente al mismo valor asi que
se tomo como referencia el menor numero de time_step y unidades (neuronas),
A continuación de declara la lista de parametros
'''

neuronas = 186
activacion = 'relu'
batches = 902
aprendizaje = 0.005
loss = 'mae'
optimizador = 'Adam'
time_step = 9
X_op, y_op = preparar_datos(X,lat_lon, 0, time_step)

#generamos una lista varia donde iremos guardando la info
metrica = []

for i in repes:
    red = Sequential()
    # for i in range(num_layers):
    #     if i == 0:
    #         red.add(Dense(num_neurons, input_dim = X_train.shape[1], activation=activation))
    #     else:
    #         red.add(Dense(num_neurons, activation=activation))
    red.add(SimpleRNN(neuronas, activation = activacion, input_shape = (time_step,X_op.shape[2])))
    #Capa de salida
    red.add(Dense(1, activation='linear'))
    #definimos el optmizador
    optimizer = getattr(keras.optimizers, optimizador)(learning_rate=aprendizaje)
    #compilamos el modelo
    #definimos la función de perdida que sera utilizada
    if loss == 'mse':
        red.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mse'])
    if loss == 'mae':
        red.compile(optimizer=optimizer,
                    loss='mean_absolute_error',
                    metrics=['mse'])
    if loss == 'mape':
        red.compile(optimizer=optimizer,
                    loss='mean_absolute_percentage_error',
                    metrics=['mse'])
    
    #definimos el detenimiento temprano para que el modelo se detenga despues de que
    #no haya variación en la función de perdida
    early_stopping = EarlyStopping(monitor='loss', patience = 20)
    
    #entrenamiento del modelo
    perdida = red.fit(X_op, y_op, epochs = 100, validation_split = 0.2, shuffle = True, batch_size = batches, callbacks=[early_stopping])
    
    #despues del entrenamiento evaluamos con los datos de prueba y seleccionamos solamente la metrica
    #metrica = red.evaluate(X_test,y_test, verbose = 0)[1]
    metrica.append(perdida.history['val_mse'][-1])

#gurdamos la info en el df
df_mse = df_mse.assign(RNN = np.array(metrica))

###############################################################################
#                                     CNN                                     #
###############################################################################

'''
En esta hubo varios modelos que llegaron practicamente al mismo valor asi que
se tomo como referencia el menor numero de time_step y unidades (neuronas),
A continuación de declara la lista de parametros
'''

neuronas = 233
filtros = 45
activacion = 'leaky_relu'
batches = 1674
aprendizaje = 0.004
loss = 'mse'
optimizador = 'RMSprop'
time_step = 10
X_op, y_op = preparar_datos(X,lat_lon, 0, time_step)

#generamos una lista varia donde iremos guardando la info
metrica = []

for i in repes:
    #construimos el modelo
    red = Sequential()
    red.add(Conv1D(filters = filtros, activation = activacion, kernel_size = 3, input_shape = (time_step,X_op.shape[2])))
    red.add(MaxPooling1D(pool_size = 2))
    red.add(Flatten()) #convierete todas las salidas del maxpooling en un solo vector de salida
    red.add(Dense(neuronas, activation = activacion))
    #Capa de salida
    red.add(Dense(1, activation='linear'))
    #definimos el optmizador
    optimizer = getattr(keras.optimizers, optimizador)(learning_rate=aprendizaje)
    #compilamos el modelo
    #definimos la función de perdida que sera utilizada
    if loss == 'mse':
        red.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mse'])
    if loss == 'mae':
        red.compile(optimizer=optimizer,
                    loss='mean_absolute_error',
                    metrics=['mse'])
    if loss == 'mape':
        red.compile(optimizer=optimizer,
                    loss='mean_absolute_percentage_error',
                    metrics=['mse'])
    
    #definimos el detenimiento temprano para que el modelo se detenga despues de que
    #no haya variación en la función de perdida
    early_stopping = EarlyStopping(monitor='loss', patience = 20)
    
    #entrenamiento del modelo
    perdida = red.fit(X_op, y_op, epochs = 100, validation_split = 0.2, shuffle = True, batch_size = batches, callbacks=[early_stopping])
    
    #despues del entrenamiento evaluamos con los datos de prueba y seleccionamos solamente la metrica
    #metrica = red.evaluate(X_test,y_test, verbose = 0)[1]
    metrica.append(perdida.history['val_loss'][-1])

#gurdamos la info en el df
df_mse = df_mse.assign(CNN = np.array(metrica))

###############################################################################
#                                  FNN                                        #
###############################################################################

'''
En esta hubo varios modelos que llegaron practicamente al mismo valor asi que
se tomo como referencia el menor numero de time_step y unidades (neuronas),
A continuación de declara la lista de parametros

Pero este modelo solo usa el estado previo como entrada por lo que las funciones
para generar la entrada del dataset del modelo se modifican
'''

# 1. Filtrar las estaciones por coordenadas (latitud, longitud)
def filtrar_estaciones(X, lat, lon):
    # Filtrar las estaciones que están dentro del rango de latitud y longitud
    X_filtrado = X[(X[:,2] == lat) & (X[:,3] == lon)]
    return X_filtrado

# 2. Crear las secuencias manualmente
def crear_secuencias(X_df):
    """
    Separamos la fila de la respuesta del modelo y eliminamos la última fila de las X
    """
    
    #salida del modelo
    y = X_df[1:,0]*1
    #entradas del modelo
    X = X_df*1
    X[0:-1,1:] = X_df[1:,1:]*1
    X = X[:-1,:]*1
    
    return np.array(X), np.array(y)

# 3. Función para preparar los datos (filtrado y creación de secuencias)
def preparar_datos(df, lat_lon):
    
    #aplicamos un ciclo para ir filtrando por estacion de monitoreo
    for i in range(len(lat_lon)):
        # Filtrar el DataFrame por coordenadas
        df_filtrado = filtrar_estaciones(df, lat_lon[i,0], lat_lon[i,1])
        
        #para la 1ra vuelta
        if i == 0:
            #generamos la secuencia de esos datos
            X, y = crear_secuencias(df_filtrado)
        else:
            #generamos las secuencias de esos datos
            X_con, y_con = crear_secuencias(df_filtrado)
            #concatenamos con el original
            X = np.concatenate((X, X_con), axis = 0)
            y = np.concatenate((y, y_con), axis = 0)
    
    return X, y

#definiomos los hiperparametros

#evaluamos nuestro dataset
X_op, y_op = preparar_datos(X,lat_lon)

num_layers = 2
neuronas = 103
activacion = 'leaky_relu'
batches = 339
aprendizaje = 0.005
loss = 'mae'
optimizador = 'Adam'

#generamos una lista varia donde iremos guardando la info
metrica = []

#generamos el ciclo
for j in repes:        
    #construimos el modelo
    red = Sequential()
    for i in range(num_layers):
        if i == 0:
            red.add(Dense(neuronas, input_dim = X_op.shape[1], activation=activacion))
        else:
            red.add(Dense(neuronas, activation=activacion))
    #Capa de salida
    red.add(Dense(1, activation='linear'))
    #definimos el optmizador
    optimizer = getattr(keras.optimizers, optimizador)(learning_rate=aprendizaje)
    #compilamos el modelo
    #definimos la función de perdida que sera utilizada
    if loss == 'mse':
        red.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mse'])
    if loss == 'mae':
        red.compile(optimizer=optimizer,
                    loss='mean_absolute_error',
                    metrics=['mse'])
    if loss == 'mape':
        red.compile(optimizer=optimizer,
                    loss='mean_absolute_percentage_error',
                    metrics=['mse'])
    
    #definimos el detenimiento temprano para que el modelo se detenga despues de que
    #no haya variación en la función de perdida
    early_stopping = EarlyStopping(monitor='loss', patience = 20)
    
    #entrenamiento del modelo
    perdida = red.fit(X_op, y_op, epochs = 100, validation_split = 0.2, shuffle = True, batch_size = batches, callbacks=[early_stopping])
    
    #despues del entrenamiento evaluamos con los datos de prueba y seleccionamos solamente la metrica
    #metrica = red.evaluate(X_test,y_test, verbose = 0)[1]
    metrica.append(perdida.history['val_mse'][-1])
    
#gurdamos la info en el df
df_mse = df_mse.assign(FNN = np.array(metrica))

df_mse.to_csv(ruta + 'resultados_mse_training.csv', index = False)