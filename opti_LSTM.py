# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 10:28:05 2025

@author: hecto
"""

'''
Codigo para evaluar diferentes configuraciones de RN (FNN, LSTM, GRU y CNN) 
usando la libreria optuna para estimar la concentración de PM2.5 en la 
ciudad de Aguascalientes
'''

#importamos las librerias
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
#from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import keras.optimizers
from keras.callbacks import EarlyStopping
import optuna

#definimos la ruta donde esta la base de datos
ruta = 'C:/Users/hecto/OneDrive/Documentos/ITA/3er Congreso internacional multidisciplinario de la divulgación científica/'

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

#generamos los datos de entrenamiento
#X, y = preparar_datos(X,lat_lon, 0, time_step)


#hacemos reshape de y
#y = y.reshape(-1, 1)

# #generamos el modelo
# model = Sequential()
# model.add(GRU(200, activation='leaky_relu', input_shape=(3, X.shape[2])))
# model.add(Dense(1))

# model.compile(optimizer='adam', loss='mean_squared_error')

# # Entrenar el modelo
# perdida = model.fit(X, y, epochs=30, validation_split = 0.2, shuffle = True, batch_size = 200)

# #graficamos la función de perdida
# plt.plot(perdida.history['loss'], label = 'Entrenamiento')
# plt.plot(perdida.history['val_loss'], label = 'Validación')
# plt.legend()
# plt.xlabel('Épocas')
# plt.ylabel('MSE')
    
def objetive(trial):
    
    #definimos los estimadores en la 1ra parte
    num_units = trial.suggest_int('num_units', 50, 300)  # Unidades de la capa GRU
    #num_neurons = trial.suggest_int("num_neurons", 50, 300, log=True)
    #num_neurons = trial.suggest_int("num_neurons", 50, 300)
    activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh", "leaky_relu"])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    #learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32,2000)
    loss = trial.suggest_categorical("loss", ["mse", "mae", "mape"])
    
    #otro parametro a modificar sera el time_step que modifica la estrucutra de los datos de como
    #son alimentados al modelo
    time_step = trial.suggest_int("time_step", 1,10)
    X_op, y_op = preparar_datos(X,lat_lon, 0, time_step)
    
    #construimos el modelo
    red = Sequential()
    # for i in range(num_layers):
    #     if i == 0:
    #         red.add(Dense(num_neurons, input_dim = X_train.shape[1], activation=activation))
    #     else:
    #         red.add(Dense(num_neurons, activation=activation))
    red.add(LSTM(num_units, activation = activation, input_shape = (time_step,X_op.shape[2])))
    #Capa de salida
    red.add(Dense(1, activation='linear'))
    #definimos el optmizador
    optimizer = getattr(keras.optimizers, optimizer_name)(learning_rate=learning_rate)
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
    perdida = red.fit(X_op, y_op, epochs = 100, validation_split = 0.2, shuffle = True, batch_size = batch_size, callbacks=[early_stopping])
    
    #despues del entrenamiento evaluamos con los datos de prueba y seleccionamos solamente la metrica
    #metrica = red.evaluate(X_test,y_test, verbose = 0)[1]
    metrica = perdida.history['val_mse'][-1]
    #definimos un tamaño de batch para evaluar en el modelo
    #tamano_batch = 1000
    
    #ecm = sum((red.predict(X_op, batch_size = tamano_batch) - y_op[:tamano_batch])**2)/tamano_batch
    
    if metrica > 1:
        metrica = np.nan
    
    return metrica

#ahora definimos el optimizador de hiperparametros con optuna
study = optuna.create_study(direction = 'minimize', sampler = optuna.samplers.TPESampler())
study.optimize(objetive, n_trials = 30)

#imprimimos en pantalla los mejores resultados
print(study.best_params)

#guardamos el df con los resultados
df_opti = study.trials_dataframe()

#dropeamos las columnas que no nos interesan del df_opti
df_opti.drop(columns = ['datetime_start', 'datetime_complete', 'duration', 'state'], inplace = True)

#damos de alta la ruta donde se guardaran las figuras
ruta_fig = 'C:/Users/hecto/OneDrive/Documentos/ITA/3er Congreso internacional multidisciplinario de la divulgación científica/'

#mostramos las graficas (en comentrarios algunas formas sobre como generar la figura)
# optuna.visualization.matplotlib.plot_optimization_history(study)
#optuna.visualization.plot_optimization_history(study).show(renderer="browser")


fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
plt.savefig(ruta_fig + "optimization_history_lstm.pdf",  bbox_inches='tight')


#optuna.visualization.plot_slice(study, params = ['num_layers', 'num_neurons', 'activation', 'optimizer', 'learning_rate',
#                                                 'batch_size', 'loss']).show(renderer="browser")


#optuna.visualization.plot_param_importances(study).show(renderer = 'browser')
fig1 = optuna.visualization.matplotlib.plot_param_importances(study)
plt.savefig(ruta_fig + "hip_param_impot_lst.pdf",  bbox_inches='tight')


# optuna.visualization.matplotlib.plot_parallel_coordinate(study)
#optuna.visualization.plot_parallel_coordinate(study).show(renderer="browser")
plt.rcParams["axes.grid"] = False
plt.rcParams["figure.facecolor"] = 'white'
plt.rcParams["font.size"] = 16
fig1 = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
fig = plt.gcf()  # Obtener la figura actual
fig.set_size_inches(15, 8)  # Cambiar el tamaño de la figura
#optuna.visualization.plot_parallel_coordinate(study)
plt.savefig(ruta_fig + "parallel_coordinate_lstm.pdf", bbox_inches='tight')
plt.rcdefaults()

#redondemos los valores antes de guardarlos
df_opti.value = df_opti.value.round(3)
df_opti.params_learning_rate = df_opti.params_learning_rate.round(3)

#guardamos el df de la tabla para abrirlo en latex
with open(ruta_fig + 'df_opti_lstm.tex', 'w') as f:
    f.write(df_opti.to_latex(index=False))

#de esta otra manera puede llegar a funcionar
#fig = optuna.visualization.plot_contour(study)
#fig.show()