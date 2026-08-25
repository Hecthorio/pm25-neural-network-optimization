# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:32:26 2024

@author: hecto
"""

#script para obtener información de la API del INEGI

#importamos librerias
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importamos estas librerias para evaluar el area
from shapely.geometry import Polygon
import geopandas as gpd

'''
Esta url sirve para consultar la información de la población por MANZANAS (mza),
a continuación, así aparece en la pagina del INEGI (https://www.inegi.org.mx/servicios/catalogoUnico.html)
Consulta de Manzanas
Servicios con información vectorial en formato geojson

Los últimos numeros de la url simbolizan:
Clave de  la entidad (cve_ent), también se le llama clave del AGEE: 01 para Aguascalientes
Clave del municipio (cve_mun), también se le llama clave del AGEM: 001 para el municipio (ciudad) de Aguascalientes
Clave de localidad (cve_loc): 0001 para la localidad de Aguascalientes
La U define si es entorno rural (R) o urbano (U)

Para sacar más información sobre estos valores se puede consultar en:
https://www.inegi.org.mx/app/ageeml/
'''

#definimos la url de la api que vamos a utilizar con base en los parametros antes definidos
url = "https://gaia.inegi.org.mx/wscatgeo/v2/geo/mza/01/001/0001"

response = requests.get(url)

data = response.json()

df = pd.json_normalize(data['features'])

#eliminamos los valores nulos
print(f'Valores nulos: {df['properties.pob_total'].isna().sum()}')
df = df.dropna(subset=['properties.pob_total'])
df = df.reset_index(drop=True)

#generamos un arreglo vacio para ir guardando en este las áreas en km2
area = np.zeros(len(df))
lon = np.zeros(len(df))
lat = np.zeros(len(df))

#aqui generamos un ciclo for que vaya por cada una de las manzanas de la ciudad/municipio
for i in range(len(df)):
    #creamos los poligonos usando las coordenadas
    poligono = Polygon(df['geometry.coordinates'][i][0][0])
    
    #print("Área en grados cuadrados (WGS84):", poligono.area)
    
    # Crear un GeoDataFrame con las coordenadas
    #gdf = gpd.GeoDataFrame({'geometry': [poligono]}, crs="EPSG:4326")  # EPSG:4326 es WGS84
    gdf = gpd.GeoDataFrame({'geometry': [poligono]}, crs="EPSG:6365")
    
    # Convertir las coordenadas de lat/lon a UTM (proyección adecuada para áreas)
    gdf_utm = gdf.to_crs(epsg=3395)  # EPSG:3395 es una proyección mundial (como metros)
    
    # Calcular el área en metros cuadrados
    area_metros_cuadrados = gdf_utm.area[0]
    #area_metros_cuadrados = gdf.geometry.area
    #print("Área en metros cuadrados:", area_metros_cuadrados)
    
    #area_km2 = area_metros_cuadrados / 1e6
    area_km2 = area_metros_cuadrados
    #print("Área en kilómetros cuadrados:", area_km2)
    print("Área en metros cuadrados:", area_km2)
    
    #vamos guardando cada una de las areas
    area[i] = area_km2
    
    #guardamos cada uno de los valores promedio de lat y lon
    lon[i] = np.array(df['geometry.coordinates'][i][0][0])[:,0].mean()
    lat[i] = np.array(df['geometry.coordinates'][i][0][0])[:,1].mean()
    
#evaluamos el área total
#area_tot = sum(area)

#ponderación de cada una de las áreas
#area_pon = area/area_tot

#evaluar el promedio de densidad poblacional
#evaluamos el num de habitantes por km2
prom_den_po = df['properties.pob_total'].astype(float)/area
'''
#los nan los hacemos ceros
prom_den_po = prom_den_po.fillna(0)
#eliminamos el dato atipico y los ceros
prom_den_po = prom_den_po.drop(df['properties.pob_total'].astype(float).idxmax())
#prom_den_po = prom_den_po.drop(df['properties.pob_total'].astype(float).idxmax())
#evaluamos el promedio
prom_den_po = np.mean(prom_den_po)

#imprimimos el núm de habitantes, area superficial y relación
print(f'{sum(df['properties.pob_total'].astype(float).fillna(0)):.0f} habitantes')
print(f'{sum(area):.2f} área superficial, m2')
print(f'{prom_den_po:.4f} habitantes/m2')
#print(f'{sum(area):.2f} área superficial, km2')
#print(f'{prom_den_po:.4f} habitantes/km2')
'''

#graficamos
plt.hist(df['properties.pob_total'].astype(float)/area_km2, bins = 100)

#antes de hacer el grafico de burbujas, reacomodamos de menor a mayor todos los valores 
#de los vectores tomando como referencia el prom_den_po
indices = np.array(prom_den_po).argsort()
prom_den_po = prom_den_po[indices]
lat = lat[indices]
lon = lon[indices]


#importamos las libreria para generar mapabase
import contextily as ctx
from contextily import Place
import contextily as ctx
from scipy.ndimage import gaussian_filter

plt.close('all')
fig, ax = plt.subplots(figsize=(7,7))

ax.set_xlim([-102.37568491447393, -102.20035522245273])
#ax.set_ylim([21.788600065285202, 22.025731345356583])
ax.set_ylim([21.788600065285202, 21.97505731345356583])

'''
sources:
ctx.providers.OpenStreetMap.DE
ctx.providers.Esri.WorldStreetMap
ctx.providers.Esri.WorldImagery
ctx.providers.OpenTopoMap
'''

#graficamos el mapa
ctx.add_basemap(ax, zoom = 12, crs = "EPSG:4326", attribution=False, source = ctx.providers.OpenTopoMap)

#grafica de burbujas
#plt.scatter(lon, lat, s=prom_den_po*1000, c=prom_den_po, alpha=0.8, cmap='turbo', label="Densidad poblacional")  # Marcadores
plt.scatter(lon, lat, c=prom_den_po, alpha=0.75, cmap='turbo', label="Densidad poblacional")  # Marcadores
plt.colorbar().set_label('Habitantes/m$^2$')
plt.xlabel('Longitud')
plt.xticks(rotation = 45)
plt.ylabel('Latitud')
plt.legend(edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 0.1))
plt.tight_layout()

#aplicamos IDW para determinar la densidad de población en m2 en los lugares donde
#se localizan las estaciones de monitoreo
#pero 1ero leemos la base de datos para quedarnos solamente con las lat y lon que
#de las cuales nos interasa evaluar en esos puntos
#definimos la ruta
ruta = 'C:/Users/hecto/OneDrive/Documentos/ITA/3er Congreso internacional multidisciplinario de la divulgación científica/'

#leemos la base de datos
df_bd = pd.read_csv(ruta + 'base_datos.csv')

#separamos los pares ordenados de lat y lon unicos
lat_lon = np.array(df_bd.drop_duplicates(subset=['latitud','longitud'])[['latitud','longitud']])

#definimos la función IDW
def idw_interpolation(x, y, z, xi, yi, power=2):
    """
    Interpolación IDW (Inverse Distance Weighting)
    
    :param x: Array de coordenadas X de los puntos conocidos
    :param y: Array de coordenadas Y de los puntos conocidos
    :param z: Array de valores (mediciones) en los puntos conocidos
    :param xi: Array de coordenadas X de los puntos a evaluar
    :param yi: Array de coordenadas Y de los puntos a evaluar
    :param power: Exponente que controla el peso (por defecto es 2)
    
    :return: Un array con los valores interpolados en los puntos (xi, yi)
    """
    # Inicializar un array para los resultados de la interpolación
    interpolated_values = np.zeros(len(xi))
    
    # Para cada punto en (xi, yi), calcular su valor interpolado
    for i in range(len(xi)):
        # Calcular distancias de cada punto conocido al punto (xi[i], yi[i])
        distances = np.sqrt((x - xi[i])**2 + (y - yi[i])**2)
        
        # Evitar división por cero
        distances[distances == 0] = np.finfo(float).eps
        
        # Cálculo de los pesos usando la distancia y el exponente
        weights = 1 / (distances ** power)
        
        # Realizar la interpolación ponderada
        weighted_sum = np.sum(weights * z)
        sum_weights = np.sum(weights)
        
        # El valor interpolado es la suma ponderada dividida por la suma de los pesos
        interpolated_values[i] = weighted_sum / sum_weights
    
    return interpolated_values

#mandamos llamar la función
den_po_IDW = idw_interpolation(lon, lat, prom_den_po, lat_lon[:,1], lat_lon[:,0])

#hacemos cero las últimas tres
den_po_IDW[-3:] = 0 

#agregamos los ptos de las estaciones de monitoreo
plt.plot(lat_lon[:4,1],lat_lon[:4,0],'*', ms = 18, markeredgecolor='black', markerfacecolor='white', label = 'Estaciones monitoreo')
plt.plot(lat_lon[4:,1],lat_lon[4:,0],'s', ms = 14, markeredgecolor='black', markerfacecolor='green', label = 'Fronteras')
plt.legend()

#agregamos la columna de densidad poblacional al df_bd
df_bd = df_bd.assign(den_po = 0)

#agregamos el dato que le correponde en función de si cumple con que sea la misma lat y lon
for i in range(len(den_po_IDW)):
    indices = df_bd[(df_bd['latitud'] == lat_lon[i,0]) & (df_bd['longitud'] == lat_lon[i,1])].index
    df_bd.loc[indices, 'den_po'] = den_po_IDW[i]

#guardamos la modificación de la base de datos modificada con la densidad de población
df_bd.to_csv(ruta + 'base_datos_mod.csv', index = False)
