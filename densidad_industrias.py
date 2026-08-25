# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 11:19:05 2026

@author: hecto
"""

import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
from osmnx._errors import InsufficientResponseError

def fuentes_industriales_osm(
    lat,
    lon,
    radio_m=1000
):
    """
    Cuenta fuentes industriales dentro de un radio (m)
    usando OpenStreetMap
    """

    # tags industriales relevantes
    tags = {
    "landuse": ["industrial"], #checar si vamos dejar comercios tambien 🤔
    "building": ["industrial", "warehouse", "factory", "transportation"],
    "industrial": True,
    "man_made": "works"
            }

    try:
        gdf = ox.features.features_from_point(
            (lat, lon),
            tags=tags,
            dist=radio_m
        )
    except InsufficientResponseError:
        # No hay industrias en el radio
        return 0

    if gdf.empty:
        return 0

    # reproyectar
    gdf = gdf.to_crs(epsg=32613)
    
    #imprimimos lo que hay de cada uno
    for i in gdf:
        print(f'{i} = {gdf[i].value_counts()}')

    # buffer
    punto = gpd.GeoDataFrame(
        geometry=[Point(lon, lat)],
        crs="EPSG:4326"
    ).to_crs(epsg=32613)

    buffer = punto.buffer(radio_m).iloc[0]

    # recorte
    gdf_clip = gpd.clip(gdf, buffer)

    # conteo de fuentes
    return len(gdf_clip)/buffer.area * 1e6


if __name__ == '__main__':
    
    N_ind = fuentes_industriales_osm(
        lat=21.903366164779523,   
        lon= -102.27608917314508,
        radio_m=1000
    )
    
    print(f"Densidad de Fuentes industriales: {N_ind}/km2")
