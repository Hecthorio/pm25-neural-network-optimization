# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 18:57:17 2026

@author: hecto
"""

import requests
import geopandas as gpd
from shapely.geometry import Polygon, Point

def cargar_manzanas_inegi(
    cve_ent="01",
    cve_mun="001",
    cve_loc="0001",
    epsg_proy="EPSG:32613"
):

    url = f"https://gaia.inegi.org.mx/wscatgeo/v2/geo/mza/{cve_ent}/{cve_mun}/{cve_loc}"
    data = requests.get(url).json()

    geometries = []
    poblacion = []

    for i, feat in enumerate(data["features"]):
        props = feat.get("properties", {})

        pob = (
            props.get("POBTOT") or
            props.get("pobtot") or
            props.get("pob_total")
        )

        if pob is None:
            continue

        coords = feat["geometry"]["coordinates"][0][0]
        geometries.append(Polygon(coords))
        poblacion.append(float(pob))

    gdf = gpd.GeoDataFrame(
        {
            "id_mza": range(len(geometries)),  # 🔹 ID estable
            "pob_total": poblacion
        },
        geometry=geometries,
        crs="EPSG:4326"
    )

    # 🔹 Proyectar una sola vez
    gdf = gdf.to_crs(epsg_proy)

    # 🔹 Calcular área original UNA vez
    gdf["area_mza"] = gdf.area

    return gdf


def densidad_poblacion(
    lat,
    lon,
    radio_m,
    manzanas_gdf
):
    """
    Calcula densidad poblacional (hab/km²) dentro de un radio dado
    Versión corregida (sin desalineación de índices)
    """

    #Punto en misma proyección
    punto = gpd.GeoSeries(
        [Point(lon, lat)],
        crs="EPSG:4326"
    ).to_crs(manzanas_gdf.crs)

    buffer_geom = punto.buffer(radio_m).iloc[0]
    area_km2 = buffer_geom.area / 1e6

    buffer_gdf = gpd.GeoDataFrame(
        geometry=[buffer_geom],
        crs=manzanas_gdf.crs
    )

    #Intersección
    inter = gpd.overlay(
        manzanas_gdf,
        buffer_gdf,
        how="intersection"
    )

    if inter.empty:
        return 0.0

    #Área intersecada
    inter["area_inter"] = inter.area

    #Fracción correcta (ya tiene area_mza original heredada)
    inter["frac_area"] = inter["area_inter"] / inter["area_mza"]

    #Población ponderada
    inter["pob_aportada"] = inter["pob_total"] * inter["frac_area"]

    densidad = inter["pob_aportada"].sum() / area_km2

    return densidad

#si corremos directamente la función
if __name__ == '__main__':    
    # 1. Cargar manzanas una sola vez
    manzanas_gdf = cargar_manzanas_inegi()
    
    # 2. Usar la función de densidad
    densidad = densidad_poblacion(
        lat=21.93136199221002,
        lon=-102.26858206527828,
        radio_m=1000,
        manzanas_gdf=manzanas_gdf
    )
    
    print(f"Densidad poblacional: {densidad:.0f} hab/km²")
