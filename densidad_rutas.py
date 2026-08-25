# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 20:58:08 2026

@author: hecto
"""

import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

def longitud_calles(
    lat,
    lon,
    radio_m=500,
    tipo_vial=None
):
    """
    Calcula longitud total de vialidades dentro de un radio (m)
    
    tipo_vial:
      None            → todas
      motorway
      motorway_link
      trunk
      trunk_link
      primary
      primary_link
      secondary
      secondary_link
      tertiary
      tertiary_link
      unclassified
      residential
      living_street
    """
    
    # 1. Descargar red vial desde OSM
    G = ox.graph_from_point(
        (lat, lon),
        dist=radio_m,
        network_type="drive"
    )

    # 2. Convertir a GeoDataFrame (edges)
    edges = ox.graph_to_gdfs(G, nodes=False)

    # 3. Filtro por tipo de vialidad (opcional)
    if tipo_vial is not None:
        edges = edges[edges["highway"].apply(
            lambda x: tipo_vial in x if isinstance(x, list) else x == tipo_vial
        )]

    # 4. Proyección métrica (Aguascalientes)
    edges = edges.to_crs(epsg=32613)

    # 5. Buffer
    punto = gpd.GeoDataFrame(
        geometry=[Point(lon, lat)],
        crs="EPSG:4326"
    ).to_crs(epsg=32613)

    buffer = punto.buffer(radio_m).iloc[0]

    # 6. Recorte
    edges_clip = gpd.clip(edges, buffer)

    # 7. Longitud total (km)
    longitud_km = edges_clip.length.sum() / 1e3

    return longitud_km/buffer.area * 1e6

#esto es si corremos el código normal
if __name__ == '__main__':

    L_total = longitud_calles(
        lat=21.93136199221002,
        lon=-102.26858206527828,
        radio_m=500
    )
    
    L_primarias = longitud_calles(
        lat=21.93136199221002,
        lon=-102.26858206527828,
        radio_m=500,
        tipo_vial="primary"
    )
    
    L_sin_cla = longitud_calles(
        lat=21.93136199221002,
        lon=-102.26858206527828,
        radio_m=500,
        tipo_vial="unclassified"
    )
    
    print(f"Densidad de Longitud total de calles: {L_total:.2f} km/km2")
    print(f"Densidad de Vías primarias: {L_primarias:.2f} km/km2")
    print(f"Densidad de Vías primarias: {L_sin_cla:.2f} km/km2")
