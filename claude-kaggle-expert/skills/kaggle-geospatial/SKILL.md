---
name: kaggle-geospatial
description: Expert en analyse géospatiale pour compétitions Kaggle. Utiliser quand l'utilisateur travaille avec des données géographiques, GPS, coordonnées, shapefiles, cartes, GeoPandas, Folium, ou des compétitions géospatiales.
argument-hint: <type d'analyse géospatiale ou données>
---

# Expert Analyse Géospatiale - Kaggle

Tu es un expert en analyse géospatiale. Tu maîtrises GeoPandas, Folium, Shapely, et les techniques de manipulation de données géographiques pour les compétitions Kaggle.

## Stack Technique Géospatial

```python
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
```

## 1. GeoDataFrame — Fondamentaux

### Chargement de Données Géospatiales

```python
# Lire un shapefile, GeoJSON, GeoPackage, KML
gdf = gpd.read_file("data/regions.shp")
gdf = gpd.read_file("data/zones.geojson")
gdf = gpd.read_file("data/map.gpkg")

# Inspecter
print(f"Type: {type(gdf)}")          # GeoDataFrame
print(f"CRS: {gdf.crs}")             # Système de coordonnées
print(f"Geometry types: {gdf.geom_type.unique()}")  # Point, LineString, Polygon
print(f"Colonnes: {gdf.columns.tolist()}")
gdf.head()
```

### Créer un GeoDataFrame depuis un DataFrame

```python
# Depuis des colonnes latitude/longitude
df = pd.read_csv("data/locations.csv")
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
    crs="EPSG:4326"  # WGS84 lat/lon
)

# Depuis des coordonnées WKT (Well-Known Text)
from shapely import wkt
df['geometry'] = df['wkt_column'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
```

### Types de Géométrie

```python
# Point : localisation unique (ex: magasin, capteur)
point = Point(2.3522, 48.8566)  # (longitude, latitude)

# LineString : ligne (ex: route, rivière, trajet)
line = LineString([(0, 0), (1, 1), (2, 0)])

# Polygon : surface (ex: quartier, parcelle, zone)
polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

# Attributs
point.x, point.y            # Coordonnées
line.length                  # Longueur
polygon.area                 # Surface
polygon.centroid             # Centre de gravité
polygon.bounds               # Bounding box (minx, miny, maxx, maxy)
polygon.exterior.coords[:]   # Liste des coordonnées
```

## 2. Systèmes de Coordonnées (CRS)

### EPSG Courants

| EPSG | Nom | Usage | Unités |
|------|-----|-------|--------|
| 4326 | WGS 84 | GPS, web mapping standard | Degrés (lat/lon) |
| 3857 | Web Mercator | Google Maps, OpenStreetMap | Mètres |
| 2154 | Lambert 93 | France métropolitaine | Mètres |
| 32630-32660 | UTM zones | Mesures locales précises | Mètres |

### Opérations CRS

```python
# Vérifier le CRS
print(gdf.crs)

# Définir le CRS (si non défini)
gdf = gdf.set_crs("EPSG:4326")

# Re-projeter (changer de CRS)
gdf_meters = gdf.to_crs(epsg=3857)      # Pour calculs de distance en mètres
gdf_france = gdf.to_crs(epsg=2154)      # Lambert 93 pour la France
gdf_wgs84 = gdf.to_crs(epsg=4326)       # Retour en lat/lon

# IMPORTANT : toujours projeter en mètres AVANT de calculer distances/surfaces
gdf_proj = gdf.to_crs(epsg=3857)
gdf_proj['area_km2'] = gdf_proj.geometry.area / 1e6  # m² → km²
```

## 3. Opérations Spatiales

### Jointure Spatiale (Spatial Join)

```python
# Points dans polygones : trouver dans quel polygone chaque point se trouve
points_in_zones = gpd.sjoin(
    points_gdf,       # GeoDataFrame de points
    zones_gdf,         # GeoDataFrame de polygones
    how='left',        # 'inner', 'left', 'right'
    predicate='within'  # 'intersects', 'within', 'contains'
)

# Résultat : chaque point a les attributs du polygone qui le contient
# index_right = index du polygone correspondant

# Compter les points par zone
points_per_zone = points_in_zones.groupby('zone_name').size().reset_index(name='count')
```

### Jointure par Attribut

```python
# Merge classique (comme Pandas)
gdf_enriched = gdf.merge(stats_df, on='region_name', how='left')
```

### Distances

```python
# IMPORTANT : projeter en mètres d'abord !
gdf_m = gdf.to_crs(epsg=3857)

# Distance d'un point à tous les autres
ref_point = gdf_m.iloc[0].geometry
gdf_m['distance_to_ref'] = gdf_m.geometry.distance(ref_point)

# Plus proche voisin
nearest_idx = gdf_m.geometry.distance(ref_point).idxmin()
nearest = gdf_m.iloc[nearest_idx]

# Matrice de distances
from scipy.spatial.distance import cdist
coords = np.column_stack([gdf_m.geometry.x, gdf_m.geometry.y])
dist_matrix = cdist(coords, coords)

# Distance Haversine (pour données GPS en degrés)
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """Distance en km entre deux points GPS."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))  # Rayon terrestre en km
```

### Buffers et Zones Tampons

```python
# Buffer autour de chaque géométrie (en unités du CRS)
gdf_m = gdf.to_crs(epsg=3857)  # En mètres
buffer_5km = gdf_m.geometry.buffer(5000)  # 5 km

# Union de tous les buffers
zone_couverte = buffer_5km.unary_union  # MultiPolygon unique

# Test de contenance
point_test = Point(x_test, y_test)
est_couvert = zone_couverte.contains(point_test)  # True/False

# Surface couverte
print(f"Surface couverte : {zone_couverte.area / 1e6:.1f} km²")
```

### Opérations Ensemblistes

```python
# Intersection de deux géométries
intersection = geom1.intersection(geom2)

# Union
union = geom1.union(geom2)

# Différence
difference = geom1.difference(geom2)

# Overlay entre GeoDataFrames
result = gpd.overlay(gdf1, gdf2, how='intersection')  # 'union', 'difference', 'symmetric_difference'
```

## 4. Geocodage

```python
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

geolocator = Nominatim(user_agent="kaggle_competition")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# Geocoder une adresse
location = geolocator.geocode("Tour Eiffel, Paris")
print(f"Lat: {location.latitude}, Lon: {location.longitude}")
print(f"Adresse: {location.address}")

# Geocodage en batch avec gestion d'erreurs
def safe_geocode(address):
    try:
        loc = geocode(address)
        if loc:
            return pd.Series({'lat': loc.latitude, 'lon': loc.longitude})
    except Exception:
        pass
    return pd.Series({'lat': np.nan, 'lon': np.nan})

df[['lat', 'lon']] = df['address'].apply(safe_geocode)

# Taux de succès
success_rate = df['lat'].notna().mean() * 100
print(f"Geocodage réussi : {success_rate:.1f}%")
```

## 5. Visualisation Statique (Matplotlib + GeoPandas)

```python
# Carte multi-couches
fig, ax = plt.subplots(figsize=(12, 10))

# Fond : polygones (régions, pays)
regions_gdf.plot(ax=ax, color='lightgrey', edgecolor='white', linewidth=0.5)

# Couche : lignes (routes, rivières)
routes_gdf.plot(ax=ax, color='steelblue', linewidth=1, alpha=0.7)

# Couche : points (villes, capteurs)
points_gdf.plot(ax=ax, color='red', markersize=10, alpha=0.8, label='Points')

# Choroplèthe statique (colorier par valeur)
regions_gdf.plot(column='population', cmap='YlOrRd', legend=True,
                 legend_kwds={'label': 'Population'}, ax=ax,
                 edgecolor='white', linewidth=0.5)

ax.set_title('Carte', fontsize=14)
ax.set_axis_off()
plt.tight_layout()
plt.show()
```

## 6. Cartes Interactives (Folium)

### Carte de Base

```python
# Créer la carte centrée sur un point
m = folium.Map(
    location=[48.8566, 2.3522],  # [lat, lon] Paris
    zoom_start=12,
    tiles='cartodbpositron'  # 'openstreetmap', 'cartodbdark_matter', 'stamentoner'
)
```

### Marqueurs

```python
# Marqueurs simples
for idx, row in gdf.iterrows():
    Marker(
        location=[row['latitude'], row['longitude']],
        popup=row['name'],
        tooltip=f"{row['name']}: {row['value']:.0f}"
    ).add_to(m)

# Marqueurs clusterisés (pour beaucoup de points)
mc = MarkerCluster()
for idx, row in gdf.iterrows():
    mc.add_child(Marker([row['lat'], row['lon']], tooltip=row['name']))
m.add_child(mc)
```

### Cercles (Bubble Map)

```python
# Cercles proportionnels à une valeur
for idx, row in gdf.iterrows():
    Circle(
        location=[row['lat'], row['lon']],
        radius=row['value'] * 100,  # Rayon en mètres
        color='blue' if row['category'] == 'A' else 'red',
        fill=True,
        fill_opacity=0.4,
        popup=f"{row['name']}: {row['value']}"
    ).add_to(m)
```

### Heatmap

```python
# Carte de chaleur
HeatMap(
    data=gdf[['latitude', 'longitude']].dropna().values.tolist(),
    radius=15,
    blur=10,
    max_zoom=13
).add_to(m)

# Heatmap pondérée
heat_data = [[row['lat'], row['lon'], row['weight']]
             for _, row in gdf.iterrows()]
HeatMap(heat_data, radius=20).add_to(m)
```

### Choroplèthe Interactif

```python
# Choroplèthe : colorier des zones par valeur
Choropleth(
    geo_data=zones_gdf.__geo_interface__,  # GeoJSON
    data=zones_gdf.set_index('zone_id')['value'],  # Série indexée
    key_on='feature.id',
    fill_color='YlOrRd',  # 'BuGn', 'YlGnBu', 'PuBuGn', 'RdYlGn'
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Valeur'
).add_to(m)
```

### GeoJSON Layer

```python
from folium import GeoJson

# Ajouter un GeoDataFrame comme couche
GeoJson(
    gdf.to_crs(epsg=4326),
    style_function=lambda x: {
        'fillColor': 'green',
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.3
    }
).add_to(m)
```

## 7. Feature Engineering Géospatial (pour compétitions)

```python
def create_geo_features(df, lat_col='latitude', lon_col='longitude'):
    """Créer des features géospatiales pour ML."""

    # Distance au centre de la ville
    city_center = (48.8566, 2.3522)  # ADAPTER
    df['dist_to_center'] = df.apply(
        lambda row: haversine(row[lon_col], row[lat_col],
                             city_center[1], city_center[0]), axis=1
    )

    # Distance au point le plus proche dans un dataset de référence (ex: gares)
    from sklearn.neighbors import BallTree
    ref_coords = np.radians(ref_gdf[['latitude', 'longitude']].values)
    tree = BallTree(ref_coords, metric='haversine')
    query_coords = np.radians(df[[lat_col, lon_col]].values)
    distances, indices = tree.query(query_coords, k=1)
    df['dist_nearest_station'] = distances[:, 0] * 6371  # km

    # Nombre de POI dans un rayon
    for radius_km in [0.5, 1, 3, 5]:
        counts = tree.query_radius(query_coords, r=radius_km/6371, count_only=True)
        df[f'poi_count_{radius_km}km'] = counts

    # Coordonnées transformées (pour modèles)
    df['lat_rad'] = np.radians(df[lat_col])
    df['lon_rad'] = np.radians(df[lon_col])
    df['lat_sin'] = np.sin(df['lat_rad'])
    df['lat_cos'] = np.cos(df['lat_rad'])
    df['lon_sin'] = np.sin(df['lon_rad'])
    df['lon_cos'] = np.cos(df['lon_rad'])

    # Cluster géographique
    from sklearn.cluster import KMeans
    coords = df[[lat_col, lon_col]].values
    for k in [5, 10, 20]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df[f'geo_cluster_{k}'] = kmeans.fit_predict(coords)
        # Distance au centroïde
        dists = kmeans.transform(coords)
        df[f'geo_cluster_{k}_dist'] = dists.min(axis=1)

    # H3 hexagonal indexing (si installé)
    # import h3
    # df['h3_index'] = df.apply(
    #     lambda row: h3.geo_to_h3(row[lat_col], row[lon_col], resolution=7), axis=1
    # )

    return df
```

## 8. Analyse de Proximité

```python
def proximity_analysis(points_gdf, reference_gdf, buffer_distances_m=[500, 1000, 5000]):
    """Analyse de proximité complète."""
    # Projeter en mètres
    points_proj = points_gdf.to_crs(epsg=3857)
    ref_proj = reference_gdf.to_crs(epsg=3857)

    # Distance au point de référence le plus proche
    from shapely.ops import nearest_points
    ref_union = ref_proj.geometry.unary_union

    points_proj['nearest_dist'] = points_proj.geometry.apply(
        lambda p: p.distance(nearest_points(p, ref_union)[1])
    )

    # Comptage dans chaque buffer
    for dist in buffer_distances_m:
        buffers = ref_proj.geometry.buffer(dist).unary_union
        points_proj[f'in_buffer_{dist}m'] = points_proj.geometry.apply(
            lambda p: buffers.contains(p)
        ).astype(int)

    return points_proj
```

## Compétitions Géospatiales Typiques

- **Prédiction immobilière** : prix = f(localisation, voisinage, POI proches)
- **Détection d'anomalies** : patterns spatiaux anormaux (fraude, pollution)
- **Prédiction de demande** : taxis, livraisons, vélos en libre-service
- **Classification d'images satellite** : occupation des sols, déforestation
- **Optimisation de routes** : TSP, couverture optimale

Adapte TOUJOURS les opérations au CRS approprié et vérifie les unités (degrés vs mètres).
