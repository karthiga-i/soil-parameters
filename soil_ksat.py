# %%
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import csv
from pathlib import Path

# %%
import re
import io
from dataclasses import dataclass
from typing import List, Tuple, Dict
import pandas as pd
from owslib.wcs import WebCoverageService
from osgeo import gdal
import rasterio
from rasterio.io import MemoryFile
from pyproj import Transformer
from rosetta import rosetta, SoilData
from rasterio import plot

# %%
# Create data directory 
os.makedirs('./data', exist_ok=True)

# %%
def get_soil_texure_only(lat, lon):
    # location coordinates (lat, lon)
    # lat = 37.7302
    # lon = -6.5403

    print(f"Getting soil texture for location: Lat {lat}, Lon {lon}")
    print("="*70)

    # Transform WGS84 (EPSG:4326) to Homolosine projection
    # ISRIC SoilGrids uses a custom Homolosine projection
    # We'll use the PROJ string directly
    homolosine_proj = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs("EPSG:4326", homolosine_proj, always_xy=True)
    x, y = transformer.transform(lon, lat)

    print(f"Transformed coordinates: X={x:.2f}, Y={y:.2f}")

    # Create a small bounding box around the point (±200 meters)
    buffer = 200
    bbox = (x - buffer, y - buffer, x + buffer, y + buffer)

    print(f"Bounding box: {bbox}")
    print("="*70)

    # Resolution in meters
    resolution = 250

    # Fetch Sand content
    print("\n1. Fetching Sand content (0-5cm depth)...")
    wcs_sand = WebCoverageService('http://maps.isric.org/mapserv?map=/map/sand.map', 
                                version='1.0.0')

    response_sand = wcs_sand.getCoverage(
        identifier='sand_0-5cm_mean',
        crs='urn:ogc:def:crs:EPSG::152160',
        bbox=bbox,
        resx=resolution, resy=resolution,
        format='GEOTIFF_INT16'
    )

    with open('./data/point_sand_0-5cm.tif', 'wb') as f:
        f.write(response_sand.read())

    # Fetch Silt content
    print("2. Fetching Silt content (0-5cm depth)...")
    wcs_silt = WebCoverageService('http://maps.isric.org/mapserv?map=/map/silt.map', 
                                version='1.0.0')

    response_silt = wcs_silt.getCoverage(
        identifier='silt_0-5cm_mean',
        crs='urn:ogc:def:crs:EPSG::152160',
        bbox=bbox,
        resx=resolution, resy=resolution,
        format='GEOTIFF_INT16'
    )

    with open('./data/point_silt_0-5cm.tif', 'wb') as f:
        f.write(response_silt.read())

    # Fetch Clay content
    print("3. Fetching Clay content (0-5cm depth)...")
    wcs_clay = WebCoverageService('http://maps.isric.org/mapserv?map=/map/clay.map', 
                                version='1.0.0')

    response_clay = wcs_clay.getCoverage(
        identifier='clay_0-5cm_mean',
        crs='urn:ogc:def:crs:EPSG::152160',
        bbox=bbox,
        resx=resolution, resy=resolution,
        format='GEOTIFF_INT16'
    )

    with open('./data/point_clay_0-5cm.tif', 'wb') as f:
        f.write(response_clay.read())

    print("\nData successfully downloaded!")
    print("="*70)

    # Read and extract values at the point location
    print("\nExtracting soil texture values at the point location...")
    print("="*70)

    # Open the raster files
    sand = rasterio.open("./data/point_sand_0-5cm.tif")
    silt = rasterio.open("./data/point_silt_0-5cm.tif")
    clay = rasterio.open("./data/point_clay_0-5cm.tif")

    # Get the pixel coordinates for our point
    row, col = sand.index(x, y)

    # Read the values
    sand_data = sand.read(1)
    silt_data = silt.read(1)
    clay_data = clay.read(1)

    # Extract values at the point
    sand_pct = sand_data[row, col]/10
    silt_pct = silt_data[row, col]/10
    clay_pct = clay_data[row, col]/10

    def get_soil_texture_class(sand_pct, silt_pct, clay_pct):
        """
        Determine USDA soil texture class based on sand, silt, and clay percentages
        """
        if silt_pct >= 80 and clay_pct < 12:
            return "Silt"

        # Sand
        elif sand_pct >= 85 and silt_pct <= 15 and clay_pct <= 10:
            return "Sand"

        # Loamy Sand
        elif (sand_pct >= 70 and sand_pct < 90) and clay_pct <= 15 and silt_pct <= 30:
            return "Loamy Sand"

        # Sandy Clay
        elif clay_pct >= 35 and sand_pct >= 45:
            return "Sandy Clay"

        # Silty Clay
        elif clay_pct >= 40 and silt_pct >= 40:
            return "Silty Clay"

        # Clay (the rest of high-clay region)
        elif clay_pct >= 40 and sand_pct < 45 and silt_pct < 40:
            return "Clay"

        # Sandy Clay Loam
        elif (clay_pct >= 20 and clay_pct < 35) and (sand_pct >= 45 and sand_pct <= 65):
            return "Sandy Clay Loam"

        # Silty Clay Loam
        elif (clay_pct >= 27 and clay_pct < 40) and (silt_pct >= 40) and (sand_pct <= 20):
            return "Silty Clay Loam"

        # Clay Loam
        elif (clay_pct >= 27 and clay_pct < 40) and (sand_pct >= 20 and sand_pct < 45) and (silt_pct >= 15 and silt_pct < 53):
            return "Clay Loam"

        # Sandy Loam
        # (covers the sandy corner with moderate clay; includes a small low-clay tail)
        elif ((clay_pct >= 7 and clay_pct < 20) and (sand_pct >= 43 and sand_pct < 85)) or (clay_pct < 7 and sand_pct >= 52 and silt_pct < 50):
            return "Sandy Loam"

        # Silt Loam
        elif (silt_pct >= 50 and clay_pct < 27) or (silt_pct >= 50 and clay_pct < 12):
            return "Silt Loam"

        # Loam
        elif (clay_pct >= 7 and clay_pct < 27) and (silt_pct >= 28 and silt_pct < 50) and (sand_pct <= 52):
            return "Loam"

        # Fallbacks near boundaries (rare floating-point edge cases)
        if silt_pct >= 50 and clay_pct < 27:
            return "Silt Loam"
        if sand_pct >= 43 and clay_pct >= 7 and clay_pct < 20:
            return "Sandy Loam"

        else:# If we get here, numbers are likely on a triangle edge not captured above
            return "Loam"

    texture_class = get_soil_texture_class(sand_pct, silt_pct, clay_pct)


    return sand_pct, silt_pct, clay_pct, texture_class

sand_pct, silt_pct, clay_pct, texture_class = get_soil_texure_only(lat = 37.7302, lon = -6.5403)
print(f"{sand_pct}, {silt_pct}, {clay_pct}, {texture_class}")


# %%
# Prepare input data for Rosetta
# Rosetta expects percentages (not g/kg)
# Model 2: Sand, Silt, Clay percentages

def get_values(sand_pct, silt_pct, clay_pct, texture_class):
    print(f"\nInput data:")
    print(f"  Sand: {sand_pct:.1f}%")
    print(f"  Silt: {silt_pct:.1f}%")
    print(f"  Clay: {clay_pct:.1f}%")
    print(f"  Texture class: {texture_class}")

    # Create SoilData object
    # Using model 2 (SSC - Sand, Silt, Clay)
    soil_data = SoilData.from_array([
        [sand_pct, silt_pct, clay_pct]
    ])

    # Run Rosetta model
    print("\nRunning Rosetta PTF model...")

    # Model 2: Uses only sand, silt, clay percentages
    mean, stdev, codes = rosetta(2, soil_data)

    # Extract van Genuchten parameters and Ksat
    # Rosetta returns: theta_r, theta_s, alpha, n, Ksat (log10 cm/day)
    theta_r = mean[0, 0]  # Residual water content
    theta_s = mean[0, 1]  # Saturated water content
    alpha = mean[0, 2]    # van Genuchten alpha (1/cm)
    n = mean[0, 3]        # van Genuchten n (-)
    log_ksat = mean[0, 4] # log10(Ksat) in cm/day

    # Convert Ksat from log10(cm/day) to different units
    ksat_cm_day = 10 ** log_ksat
    ksat_cm_hr = ksat_cm_day / 24
    ksat_mm_hr = ksat_cm_hr * 10
    ksat_m_day = ksat_cm_day / 100

    print("\n" + "="*70)
    print("HYDRAULIC PROPERTIES (VAN GENUCHTEN PARAMETERS)")
    print("="*70)
    print(f"Residual water content (θr):    {theta_r:.4f} cm³/cm³")
    print(f"Saturated water content (θs):   {theta_s:.4f} cm³/cm³")
    print(f"van Genuchten alpha (α):        {alpha:.4f} 1/cm")
    print(f"van Genuchten n:                {n:.4f} -")

    print("\n" + "="*70)
    print("SATURATED HYDRAULIC CONDUCTIVITY (Ksat)")
    print("="*70)
    print(f"Ksat: {ksat_cm_day:.2f} cm/day")
    print(f"Ksat: {ksat_cm_hr:.2f} cm/hr")
    print(f"Ksat: {ksat_mm_hr:.2f} mm/hr")
    print(f"Ksat: {ksat_m_day:.4f} m/day")

    return (float(theta_r), float(theta_s), float(alpha), float(n), float(ksat_cm_day), float(ksat_m_day) , float(ksat_mm_hr))

get_values (sand_pct, silt_pct, clay_pct, texture_class)

# %%

def normalize(s: str) -> str:
    # Strip BOMs, spaces, and unify case
    return (s or "").replace("\ufeff", "").strip().lower()

def find_col(reader, *candidates):
    # Map normalized header -> original header
    norm_map = {normalize(h): h for h in reader.fieldnames if h is not None}
    for cand in candidates:
        h = norm_map.get(normalize(cand))
        if h:
            return h
    # try common synonyms
    for syns in (("latitude","lat"), ("longitude","lon","long")):
        if any(c in candidates for c in syns):
            for s in syns:
                h = norm_map.get(s)
                if h:
                    return h
    raise KeyError(f"Required column {candidates} not found in input CSV headers: {reader.fieldnames}")

def process_csv(input_csv, output_csv):
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    with input_path.open(newline="", encoding="utf-8-sig") as f_in, \
         output_path.open("w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=[
            "ID", "Latitude", "Longitude", "Sand", "Silt", "Clay", "Texture", 
            "theta_r", "theta_s", "alpha", "n", "ksat_cm_day", "ksat_m_day", "ksat_mm_hr"])
        writer.writeheader()



        id_col  = find_col(reader, "ID")
        lat_col = find_col(reader, "Latitude")
        lon_col = find_col(reader, "Longitude")

        for i, row in enumerate(reader, start=1):
            id_val = row[id_col]
            lat = float(row[lat_col])
            lon = float(row[lon_col])

            sand, silt, clay, texture = get_soil_texure_only(lat, lon)
            theta_r, theta_s, alpha, n, ksat_cm_day, ksat_m_day, ksat_mm_hr = get_values (sand, silt, clay, texture)

            # Write only the requested columns
            writer.writerow({
                "ID": id_val,
                "Latitude": f"{lat:.6f}",
                "Longitude": f"{lon:.6f}",
                "Sand": f"{sand:.6f}",
                "Silt": f"{silt:.6f}",
                "Clay": f"{clay:.6f}",
                "Texture": texture,
                "theta_r": f"{theta_r:.6f}",
                "theta_s": f"{theta_s:.6f}",
                "alpha": f"{alpha:.6f}",
                "n": f"{n:.6f}",
                "ksat_cm_day": f"{ksat_cm_day:.6f}",
                "ksat_m_day": f"{ksat_m_day:.6f}",
                "ksat_mm_hr": f"{ksat_mm_hr:.6f}",
            })

process_csv("../Locations__soil2.csv", "Soil_texure.csv")
