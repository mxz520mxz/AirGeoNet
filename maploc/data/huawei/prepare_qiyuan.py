# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
sys.path.append('/root/project/OrienterNet')

from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from maploc.osm.viz import GeoPlotter
from maploc.utils.geo import BoundaryBox, Projection
from maploc.osm.tiling import TileManager

from typing import Literal
import glob
import os

import natsort
import einops as ein
import pickle
from DINOV2.utilities import VLAD,DinoV2ExtractFeatures
import torch
from torchvision import transforms as tvf
from PIL import Image
import xml.etree.ElementTree as ET

def parse_gps_file(path, projection: Projection = None):
    # with open(path, "r") as fid:
    #     print('fid.read(): ',fid.read())
    #     print('fid.read() type: ',type(fid.read()))
    #     lat, lon = map(float, fid.read().split())
    # latlon = np.array([lat, lon])

    all_latlon = []
    with open(path, 'r') as fid:
        for line in fid:
            lat_str, lon_str = line.strip().split()
            latlon_values = [float(lat_str), float(lon_str)]
            all_latlon.append(latlon_values)
    
    return np.array(all_latlon)


def prepare_osm(
    data_dir,
    osm_path,
    tile_margin=512,
    ppm=2,
):
    gps_path = '/home/Dataset/UAVwithGPS/huaiwei/latlon.txt'
    all_latlon = parse_gps_file(gps_path)
    print('all_latlon: ',all_latlon.shape)
    
    projection = Projection.from_points(all_latlon)
    all_xy = projection.project(all_latlon)
    print('allxy:',all_xy)
    print('allxy shape:',all_xy.shape)
    bbox_map = BoundaryBox(all_xy.min(0), all_xy.max(0)) + tile_margin
    print('bbox_map: ',bbox_map)



    plotter = GeoPlotter()
    plotter.points(all_latlon, "red", name="GPS")
    plotter.bbox(projection.unproject(bbox_map), "blue", "tiling bounding box")

    plotter.fig.write_html("split_kitti.html")

    tile_manager = TileManager.from_bbox(
        projection,
        bbox_map,
        ppm,
        path=osm_path,
    )
    output_path = '/root/project/OrienterNet/datasets/huawei/tiles.pkl'
    tile_manager.save(output_path)
    return tile_manager



if __name__ == "__main__":
    data_dir = '/home/Dataset/UAVwithGPS/huaiwei'
    osm_path = Path('/home/Dataset/UAVwithGPS/huaiwei/map_huawei.osm')
    pixel_per_meter = 2
    prepare_osm(data_dir, osm_path, ppm = pixel_per_meter)

    
    
