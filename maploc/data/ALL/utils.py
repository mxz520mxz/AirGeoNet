# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
from PIL.ExifTags import TAGS

from ...utils.geo import Projection

split_files = ["test1_files.txt", "test2_files.txt", "train_files.txt"]


def parse_gps_file(path, projection: Projection = None):
    with open(path, "r") as fid:
        lat, lon, _, roll, pitch, yaw, *_ = map(float, fid.read().split())
    latlon = np.array([lat, lon])
    R_world_gps = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
    t_world_gps = None if projection is None else np.r_[projection.project(latlon), 0]
    return latlon, R_world_gps, t_world_gps

def extract_gps_info(image_path):
   
        img = Image.open(image_path)

        exif_data = img._getexif()

        gps_info = {}

        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            gps_info[tag_name] = value

        if 'GPSInfo' in gps_info:
            
            (degrees, minutes, seconds) = gps_info['GPSInfo'][2]
            lat = float(degrees) + (float(minutes) / 60) + (float(seconds) / 3600)
            (degrees, minutes, seconds) = gps_info['GPSInfo'][4]
            lon = float(degrees) + (float(minutes) / 60) + (float(seconds) / 3600)
            alt = float(gps_info['GPSInfo'][6])
        else:
            raise 'no gps information'
        return [lat,lon,alt]

def parse_split_file(path: Path):
    with open(path, "r") as fid:
        info = fid.read()
    names = []
    others = []
    for line in info.split("\n"):
        if not line:
            continue
        name, *other = line.split(',')
        names.append(name)
        if len(other) > 0:
            others.append(np.array(other, float))
        else:
            # infomation stored in the image, resume euler are zero
            [lat,lon,alt] = extract_gps_info(name)
            others.append([float(lat),float(lon),float(alt),0,0,0])

    others = None if len(others) == 0 else np.stack(others)
    
    return names, others


def parse_calibration_file(path):
    calib = {}
    with open(path, "r") as fid:
        for line in fid.read().split("\n"):
            if not line:
                continue
            key, *data = line.split(" ")
            key = key.rstrip(":")
            if key.startswith("R"):
                data = np.array(data, float).reshape(3, 3)
            elif key.startswith("T"):
                data = np.array(data, float).reshape(3)
            elif key.startswith("P"):
                data = np.array(data, float).reshape(3, 4)
            calib[key] = data
    return calib

#Corrections needed
def get_camera_calibration():
    K = np.array([[320.0,0.0,256.0],
                  [0.0,320.0,256.0],
                  [0.0,0.0,1.0]])
    size = np.array([640,480])
    camera = {
        "model": "PINHOLE",
        "width": size[0],
        "height": size[1],
        "params": K[[0, 1, 0, 1], [0, 1, 2, 2]],
    }

    return camera
