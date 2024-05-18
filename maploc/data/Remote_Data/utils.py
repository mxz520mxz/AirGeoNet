# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ...utils.geo import Projection

split_files = ["test1_files.txt", "test2_files.txt", "train_files.txt"]


def parse_gps_file(path, projection: Projection = None):
    with open(path, "r") as fid:
        lat, lon, _, roll, pitch, yaw, *_ = map(float, fid.read().split())
    latlon = np.array([lat, lon])
    R_world_gps = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
    t_world_gps = None if projection is None else np.r_[projection.project(latlon), 0]
    return latlon, R_world_gps, t_world_gps


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
    others = None if len(others) == 0 else np.stack(others)
    
    return names, others


# def parse_calibration_file(path):
#     calib = {}
#     with open(path, "r") as fid:
#         for line in fid.read().split("\n"):
#             if not line:
#                 continue
#             key, *data = line.split(" ")
#             key = key.rstrip(":")
#             if key.startswith("R"):
#                 data = np.array(data, float).reshape(3, 3)
#             elif key.startswith("T"):
#                 data = np.array(data, float).reshape(3)
#             elif key.startswith("P"):
#                 data = np.array(data, float).reshape(3, 4)
#             calib[key] = data
#     return calib

#Corrections needed
def get_camera_calibration():
    K = np.array([[400.0,0.0,400],
                  [0.0,400.0,400],
                  [0.0,0.0,1.0]])
    size = np.array([800,600])
    camera = {
        "model": "PINHOLE",
        "width": size[0],
        "height": size[1],
        "params": K[[0, 1, 0, 1], [0, 1, 2, 2]],
    }

    return camera


#Corrections needed
# def get_camera_calibration():
#     K = np.array([[750.62614972,0.0,402.41007535],
#                   [0.0,750.26301185,292.98832147],
#                   [0.0,0.0,1.0]])
#     size = np.array([800,600])
#     camera = {
#         "model": "PINHOLE",
#         "width": size[0],
#         "height": size[1],
#         "params": K[[0, 1, 0, 1], [0, 1, 2, 2]],
#     }

#     return camera