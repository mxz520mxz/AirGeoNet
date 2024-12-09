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
    shifts = []
    for line in info.split("\n"):
        if not line:
            continue
        name, *shift = line.split()
        names.append(tuple(name.split("/")))
        if len(shift) > 0:
            assert len(shift) == 3
            shifts.append(np.array(shift, float))
    shifts = None if len(shifts) == 0 else np.stack(shifts)
    return names, shifts


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
