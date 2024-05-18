# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
sys.path.append('/root/project/AirGeoNet')

from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from maploc.osm.viz import GeoPlotter
from maploc.utils.geo import BoundaryBox, Projection
from maploc.osm.tiling import TileManager

from pyproj import Proj, transform
import csv

import cv2 as cv
import torch

from torchvision import transforms as tvf

from PIL import Image

from typing import Literal
import glob
import os

import natsort
import einops as ein
import pickle
from DINOV2.utilities import VLAD,DinoV2ExtractFeatures

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class LocalArgs:
    
    # Input directory containing images
    in_dir: str = "/root/project/AirGeoNet/datasets/VPAir/VPAir/queries"
    # Image file extension
    imgs_ext: str = "png"
    # Output directory where global descriptors will be stored
    out_dir: str = "/root/project/AirGeoNet/datasets/VPAir/VLAD_DataBase"
    # gps information 
    gps_path = '/root/project/AirGeoNet/datasets/VPAir/VPAir/poses.csv'
    # c_center save path
    VLAD_path = '/root/project/AirGeoNet/datasets/VPAir'
    # Maximum edge length (expected) across all images
    max_img_size: int = 1024
 
    # Number of clusters (cluster centers for VLAD) - read from cache
    num_c: int = 8

    desc_layer: int = 31
    desc_facet: Literal["query", "key", "value", "token"] = "value"

    # Domain for use case (deployment environment)
    domain = "vpair"
    # Maximum image dimension
    max_img_size: int = 1024

    device = 'cuda'  


def trans_xy2latlon(x, y):

    utm_proj = Proj('epsg:32633')  # UTM Zone 33N
    wgs84_proj = Proj('epsg:4326')  # WGS 84

    lon, lat= transform(utm_proj, wgs84_proj, x, y)

    return [lat,lon]


def parse_gps_file(path, projection: Projection = None):
    all_latlon = []

    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            all_latlon.append([float(row['lat']),float(row['lon'])])
    
    return np.array(all_latlon)


def prepare_osm(
    data_dir,
    osm_path,
    tile_margin=512,
    ppm=2,
):
    gps_path = '/root/project/AirGeoNet/datasets/VPAir/VPAir/poses.csv'
    all_latlon = parse_gps_file(gps_path)
    
    projection = Projection.from_points(all_latlon)
    all_xy = projection.project(all_latlon)

    bbox_map = BoundaryBox(all_xy.min(0), all_xy.max(0)) + tile_margin
   
    plotter = GeoPlotter()
    plotter.points(all_latlon, "red", name="GPS")
    plotter.bbox(projection.unproject(bbox_map), "blue", "tiling bounding box")

    plotter.fig.write_html("split_vpair.html")

    tile_manager = TileManager.from_bbox(
        projection,
        bbox_map,
        ppm,
        path=osm_path,
    )
    output_path = '/root/project/AirGeoNet/datasets/VPAir/tiles.pkl'
    tile_manager.save(output_path)
    return tile_manager

def prepare_VLAD(
    largs
):
    # Realpath expansion
    _ex = lambda x: os.path.realpath(os.path.expanduser(x))
    # Dino_v2 properties (parameters)

    save_dir = _ex(largs.out_dir)
    device = torch.device(largs.device)
    
    desc_layer: int = largs.desc_layer
    desc_facet: Literal["query", "key", "value", "token"] = largs.desc_facet
    num_c: int = largs.num_c
    domain:str =largs.domain
    max_img_size: int = largs.max_img_size
      
    # Ensure inputs are fine
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        print(f"Creating directory: {save_dir}")
    else:
        print("Save directory already exists, overwriting possible!")

    # Load the DINO extractor model
    extractor = DinoV2ExtractFeatures("dinov2_vitg14", desc_layer,
        desc_facet, device=device)
    base_tf = tvf.Compose([ # Base image transformations
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])

    imgs_dir = _ex(largs.in_dir)
    assert os.path.isdir(imgs_dir), "Input directory doesn't exist!"
    img_fnames = glob.glob(f"{imgs_dir}/*.{largs.imgs_ext}")
    img_fnames = natsort.natsorted(img_fnames)

    imgs_dir = _ex(largs.in_dir)
    assert os.path.isdir(imgs_dir), "Input directory doesn't exist!"
    img_fnames = glob.glob(f"{imgs_dir}/*.{largs.imgs_ext}")
    img_fnames = natsort.natsorted(img_fnames)
    
    img_patch_descs = []
    
    for img_fname in tqdm(img_fnames):
        with torch.no_grad():
            pil_img = Image.open(img_fname).convert('RGB')
            img_pt = base_tf(pil_img).to(device)
            if max(img_pt.shape[-2:]) > max_img_size:
                pass
            c,h,w = img_pt.shape
            h_new, w_new = (h // 14) * 14, (w // 14) * 14
            img_pt = tvf.CenterCrop((h_new,w_new))(img_pt)[None,...]
            ret = extractor(img_pt)
            img_patch_descs.append(ret.to('cpu'))
            

    result_tensor = torch.cat(img_patch_descs, dim=0)
    
    vlad = VLAD(num_c, desc_dim=result_tensor[0].shape[1], cache_dir= _ex(largs.VLAD_path))
    vlad.fit(ein.rearrange(result_tensor, "n k d -> (n k) d"))
    
    all_latlon = parse_gps_file(largs.gps_path)
    vlad_data = []
    for img_fname, ret, latlon in tqdm(zip(img_fnames, img_patch_descs, all_latlon), total=len(img_fnames)):

        # VLAD global descriptor
        gd = vlad.generate(ret.squeeze()) # VLAD:  [agg_dim]
        gd_np = gd.numpy()[np.newaxis, ...] # shape: [1, agg_dim]
        vlad_data.append({'gd_np':gd_np,'latlon':latlon})

    with open(f"{save_dir}/vlad_descriptors.pkl", 'wb') as file:
        pickle.dump(vlad_data, file)
        


if __name__ == "__main__":
    data_dir = '/root/project/AirGeoNet/datasets/VPAir/VPAir/queries'
    # osm_path = Path('/root/project/AirGeoNet/datasets/VPAir/map.osm')
    # pixel_per_meter = 2
    # prepare_osm(data_dir, osm_path, ppm = pixel_per_meter)

    

    prepare_VLAD(LocalArgs)

    
