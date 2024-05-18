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

import json




import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class LocalArgs:
    
    # Input directory containing images
    in_dir: str = "/home/Dataset/HDairport_GPS_tagged/"
    # Image file extension
    imgs_ext: str = "jpg"
    # Output directory where global descriptors will be stored
    out_dir: str = "/root/project/OrienterNet/datasets/HDAirport/VLAD_DataBase/"
    
    # c_center save path
    VLAD_path = '/root/project/OrienterNet/datasets/HDAirport'
    # Maximum edge length (expected) across all images
    max_img_size: int = 1024

    # Number of clusters (cluster centers for VLAD) - read from cache
    num_c: int = 32

    desc_layer: int = 31
    desc_facet: Literal["query", "key", "value", "token"] = "value"

    # Domain for use case (deployment environment)
    domain = "HDAirport"
    # Maximum image dimension
    max_img_size: int = 1024

    device = 'cuda'  


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
    filenames = os.listdir(imgs_dir)
    img_fnames = []
    kml_fnames = []
    for filename in filenames:
        img_path = imgs_dir +'/'+ filename       
        img_fnames.append(img_path)
            

    img_patch_descs = []
    
    for img_fname in tqdm(img_fnames):
        with torch.no_grad():
            pil_img = Image.open(img_fname).convert('RGB')
            # pil_img = pil_img.resize((640, 480))
            img_pt = base_tf(pil_img).to(device)
            if max(img_pt.shape[-2:]) > max_img_size:
                pass
            c,h,w = img_pt.shape
            h_new, w_new = (h // 14) * 14, (w // 14) * 14
            img_pt = tvf.CenterCrop((h_new,w_new))(img_pt)[None,...]
            ret = extractor(img_pt)
            img_patch_descs.append(ret.to('cpu'))
            

    lat_lons = []
    
    for kml_file_path in tqdm(kml_fnames):
        tree = ET.parse(kml_file_path)
        root = tree.getroot()

        lat_lon_box = root.find('.//LatLonBox')

        north = float(lat_lon_box.find('north').text)
        south = float(lat_lon_box.find('south').text)
        east = float(lat_lon_box.find('east').text)
        west = float(lat_lon_box.find('west').text)

        lat_lons.append([(north+south)/2,(east + west)/2])
       
            

    result_tensor = torch.cat(img_patch_descs, dim=0)


    
    vlad = VLAD(num_c, desc_dim=result_tensor[0].shape[1], cache_dir= _ex(largs.VLAD_path))
    vlad.fit(ein.rearrange(result_tensor, "n k d -> (n k) d"))
    

    vlad_data = []
    for img_fname, ret, latlon in tqdm(zip(img_fnames, img_patch_descs, lat_lons), total=len(img_fnames)):

        # VLAD global descriptor
        gd = vlad.generate(ret.squeeze()) # VLAD:  [agg_dim]
        gd_np = gd.numpy()[np.newaxis, ...] # shape: [1, agg_dim]
        vlad_data.append({'gd_np':gd_np,'latlon':latlon,'f_name':img_fname})

    with open(f"{save_dir}/vlad_descriptors.pkl", 'wb') as file:
        pickle.dump(vlad_data, file)




def prepare_osm(
    data_dir,
    osm_path,
    tile_margin=512,
    ppm=2,
):
    # 读取 JSON 文件
    with open(data_dir, 'r') as json_file:
        loaded_data = json.load(json_file)

  
    all_latlon = []
    for data in loaded_data:
        all_latlon.append(data['lat_lon'])
    all_latlon = np.array(all_latlon)
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

    plotter.fig.write_html("/root/project/OrienterNet/maploc/data/HDAirport/split_kitti.html")

    tile_manager = TileManager.from_bbox(
        projection,
        bbox_map,
        ppm,
        path=osm_path,
    )
    output_path = '/root/project/OrienterNet/datasets/HDAirport/tiles.pkl'
    tile_manager.save(output_path)
    return tile_manager



if __name__ == "__main__":
    data_dir = '/root/project/OrienterNet/datasets/HDAirport/gps_data.json'
    osm_path = Path('/root/project/OrienterNet/datasets/HDAirport/HDAirport_map.osm')
    pixel_per_meter = 2
    prepare_osm(data_dir, osm_path, ppm = pixel_per_meter)

    # prepare_VLAD(LocalArgs)
    
