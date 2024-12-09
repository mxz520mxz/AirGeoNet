import sys
sys.path.append('/root/project/AirGeoNet')

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




import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class LocalArgs:
    
    # Input directory containing images
    in_dir: str = "/home/Dataset/beijing_remote/remote_image/"
    # Image file extension
    imgs_ext: str = "tif"
    # Output directory where global descriptors will be stored
    out_dir: str = "/root/project/OrienterNet/datasets/huawei/VLAD_DataBase/"
    
    # c_center save path
    VLAD_path = '/root/project/OrienterNet/datasets/huawei'
    # Maximum edge length (expected) across all images
    max_img_size: int = 1024

    # Number of clusters (cluster centers for VLAD) - read from cache
    num_c: int = 32

    desc_layer: int = 31
    desc_facet: Literal["query", "key", "value", "token"] = "value"

    # Domain for use case (deployment environment)
    domain = "huawei"
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
        if 'beijing_5' in filename and 'tif' in filename:     
            if 'beijing_5-' in filename:
                continue
            img_path = imgs_dir +'/'+ filename       
            img_fnames.append(img_path)
            kml_file_path = img_path.replace('.tif','.kml')
            kml_fnames.append(kml_file_path)
            

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

from PIL import Image
from PIL.ExifTags import TAGS


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

def prepare_osm(
    data_dir,
    osm_path,
    tile_margin=512,
    ppm=2,
):
    # gps_path = '/root/project/OrienterNet/datasets/huawei/latlon_train.txt'
    # all_latlon = parse_gps_file(gps_path)
    # print('all_latlon: ',all_latlon.shape)
    all_latlonalt = []
    for img_path in sorted(os.listdir(data_dir)):
        all_latlonalt.append(extract_gps_info(data_dir + img_path))
    all_latlonalt = np.asarray(all_latlonalt)
    print('all_latlonalt: ',all_latlonalt)
    all_latlonalt = np.load('/root/project/AirGeoNet/all_latlon.npy')
    projection = Projection.from_points(all_latlonalt)
    all_xy = projection.project(all_latlonalt)
    print('allxy:',all_xy)
    print('allxy shape:',all_xy.shape)
    bbox_map = BoundaryBox(all_xy.min(0), all_xy.max(0)) + tile_margin
    print('bbox_map: ',bbox_map)

    plotter = GeoPlotter()
    
    plotter.points(all_latlonalt[:,:2], "red", name="GPS")
    plotter.bbox(projection.unproject(bbox_map), "blue", "tiling bounding box")

    plotter.fig.write_html("test_train_split_kitti.html")

    tile_manager = TileManager.from_bbox(
        projection,
        bbox_map,
        ppm,
        path=osm_path,
    )
    output_path = '/root/project/AirGeoNet/datasets/huawei/tiles.pkl'
    tile_manager.save(output_path)
    return tile_manager



if __name__ == "__main__":
    data_dir = '/root/project/AirGeoNet/datasets/huawei/remote_image/'
    osm_path = Path('/root/project/AirGeoNet/datasets/huawei/map_huawei.osm')
    pixel_per_meter = 2
    prepare_osm(data_dir, osm_path, ppm = pixel_per_meter)

    # prepare_VLAD(LocalArgs)
    
