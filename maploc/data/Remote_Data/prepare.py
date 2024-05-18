# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
sys.path.append('/root/project/OrienterNet')
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from maploc.osm.viz import GeoPlotter
from maploc.utils.geo import BoundaryBox, Projection
from maploc.osm.tiling import TileManager
import pickle


def prepare_osm(
    data_dir,
    osm_path,
    tile_margin=512,
    ppm=2,
):
    
    with open(data_dir, 'rb') as f:
        remote_data = pickle.load(f)

    all_latlon = []
    for data in remote_data:
        all_latlon.append(data['latlon'])
    all_latlon = np.array(all_latlon)
    print('all_latlon: ',all_latlon)
    
    projection = Projection.from_points(all_latlon)
    all_xy = projection.project(all_latlon)
    print('allxy:',all_xy)
    print('allxy shape:',all_xy.shape)
    bbox_map = BoundaryBox(all_xy.min(0), all_xy.max(0)) + tile_margin
    print('bbox_map: ',bbox_map)

    plotter = GeoPlotter()
    plotter.points(all_latlon, "red", name="GPS")
    plotter.bbox(projection.unproject(bbox_map), "blue", "tiling bounding box")

    plotter.fig.write_html("/root/project/OrienterNet/maploc/data/Remote_Data/split_kitti.html")

    tile_manager = TileManager.from_bbox(
        projection,
        bbox_map,
        ppm,
        path=osm_path,
    )
    output_path = '/root/project/OrienterNet/datasets/Remote_Data/tiles.pkl'
    tile_manager.save(output_path)
    return tile_manager



if __name__ == "__main__":
    data_dir = '/root/project/OrienterNet/datasets/Remote_Data/remote_data.pkl'
    
    
    osm_path = Path('/root/project/OrienterNet/datasets/Remote_Data/map.osm')
    pixel_per_meter = 2
    prepare_osm(data_dir, osm_path, ppm = pixel_per_meter)

    
