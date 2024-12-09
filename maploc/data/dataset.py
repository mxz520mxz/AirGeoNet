from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision.transforms as tvf
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import torch.nn.functional as F

from ..models.utils import deg2rad, rotmat2d
from ..osm.tiling import TileManager
from ..utils.geo import BoundaryBox
from ..utils.io import read_image
from ..utils.wrappers import Camera
from .image import pad_image, rectify_image, resize_image
from .utils import decompose_rotmat, random_flip, random_rot90
from torchvision.transforms.functional import rotate
import random

class MapLocDataset(torchdata.Dataset):
    default_cfg = {
        "seed": 0,
        "accuracy_gps": 15,
        "random": True,
        "num_threads": None,
        # map
        "num_classes": None,
        "pixel_per_meter": "???",
        "crop_size_meters": "???",
        "max_init_error": "???",
        "max_init_error_rotation": None,
        "init_from_gps": True,
        "return_gps": False,
        "force_camera_height": None,
        # pose priors
        "add_map_mask": False,
        "mask_radius": None,
        "mask_pad": 1,
        "prior_range_rotation": None,
        # image preprocessing
        "target_focal_length": None,
        "reduce_fov": None,
        "resize_image": None,
        "pad_to_square": False,  # legacy
        "pad_to_multiple": 32,
        "rectify_pitch": True,
        "augmentation": {
            "rot90": False,
            "flip": False,
            "image": {
                "apply": False,
                "brightness": 0.5,
                "contrast": 0.4,
                "saturation": 0.4,
                "hue": 0.5 / 3.14,
            },
        },
    }

    def __init__(
        self,
        stage: str,
        cfg: DictConfig,
        names: List[str],
        data: Dict[str, Any],
        image_dirs: Dict[str, Path],
        tile_managers: Dict[str, TileManager],
        qiyuan_map,
        image_ext: str = "",
        
    ):
        self.stage = stage
        self.cfg = deepcopy(cfg)
        self.data = data
        self.image_dirs = image_dirs
        self.tile_managers = tile_managers
        self.names = names
        self.image_ext = image_ext
        self.qiyuan_map = qiyuan_map
        

        tfs = []
        if stage == "train" and cfg.augmentation.image.apply:
            args = OmegaConf.masked_copy(
                cfg.augmentation.image, ["brightness", "contrast", "saturation", "hue"]
            )
            tfs.append(tvf.ColorJitter(**args))
        self.tfs = tvf.Compose(tfs)

    def __len__(self):
        return len(self.names)
    
    def get_xy(self,target_lat,target_lon,we,ne,px_lat,px_lon):
    
        left_top_lon = we
        left_top_lat = ne

        resolution_lon = 1 / px_lon
        resolution_lat = 1 / px_lat

        delta_lon = target_lon - left_top_lon
        delta_lat = left_top_lat - target_lat  

        pixel_x = delta_lon / resolution_lon
        pixel_y = delta_lat / resolution_lat

        return [np.ceil(pixel_x), np.ceil(pixel_y)]
    
    def get_region(self,latlon_gps,latlon_gps_gt,region_size,we,ne,px_lat,px_lon):
        bg = self.qiyuan_map['bg']
        
        height, width = bg.shape

        # Calculate the center of the image
        coords = self.get_xy(float(latlon_gps[0]),float(latlon_gps[1]),we,ne,px_lat,px_lon)
   
        coords_gt = self.get_xy(float(latlon_gps_gt[0]),float(latlon_gps_gt[1]),we,ne,px_lat,px_lon)

        # Calculate the coordinates of the top-left corner of the region
        top_left_x = int(coords[0] - region_size / 2)
        top_left_y = int(coords[1] - region_size / 2)

        # Ensure that the region stays within the bounds of the image
        top_left_x = max(0, top_left_x)
        top_left_y = max(0, top_left_y)

        # Calculate the bottom-right corner coordinates
        bottom_right_x = min(top_left_x + region_size, width)
        bottom_right_y = min(top_left_y + region_size, height)

        # Extract the region from the background image
        region = bg[top_left_y:bottom_right_y, top_left_x:bottom_right_x]


        x_gt, y_gt = coords_gt

        x_gt_region = x_gt - top_left_x
        y_gt_region = y_gt - top_left_y

        return region,(x_gt_region,y_gt_region)


    def __getitem__(self, idx):
        if self.stage == "train" and self.cfg.random:
            seed = None
        else:
            seed = [self.cfg.seed, idx]
        (seed,) = np.random.SeedSequence(seed).generate_state(1)
        
        name = self.names[idx]
        dataset_name = name.split("datasets/")[1].split("/")[0]

        latlonalt_gps = self.data["gps_position"][idx]

        if "euler" in self.data:
            euler = self.data["euler"][idx]
        else:
            euler = [0,0,0]

        xy_w_init = self.tile_managers[dataset_name].projection.project(latlonalt_gps)

        if "shifts" in self.data:
            yaw = self.data["roll_pitch_yaw"][idx][-1]
            R_c2w = rotmat2d((90 - yaw) / 180 * np.pi).float()
            error = (R_c2w @ self.data["shifts"][idx][:2]).numpy()
        else:
            error = np.random.RandomState(seed).uniform(-5, 5, size=2)
        xy_w_init += error * self.cfg.max_init_error

        if self.qiyuan_map is not None:
            we = self.qiyuan_map['we']
            ne = self.qiyuan_map['ne']
            px_lat = self.qiyuan_map['px_lat']
            px_lon = self.qiyuan_map['px_lon']
            region,(x_region,y_region) = self.get_region(self.tile_managers[dataset_name].projection.unproject(xy_w_init),latlon_gps, 1024,we,ne,px_lat,px_lon) #todo 1024
            
        else:
            region = None
            x_region=None
            y_region=None
     
        bbox_tile = BoundaryBox(
            xy_w_init - self.cfg.crop_size_meters,
            xy_w_init + self.cfg.crop_size_meters,
        )
        return self.get_view(idx, name, seed, self.tile_managers[dataset_name].projection.project(latlonalt_gps), bbox_tile, euler, region,x_region,y_region)

    def get_view(self, idx, name, seed, xy_w_gt, bbox_tile, euler, region,x_region=None,y_region=None):
        data = {
            "index": idx,
            "name": name,
        }
        dataset_name = name.split("datasets/")[1].split("/")[0]
        [roll, pitch, yaw] = euler
       
        cam_dict = self.data["cameras"][self.data["camera_id"][idx]]
        cam = Camera.from_dict(cam_dict).float()

        image = read_image(self.image_dirs / Path('remote_image') / (name + self.image_ext))
        
        if "plane_params" in self.data:
            # transform the plane parameters from world to camera frames
            plane_w = self.data["plane_params"][idx]
            data["ground_plane"] = torch.cat(
                [rotmat2d(deg2rad(torch.tensor(yaw))) @ plane_w[:2], plane_w[2:]]
            )
        if self.cfg.force_camera_height is not None:
            data["camera_height"] = torch.tensor(self.cfg.force_camera_height)
        elif "camera_height" in self.data:
            data["camera_height"] = self.data["height"][idx].clone()

        canvas = self.tile_managers[dataset_name].query(bbox_tile)
        if region is not None:
            uv_gt = np.array([x_region,y_region])
        else:
            uv_gt = canvas.to_uv(xy_w_gt)
      
        
        uv_init = canvas.to_uv(bbox_tile.center)
        raster = canvas.raster  # C, H, W
     

        # image augmentations
        
        # heading = np.deg2rad(90 - yaw)  # fixme

        image = (
                torch.from_numpy(np.ascontiguousarray(image))
                .permute(2, 0, 1)
                .float()
                .div_(255)
            )
  
        image, valid, cam, roll, pitch = self.process_image(
            image, cam, roll, pitch, seed, dataset_name
        )

        image = pad_feature(image)
        if self.stage == "train":
            # if self.cfg.augmentation.rot90:
                # raster, region, uv_gt, heading = random_rot90(raster, uv_gt, heading, seed, region)
                
            # if self.cfg.augmentation.flip:
            #     image, raster, region, uv_gt, heading = random_flip(
            #         image, raster, uv_gt, heading, seed, region
            #     )
            rot_id = random.randint(0,17)
            angles = torch.arange(0.0,360.0,360.0/18)

            angle=angles[rot_id].item()
            image = rotate(image, angle)

            yaw = yaw + angle
        
    
        valid = torch.all(image != 0, dim=0)
        
        # yaw = 90 - np.rad2deg(heading)  # fixme
        # Create the mask for prior location
        # if self.cfg.add_map_mask:
        #     data["map_mask"] = torch.from_numpy(self.create_map_mask(canvas))
            # print('data["map_mask"] shape: ',data["map_mask"].shape)
            # plt.imsave('map_mask.jpg',data["map_mask"].cpu().numpy())

        if self.cfg.max_init_error_rotation is not None:
            if "shifts" in self.data:
                error = self.data["shifts"][idx][-1]
            else:
                error = np.random.RandomState(seed + 1).uniform(-1, 1)
                error = torch.tensor(error, dtype=torch.float)
            yaw_init = yaw + error * self.cfg.max_init_error_rotation
            range_ = self.cfg.prior_range_rotation or self.cfg.max_init_error_rotation
            data["yaw_prior"] = torch.stack([yaw_init, torch.tensor(range_)])
            

        if self.cfg.return_gps:
            gps = self.data["gps_position"][idx][:2].numpy()
            xy_gps = self.tile_managers[dataset_name].projection.project(gps)
            data["uv_gps"] = torch.from_numpy(canvas.to_uv(xy_gps)).float()
            data["accuracy_gps"] = torch.tensor(
                min(self.cfg.accuracy_gps, self.cfg.crop_size_meters)
            )

        if "chunk_index" in self.data:
            data["chunk_id"] = (self.data["chunk_index"][idx])

        if region is not None:
            qiyuan_map = torch.tensor(region.copy(), dtype=torch.long)
        else:
            qiyuan_map = None
        
        return {
            **data,
            "image": image,
            "valid": valid,
            "camera": cam,
            "canvas": canvas,
            "map": torch.from_numpy(np.ascontiguousarray(raster)).long(),
            "uv": torch.from_numpy(uv_gt).float(),  # TODO: maybe rename to uv?
            "uv_init": torch.from_numpy(uv_init).float(),  # TODO: maybe rename to uv?
            "roll_pitch_yaw": torch.tensor((roll, pitch, yaw)).float(),
            "pixels_per_meter": torch.tensor(canvas.ppm).float(),
            "qiyuan_map":qiyuan_map,
        }

    def process_image(self, image, cam, roll, pitch, seed, datasetname):
        # image = (
        #     torch.from_numpy(np.ascontiguousarray(image))
        #     .permute(2, 0, 1)
        #     .float()
        #     .div_(255)
        # )
        #TODO 
      
        image, valid = rectify_image(
            image, cam, roll, pitch if self.cfg.rectify_pitch else None
        )
        
        # plt.imsave('test_image2.jpg',image.squeeze().permute(1,2,0).cpu().numpy())
        roll = 0.0
        if self.cfg.rectify_pitch:
            pitch = 0.0

        if self.cfg.target_focal_length is not None:
            # resize to a canonical focal length
            factor = self.cfg.target_focal_length / cam.f.numpy()
            size = (np.array(image.shape[-2:][::-1]) * factor).astype(int)
            
            image, _, cam, valid = resize_image(image, size, camera=cam, valid=valid)
            size_out = self.cfg.resize_image[datasetname]
            if size_out is None:
                # round the edges up such that they are multiple of a factor
                stride = self.cfg.pad_to_multiple
                size_out = (np.ceil((size / stride)) * stride).astype(int)
            # crop or pad such that both edges are of the given size
            image, valid, cam = pad_image(
                image, size_out, cam, valid, crop_and_center=True
            )
        elif self.cfg.resize_image is not None:
            
            image, _, cam, valid = resize_image(
                image, self.cfg.resize_image[datasetname], camera=cam, valid=valid
            )
            if self.cfg.pad_to_square:
                # pad such that both edges are of the given size
                image, valid, cam = pad_image(image, self.cfg.resize_image[datasetname], cam, valid)
        
        if self.cfg.reduce_fov is not None:
            h, w = image.shape[-2:]
            f = float(cam.f[0])
            fov = np.arctan(w / f / 2)
            w_new = round(2 * f * np.tan(self.cfg.reduce_fov * fov))
            image, valid, cam = pad_image(
                image, (w_new, h), cam, valid, crop_and_center=True
            )
       
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            image = self.tfs(image)
        
        return image, valid, cam, roll, pitch

    def create_map_mask(self, canvas):
        map_mask = np.zeros(canvas.raster.shape[-2:], bool)
        radius = self.cfg.mask_radius or self.cfg.max_init_error
        mask_min, mask_max = np.round(
            canvas.to_uv(canvas.bbox.center)
            + np.array([[-1], [1]]) * (radius + self.cfg.mask_pad) * canvas.ppm
        ).astype(int)
        map_mask[mask_min[1] : mask_max[1], mask_min[0] : mask_max[0]] = True
        return map_mask


def pad_feature(feats_map):
    height = feats_map.size(-2)
    width  = feats_map.size(-1)
    diagonal_length = int(np.ceil(np.sqrt(height**2 + width**2)))

    padding_vertical = int(np.ceil((diagonal_length - height) / 2))
    padding_horizontal = int(np.ceil((diagonal_length - width) / 2))

    pad_top = padding_vertical // 2
    pad_bottom = padding_vertical - pad_top
    pad_left = padding_horizontal // 2
    pad_right = padding_horizontal - pad_left

    padded_tensor = F.pad(feats_map, (pad_left, pad_right, pad_top, pad_bottom, 0, 0))

    return padded_tensor