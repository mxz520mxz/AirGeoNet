from typing import Optional, Tuple

import torch
import numpy as np

from . import logger
from .evaluation.run import resolve_checkpoint_path, pretrained_models
from .models.orienternet import OrienterNet
from .models.voting import fuse_gps, argmax_xyr
from .data.image import resize_image, pad_image, rectify_image
from .osm.raster import Canvas
from .utils.wrappers import Camera
from .utils.io import read_image
from .utils.geo import BoundaryBox, Projection
from .utils.exif import EXIF
from .osm.tiling import TileManager
from maploc.module import GenericModule

import torch.nn.functional as F

try:
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="orienternet")
except ImportError:
    geolocator = None

try:
    from gradio_client import Client

    calibrator = Client("https://jinlinyi-perspectivefields.hf.space/")
except (ImportError, ValueError):
    calibrator = None


def image_calibration(image_path):
    logger.info("Calling the PerspectiveFields calibrator, this may take some time.")
    result = calibrator.predict(
        image_path, "NEW:Paramnet-360Cities-edina-centered", api_name="/predict"
    )
    result = dict(r.rsplit(" ", 1) for r in result[1].split("\n"))
    roll_pitch = float(result["roll"]), float(result["pitch"])
    return roll_pitch, float(result["vertical fov"])


def camera_from_exif(exif: EXIF, fov: Optional[float] = None) -> Camera:
    w, h = image_size = exif.extract_image_size()
    _, f_ratio = exif.extract_focal()
    if f_ratio == 0:
        if fov is not None:
            # This is the vertical FoV.
            f = h / 2 / np.tan(np.deg2rad(fov) / 2)
        else:
            return None
    else:
        f = f_ratio * max(image_size)
    return Camera.from_dict(
        dict(
            model="SIMPLE_PINHOLE",
            width=w,
            height=h,
            params=[f, w / 2 + 0.5, h / 2 + 0.5],
        )
    )
import cv2

def read_input_image(
    image_path: str,
    prior_latlon: Optional[Tuple[float, float]] = None,
    proj = None,
    tile_size_meters: int = 64,
):
    
    image = read_image(image_path)
    with open(image_path, "rb") as fid:
        exif = EXIF(fid, lambda: image.shape[:2])
    
    latlon = None
    if prior_latlon is not None:
        latlon = prior_latlon
        # logger.info("Using prior latlon %s.", prior_latlon)
    latlon = np.array(latlon)
    
    # roll_pitch = None
    # if calibrator is not None:
    #     roll_pitch, fov = image_calibration(image_path)
    # else:
    #     logger.info("Could not call PerspectiveFields, maybe install gradio_client?")
    # if roll_pitch is not None:
    #     logger.info("Using (roll, pitch) %s.", roll_pitch)

    # camera = camera_from_exif(exif, fov)
    # if camera is None:
    #     raise ValueError(
    #         "No camera intrinsics found in the EXIF, provide an FoV guess."
    #     )

    center = proj.project(latlon) # x,y
    # error = np.random.RandomState(None).uniform(-5, 5, size=2)
    # center += error * 128
  
    bbox = BoundaryBox(center -tile_size_meters, center + tile_size_meters)
    # bbox = BoundaryBox(center , center )+ tile_size_meters
    return image, bbox, center


class Demo:
    def __init__(
        self,
        experiment_or_path: Optional[str] = "OrienterNet_MGL",
        device=None,
        **kwargs
    ):
        if experiment_or_path in pretrained_models:
            experiment_or_path, _ = pretrained_models[experiment_or_path]
        path = resolve_checkpoint_path(experiment_or_path)
        ckpt = torch.load(path, map_location=(lambda storage, loc: storage))
        config = ckpt["hyper_parameters"]
        config.model.update(kwargs)
        config.model.image_encoder.backbone.pretrained = False

        # model = OrienterNet(config.model).eval()
        # state = {k[len("model.") :]: v for k, v in ckpt["state_dict"].items()}
        # model.load_state_dict(state, strict=True)

        cfg = {'model': {"num_rotations": 32, "apply_map_prior": True}}
        model = GenericModule.load_from_checkpoint(
            path, strict=False, find_best=not experiment_or_path.endswith('.ckpt'), cfg=cfg)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        self.model = model
        self.config = config
        self.device = device

    # def prepare_data(
    #     self,
    #     image: np.ndarray,
    #     camera: Camera,
    #     canvas: Canvas,
    #     roll_pitch: Optional[Tuple[float]] = None,
    # ):
    #     assert image.shape[:2][::-1] == tuple(camera.size.tolist())
    #     target_focal_length = self.config.data.resize_image / 2
    #     factor = target_focal_length / camera.f
    #     size = (camera.size * factor).round().int()
    #     print(size)
    #     image = torch.from_numpy(image).permute(2, 0, 1).float().div_(255)
    #     valid = None
    #     if roll_pitch is not None:
    #         roll, pitch = roll_pitch
    #         image, valid = rectify_image(
    #             image,
    #             camera.float(),
    #             roll=-roll,
    #             pitch=-pitch,
    #         )
    #     image, _, camera, *maybe_valid = resize_image(
    #         image, size.tolist(), camera=camera, valid=valid
    #     )
    #     valid = None if valid is None else maybe_valid

    #     max_stride = max(self.model.image_encoder.layer_strides)
    #     size = (torch.ceil(size / max_stride) * max_stride).int()
    #     image, valid, camera = pad_image(
    #         image, size.tolist(), camera, crop_and_center=True
    #     )

    #     return dict(
    #         image=image,
    #         map=torch.from_numpy(canvas.raster).long(),
    #         camera=camera.float(),
    #         valid=valid,
    #     )
    
    def pad_feature(self,feats_map):
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

    def prepare_data(
        self,
        image: np.ndarray,
        camera: Camera,
        canvas: Canvas,
        roll_pitch: Optional[Tuple[float]] = None,
        qiyuan_map = None,
    ):
       
        # size = self.config.data.resize_image
       
        image = torch.from_numpy(image).permute(2, 0, 1).float().div_(255)
        # print('image shape:',image.shape)
        image = self.pad_feature(image)
        # print('image shape:',image.shape)
        valid = None
        if roll_pitch is not None:
            roll, pitch, _ = roll_pitch
           
            image, valid = rectify_image(
                image,
                camera.float(),
                roll=-roll,
                pitch=-pitch,
            )
        # image = image.unsqueeze(0)
        # image, _, camera, *maybe_valid = resize_image(
        #     image, size, camera=camera, valid=valid
        # )
       
        # valid = None if valid is None else maybe_valid
        valid = torch.all(image != 0, dim=0)
        # print('valid shape:',valid.shape)
        # max_stride = max(self.model.image_encoder.layer_strides)
        # size = (torch.ceil(size / max_stride) * max_stride).int()
        # image, valid, camera = pad_image(
        #     image, size.tolist(), camera, crop_and_center=True
        # )

        
        map = torch.from_numpy(canvas.raster).long()
        if qiyuan_map is not None:
            qiyuan_map = torch.tensor(qiyuan_map.copy(), dtype=torch.long)
        else:
            qiyuan_map = None

        return dict(
            image=image,
            map=map,
            qiyuan_map = qiyuan_map,
            camera=camera.float(),
            valid=valid,
        )

    def localize(self, image: np.ndarray, camera: Camera, canvas: Canvas, qiyuan_map = None, **kwargs):
       
        data = self.prepare_data(image, Camera.from_dict(camera), canvas, qiyuan_map=qiyuan_map, **kwargs)
        
        data_ = {k: v.to(self.device)[None] if v is not None else None for k, v in data.items()}
       
        with torch.no_grad():
            pred = self.model(data_)

        # xy_gps = canvas.bbox.center
        # uv_gps = torch.from_numpy(canvas.to_uv(xy_gps))

        lp_xyr = pred["log_probs"].squeeze(0)
        # tile_size = canvas.bbox.size.min() 
        # sigma = tile_size - 20  # 20 meters margin
        # lp_xyr = fuse_gps(
        #     lp_xyr,
        #     uv_gps.to(lp_xyr),
        #     self.config.model.pixel_per_meter,
        #     sigma=sigma,
        # )
        xyr = argmax_xyr(lp_xyr).cpu()

        prob = lp_xyr.exp().cpu()
        neural_map = pred["map"]["map_features"][0].squeeze(0).cpu()
        features_image = pred["features_air"].squeeze(0).cpu()
        # print('features_image shape:',features_image.shape)
        return xyr[:2], xyr[2], prob, neural_map, data["image"], features_image, pred
