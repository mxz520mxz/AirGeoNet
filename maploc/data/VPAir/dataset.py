# Copyright (c) Bistu. mxz.

import collections
import collections.abc
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import re
import pytorch_lightning as pl
import torch
import torch.utils.data as torchdata
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from ... import logger, DATASETS_PATH
from ...osm.tiling import TileManager
from ..dataset import MapLocDataset
from ..torch import collate, worker_init_fn
from .utils import parse_split_file, parse_gps_file, get_camera_calibration


class VPAirDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "VPAir",
        # paths and fetch
        "data_dir": DATASETS_PATH / "VPAir",
        "tiles_filename": "tiles.pkl",
        "splits": {
            "train": "train_files.txt",
            "val": "val_files.txt",
            "test": "test_files.txt",
        },
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "max_num_val": 500,
        "selection_subset_val": "furthest",
        "drop_train_too_close_to_val": 5.0,
        "skip_frames": 1,
        "camera_index": 2,
        # overwrite
        "crop_size_meters": 64,
        "max_init_error": 20,
        "max_init_error_rotation": 10,
        "add_map_mask": True,
        "mask_pad": 2,
        "target_focal_length": None,
    }
    dummy_scene_name = "VPAir"

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.tile_manager = tile_manager
        self.qiyuan_map = None
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        assert self.cfg.selection_subset_val in ["random", "furthest"]
        self.splits = {}
        self.others = {}
        self.shifts = None
        self.calibrations = None
        self.data = {}
        self.image_paths = {}

    def prepare_data(self):
        if not (self.root.exists()):
            raise FileNotFoundError(
                "Cannot find the VPAir dataset"
            )

    def parse_split(self, split_arg):
        if isinstance(split_arg, str):
            names, others = parse_split_file(self.root / split_arg)
        elif isinstance(split_arg, collections.abc.Sequence):
            names = []
            shifts = None
            for date_drive in split_arg:
                data_dir = (
                    self.root / date_drive / f"image_{self.cfg.camera_index:02}/data"
                )
                assert data_dir.exists(), data_dir
                date_drive = tuple(date_drive.split("/"))
                n = sorted(date_drive + (p.name,) for p in data_dir.glob("*.png"))
                names.extend(n[:: self.cfg.skip_frames])
        else:
            raise ValueError(split_arg)
        return names, others

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]
        for stage in stages:
            self.splits[stage], self.others[stage] = self.parse_split(
                self.cfg.splits[stage]
            )
        do_val_subset = "val" in stages and self.cfg.max_num_val is not None
        if do_val_subset and self.cfg.selection_subset_val == "random":
            select = np.random.RandomState(self.cfg.seed).choice(
                len(self.splits["val"]), self.cfg.max_num_val, replace=False
            )
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]

        self.calibrations = get_camera_calibration()
        if self.tile_manager is None:
            logger.info("Loading the tile manager...")
            self.tile_manager = {self.cfg.name:TileManager.load(self.root / self.cfg.tiles_filename)}
        self.cfg.num_classes = {k: len(g) for k, g in self.tile_manager[self.cfg.name].groups.items()}
        self.cfg.pixel_per_meter = self.tile_manager[self.cfg.name].ppm
        
        self.pack_data(stages)

    def pack_data(self, stages):
        for stage in stages:
            names = []
            data = {}
            gps_position = []
            euler = []
            for path in enumerate(self.splits[stage]):
                names.append(path[1])
            for other in enumerate(self.others[stage]):
                other = other[1]
                gps_position.append([other[0],other[1],other[2]])
                euler.append([other[3],other[4],other[5]])
            
            data["camera_id"] = np.full(len(names), self.cfg.camera_index)

            data["cameras"] = {
                self.cfg.camera_index: self.calibrations
            }

            data['gps_position'] = gps_position
            data['euler'] = euler
            # shifts = self.shifts[stage]
            # if shifts is not None:
            #     data["shifts"] = torch.from_numpy(shifts.astype(np.float32))
            self.data[stage] = data
            self.image_paths[stage] = np.array(names)
            

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            self.root,
            self.tile_manager,
            self.qiyuan_map,
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

