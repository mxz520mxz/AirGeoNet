import numpy as np
import torch
from torch.nn.functional import normalize
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


from . import get_model
from .base import BaseModel
from .air_net import AirNet

from .voting import (
    argmax_xyr,
    conv2d_fft_batchwise,
    expectation_xyr,
    log_softmax_spatial,
    mask_yaw_prior,
    nll_loss_xyr,
    nll_loss_xyr_smoothed,
    TemplateSampler,
)
from .map_encoder import MapEncoder
from .metrics import AngleError, AngleRecall, Location2DError, Location2DRecall
from .DINOV2 import DinoV2MidEncoder

class OrienterNet(BaseModel):
    default_conf = {
        "image_encoder": "???",
        "map_encoder": "???",
        "air_net": "???",
        "DINOV2": "???",
        "latent_dim": "???",
        "matching_dim": "???",
        "scale_range": [0, 9],
        "num_scale_bins": "???",
        "z_min": None,
        "z_max": "???",
        "x_max": "???",
        "pixel_per_meter": "???",
        "num_rotations": "???",
        "add_temperature": False,
        "normalize_features": False,
        "padding_matching": "replicate",
        "apply_map_prior": True,
        "do_label_smoothing": False,
        "sigma_xy": 1,
        "sigma_r": 2,
        # depcreated
        "depth_parameterization": "scale",
        "norm_depth_scores": False,
        "normalize_scores_by_dim": False,
        "normalize_scores_by_num_valid": True,
        "prior_renorm": True,
        "retrieval_dim": None,
    }

    def _init(self, conf):
        assert not self.conf.norm_depth_scores
        assert self.conf.depth_parameterization == "scale"
        assert not self.conf.normalize_scores_by_dim
        assert self.conf.normalize_scores_by_num_valid
        assert self.conf.prior_renorm

        self.DINO_encoder = DinoV2MidEncoder()
        # Encoder = get_model(conf.image_encoder.get("name", "feature_extractor_v2"))
        # self.image_encoder = Encoder(conf.image_encoder.backbone)
        
        self.map_encoder = MapEncoder(conf.map_encoder)
        
      
        self.air_net = None if conf.air_net is None else AirNet(conf.air_net)

        self.template_sampler = TemplateSampler(
             conf.num_rotations
        )

        self.scale_classifier = torch.nn.Linear(conf.latent_dim, conf.num_scale_bins)
        
        if conf.air_net is None:
            self.feature_projection = torch.nn.Linear(
                conf.latent_dim, conf.matching_dim
            )
        if conf.add_temperature:
            temperature = torch.nn.Parameter(torch.tensor(0.0))
            self.register_parameter("temperature", temperature)


    def exhaustive_voting(self, f_air, f_map, confidence_air=None):
        if self.conf.normalize_features:
            f_air = normalize(f_air, dim=1)
            f_map = normalize(f_map, dim=1)

        # Build the templates and exhaustively match against the map.
        if confidence_air is not None:
            f_air = f_air * confidence_air.unsqueeze(1) 
      
        templates = self.template_sampler(f_air)       
        
        with torch.autocast("cuda", enabled=False):
            scores = conv2d_fft_batchwise(
                f_map.float(),
                templates.float(),
                padding_mode=self.conf.padding_matching,
            )
        
        if self.conf.add_temperature:
            scores = scores * torch.exp(self.temperature)
        return scores
            
    def _forward(self, data):
        pred = {}
        f_dino = self.DINO_encoder(data)
        # print('image encoder:',self.image_encoder)
        # print('f_dino shape:',f_dino.shape)

        # level = 0
        # f_image = self.image_encoder(data)["feature_maps"][level]

        pred_map = pred["map"] = self.map_encoder(data)
       
        f_map = pred_map["map_features"][0] 
    
        
        
        pred_air = {}
        if self.conf.air_net is None:
            # channel last -> classifier -> channel first
            f_air = self.feature_projection(f_air.moveaxis(1, -1)).moveaxis(-1, 1)
        else:
            pred_air = pred["air"] = self.air_net({"input": f_dino})
            
            f_air = pred_air["output"]

        # print('f_air shape:',f_air.shape)
        scores = self.exhaustive_voting(
            f_air, f_map, pred_air.get("confidence")
        )
        
        scores = scores.moveaxis(1, -1)
        
        if "log_prior" in pred_map and self.conf.apply_map_prior:
            scores = scores + pred_map["log_prior"][0].unsqueeze(-1)
        # pred["scores_unmasked"] = scores.clone()
        if "map_mask" in data:
            scores.masked_fill_(~data["map_mask"][..., None], -np.inf)
        # if "yaw_prior" in data:
        #     mask_yaw_prior(scores, data["yaw_prior"], self.conf.num_rotations)
      
        log_probs = log_softmax_spatial(scores)
    
        with torch.no_grad():
            uvr_max = argmax_xyr(scores).to(scores)
            uvr_avg = expectation_xyr(log_probs.exp())

       
        return {
            **pred,
            "scores": scores,
            "log_probs": log_probs,
            "uvr_max": uvr_max,
            "uv_max": uvr_max[..., :2],
            "yaw_max": uvr_max[..., 2],
            "uvr_expectation": uvr_avg,
            "uv_expectation": uvr_avg[..., :2],
            "yaw_expectation": uvr_avg[..., 2],
            "features_image": f_dino,
            "features_air": f_air,
           
            # "valid_air": valid_air.squeeze(1),
            # "f_polar":f_polar,
        }


    def loss(self, pred, data):
        xy_gt = data["uv"]
        yaw_gt = data["roll_pitch_yaw"][..., -1]
        # print('yaw_gt:',yaw_gt)
        # print('yaw_p:',pred["yaw_max"])
       
        if self.conf.do_label_smoothing:
            nll = nll_loss_xyr_smoothed(
                pred["log_probs"],
                xy_gt,
                yaw_gt,
                self.conf.sigma_xy / self.conf.pixel_per_meter,
                self.conf.sigma_r,
                mask=data.get("map_mask"),
            )
        else:
            nll = nll_loss_xyr(pred["log_probs"], xy_gt, yaw_gt)
        loss = {"total": nll, "nll": nll}
        if self.training and self.conf.add_temperature:
            loss["temperature"] = self.temperature.expand(len(nll))
        return loss

    def metrics(self):
        return {
            "xy_max_error": Location2DError("uv_max", self.conf.pixel_per_meter),
            "xy_expectation_error": Location2DError(
                "uv_expectation", self.conf.pixel_per_meter
            ),
            "yaw_max_error": AngleError("yaw_max"),
            "xy_recall_2m": Location2DRecall(2.0, self.conf.pixel_per_meter, "uv_max"),
            "xy_recall_5m": Location2DRecall(5.0, self.conf.pixel_per_meter, "uv_max"),
            "yaw_recall_2°": AngleRecall(2.0, "yaw_max"),
            "yaw_recall_5°": AngleRecall(5.0, "yaw_max"),
        }
