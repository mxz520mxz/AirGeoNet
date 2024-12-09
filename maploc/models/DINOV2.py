import torch
from torch import nn
from torch.nn import functional as F
import fast_pytorch_kmeans as fpk
from typing import Literal, Union, List

import cv2 as cv

from torchvision import transforms as tvf
from torchvision.transforms import functional as T


# Extract features from a Dino-v2 model
_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", \
                        "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]
class DinoV2ExtractFeatures:
    """
        Extract features from an intermediate layer in Dino-v2
    """
    def __init__(self, dino_model: _DINO_V2_MODELS, layer: int, 
                facet: _DINO_FACETS="token", use_cls=False, 
                norm_descs=True, device: str = "cpu") -> None:
        """
            Parameters:
            - dino_model:   The DINO-v2 model to use
            - layer:        The layer to extract features from
            - facet:    "query", "key", or "value" for the attention
                        facets. "token" for the output of the layer.
            - use_cls:  If True, the CLS token (first item) is also
                        included in the returned list of descriptors.
                        Otherwise, only patch descriptors are used.
            - norm_descs:   If True, the descriptors are normalized   
            - device:   PyTorch device to use
        """
        self.vit_type: str = dino_model
       
        # self.dino_model: nn.Module = torch.hub.load(
        #         'local dir', dino_model,source='local', force_reload=True)
        
        self.dino_model: nn.Module = torch.hub.load('facebookresearch/dinov2', dino_model)
        self.device = torch.device(device)
        self.dino_model = self.dino_model.eval().to(self.device)
        self.layer: int = layer
        self.facet = facet
        if self.facet == "token":
            self.fh_handle = self.dino_model.blocks[self.layer].\
                    register_forward_hook(
                            self._generate_forward_hook())
        else:

            self.fh_handle = self.dino_model.blocks[self.layer].\
                    attn.qkv.register_forward_hook(
                            self._generate_forward_hook())
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        # Hook data
        self._hook_out = None
    
    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output
        return _forward_hook
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
            Parameters:
            - img:   The input image
        """
        with torch.no_grad():
            # print('img shape:',img.shape)
            res = self.dino_model(img)
            # print('res shape:',res.shape)
            if self.use_cls:
                res = self._hook_out
            else:
                res = self._hook_out[:, 1:, ...]
            
            if self.facet in ["query", "key", "value"]:
                d_len = res.shape[2] // 3
                if self.facet == "query":
                    res = res[:, :, :d_len]
                elif self.facet == "key":
                    res = res[:, :, d_len:2*d_len]
                else:
                    res = res[:, :, 2*d_len:]
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None   # Reset the hook
        return res
    
    # def __del__(self):
    #     self.fh_handle.remove()




class DinoV2MidEncoder(nn.Module):
    """
        Extract features from an intermediate layer in Dino-v2
    """
    def __init__(self) -> None:
        super().__init__()
        # Dino_v2 properties (parameters)
        self.desc_layer: int = 31
        self.desc_facet: Literal["query", "key", "value", "token"] = "value"
        self.num_c: int = 8
        # Domain for use case (deployment environment)
        self.domain = "vpair"
        # Maximum image dimension
        self.max_img_size: int = 1024

        self.extractor = DinoV2ExtractFeatures("dinov2_vitg14", self.desc_layer,
                        self.desc_facet)
        self.device = self.extractor.device

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.register_buffer("mean_", torch.tensor(mean), persistent=False)
        self.register_buffer("std_", torch.tensor(std), persistent=False)


    def get_patch_descs(self,image):
        with torch.no_grad():
            # Read image
            
            # print('image shape:',image.shape)
            if max(image.shape[-2:]) > self.max_img_size:
                print(f"Image is too big!", end=' ')
                
                b, c, h, w = image.shape
                print(f"Resized from {(h, w) =}", end=' ')
                # Maintain aspect ratio
                if h == max(image.shape[-2:]):
                    w = int(w * self.max_img_size / h)
                    h = self.max_img_size
                else:
                    h = int(h * self.max_img_size / w)
                    w = self.max_img_size
                print(f"To {(h, w) =}")
                # some question??
                image = T.resize(image, (h, w), 
                        interpolation=T.InterpolationMode.BICUBIC)
                image = image.resize((w, h))  # This is cached later!
            # Make image patchable
            
            b, c, h, w = image.shape
            h_new, w_new = (h // 14) * 14, (w // 14) * 14
            image = tvf.CenterCrop((h_new, w_new))(image)
            # print('image.shape:',image.shape)
            # Extract descriptors
            ret = self.extractor(image)
            
        return ret
    

    def feature_assignments(self, desc_vects, shape):
        b, c, h, w = shape
       
        h_p, w_p = (h // 14), (w // 14)
        h_new, w_new = h_p * 14, w_p * 14
        # print(h_p,w_p)

        
        assert h_p * w_p == desc_vects.shape[1], "Descriptor vector shape is not correct"
        # Descriptor assignments
        da = desc_vects.reshape(desc_vects.shape[0],h_p, w_p,desc_vects.shape[-1]).permute(0,3,1,2)
        # print(da.shape)
        da_assign = F.interpolate(da.to(float), 
                size = (h_new, w_new), mode='bilinear').to(da.dtype)
        # print(f"\tShapes: residual: {desc_vects.shape}, assignment: {da.shape}")

        return da_assign, da

    def forward(self,data):
        image = data["image"]
        valid = data['valid']
        
        # print('dino image shape',image.shape)
        # print('dino valid shape',valid.shape)
        image = (image - self.mean_[:, None, None]) / self.std_[:, None, None]

        feature_ret = self.get_patch_descs(image)

        feature, da = self.feature_assignments(feature_ret,image.shape)

        valid_float = valid.float()
        _, _, target_height, target_width = feature.shape
        valid_resized = F.interpolate(valid_float.unsqueeze(0), size=(target_height, target_width), mode='bilinear', align_corners=False)
        valid = valid_resized.squeeze(0).round().bool()
        feature = feature.masked_fill(~valid.unsqueeze(0), 0)
       
        return feature



    

        