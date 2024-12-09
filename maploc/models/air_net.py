import torch.nn as nn
from torchvision.models.resnet import Bottleneck

from .base import BaseModel
from .feature_extractor import AdaptationBlock,AdaptationBlockOutput
from .utils import checkpointed


class AirNet(BaseModel):
    default_conf = {
        "pretrained": True,
        "num_blocks": "???",
        "latent_dim": "???",
        "input_dim": "${.latent_dim}",
        "output_dim": "${.latent_dim}",
        "confidence": False,
        "norm_layer": "nn.BatchNorm2d",  # normalization ind decoder blocks
        "checkpointed": False,  # whether to use gradient checkpointing
        "padding": "zeros",
    }

    def _init(self, conf):
        # blocks_1 = []
        # blocks_2 = []
        blocks = []
        Block = checkpointed(Bottleneck, do=conf.checkpointed)
        for i in range(conf.num_blocks):
            dim = conf.input_dim if i == 0 else conf.latent_dim
        #     if i < conf.num_blocks/2:
        #         blocks_1.append(
        #             Block(
        #                 dim,
        #                 conf.latent_dim // Bottleneck.expansion,
        #                 norm_layer=eval(conf.norm_layer),
        #             )
        #         )
        #     else:
        #         blocks_2.append(
        #             Block(
        #                 dim,
        #                 conf.latent_dim // Bottleneck.expansion,
        #                 norm_layer=eval(conf.norm_layer),
        #             )
        #         )
        # self.blocks_1 = nn.Sequential(*blocks_1)
        # self.blocks_2 = nn.Sequential(*blocks_2)
            blocks.append(
                    Block(
                        dim,
                        conf.latent_dim // Bottleneck.expansion,
                        norm_layer=eval(conf.norm_layer),
                    )
                )
        self.blocks = nn.Sequential(*blocks)
        self.output_layer = AdaptationBlockOutput(conf.latent_dim, conf.output_dim)
        if conf.confidence:
            self.confidence_layer = AdaptationBlockOutput(conf.latent_dim, 1)

        def update_padding(module):
            if isinstance(module, nn.Conv2d):
                module.padding_mode = conf.padding

        if conf.padding != "zeros":
            self.bocks.apply(update_padding)


        self.upsample_layer = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)        

    def _forward(self, data):
        # features_mid = self.blocks_1.to('cuda:3')(data["input"].to('cuda:3'))
        # features = self.blocks_2.to('cuda:2')(features_mid.to('cuda:2'))
        features = self.blocks(data["input"])
        # features = features.to('cuda:0')
        pred = {
            "output": self.output_layer(features),
        }
        
        if self.conf.confidence:
            pred["confidence"] = self.confidence_layer(features).squeeze(1).sigmoid()
        
        return pred

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError