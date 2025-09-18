"""
Version2: includes centroids into model, and shares embedding layers.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones.cifar as models
from Distance import Distance


class DFPNet(nn.Module):
    def __init__(self, backbone='ResNet18', num_classes=10, backbone_fc=False,
                 embed_dim=None, distance='cosine', scaled=True, thresholds=None):
        super(DFPNet, self).__init__()
        assert backbone_fc == False  # drop out the classifier layer.
        # num_classes = num_classes would be useless if backbone_fc=False
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.backbone = models.__dict__[backbone](num_classes=num_classes, backbone_fc=backbone_fc)
        self.feat_dim = self.get_backbone_last_layer_out_channel()  # get the channel number of backbone output
        if embed_dim:
            self.embeddingLayer = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim, embed_dim)
            )
            self.feat_dim = embed_dim
        self.classifier = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim, num_classes)
            )
        # We add 1 centroid for the unknown class, which is like a placeholder.
        self.centroids = nn.Parameter(torch.randn(num_classes + 1, self.num_classes))
        self.distance = distance
        assert self.distance in ['l1', 'l2', 'cosine']
        self.scaled = scaled
        self.register_buffer("thresholds", thresholds)

    def get_backbone_last_layer_out_channel(self):
        if self.backbone_name == "LeNetPlus":
            return 128 * 3 * 3
        last_layer = list(self.backbone.children())[-1]
        while (not isinstance(last_layer, nn.Conv2d)) and \
                (not isinstance(last_layer, nn.Linear)) and \
                (not isinstance(last_layer, nn.BatchNorm2d)):

            temp_layer = list(last_layer.children())[-1]
            if isinstance(temp_layer, nn.Sequential) and len(list(temp_layer.children())) == 0:
                temp_layer = list(last_layer.children())[-2]
            last_layer = temp_layer
        if isinstance(last_layer, nn.BatchNorm2d):
            return last_layer.num_features
        else:
            return last_layer.out_channels

    def forward(self, x):
        dist_gen2cen = None

        x = self.backbone(x)
        gap = (F.adaptive_avg_pool2d(x, 1)).view(x.size(0), -1)

        # if includes embedding layer.
        embed_fea = self.embeddingLayer(gap) if hasattr(self, 'embeddingLayer') else gap
        logits = self.classifier(embed_fea)
        # calculate distance.
        DIST = Distance(scaled=self.scaled)
        normalized_centroids = F.normalize(self.centroids, dim=1, p=2)
        normalized_logits = F.normalize(logits, dim=1, p=2)
        dist_fea2cen = getattr(DIST, self.distance)(normalized_logits, normalized_centroids)  # [n,c+1]
        dist_cen2cen = DIST.l2(normalized_centroids, normalized_centroids)  # [c+1,c+1]

        return {
            "backbone_fea": x,
            "logits": logits,
            "embed_fea": embed_fea,
            "dist_fea2cen": dist_fea2cen,
            "dist_cen2cen": dist_cen2cen,
            "dist_gen2cen": dist_gen2cen
        }


def demo():
    x = torch.rand([10, 3, 32, 32])
    net = DFPNet('ResNet18', num_classes=10, embed_dim=64, thresholds=torch.rand(11))
    output = net(x)
    print(output["logits"].shape)
    print(output["embed_fea"].shape)
    print(output["dist_fea2cen"].shape)
    print(output["dist_cen2cen"].shape)

    # x = torch.rand([1, 1, 28, 28])
    # net = DFPNet('LeNetPlus', num_classes=10, embed_dim=64)
    # output = net(x)
    # print(output["logits"].shape)
    # print(output["embed_fea"].shape)
    # print(output["dist_fea2cen"].shape)

# demo()
