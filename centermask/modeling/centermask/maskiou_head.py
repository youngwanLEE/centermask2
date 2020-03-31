# Copyright (c) Sangrok Lee and Youngwan Lee (ETRI) All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, cat
from detectron2.utils.registry import Registry
from centermask.layers import MaxPool2d, Linear

ROI_MASKIOU_HEAD_REGISTRY = Registry("ROI_MASKIOU_HEAD")
ROI_MASKIOU_HEAD_REGISTRY.__doc__ = """
Registry for maskiou heads, which predicts predicted mask iou.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def mask_iou_loss(labels, pred_maskiou, gt_maskiou, loss_weight):
    """
    Compute the maskiou loss.

    Args:
        labels (Tensor): Given mask labels (num of instance,)
        pred_maskiou (Tensor):  A tensor of shape (num of instance, C)
        gt_maskiou (Tensor): Ground Truth IOU generated in mask head (num of instance,)
    """
    def l2_loss(input, target):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        pos_inds = torch.nonzero(target > 0.0).squeeze(1)
        if pos_inds.shape[0] > 0:
            cond = torch.abs(input[pos_inds] - target[pos_inds])
            loss = 0.5 * cond**2 / pos_inds.shape[0]
        else:
            loss = input * 0.0
        return loss.sum()

    if labels.numel() == 0:
        return pred_maskiou.sum() * 0
    
    index = torch.arange(pred_maskiou.shape[0]).to(device=pred_maskiou.device)
    maskiou_loss = l2_loss(pred_maskiou[index, labels], gt_maskiou)
    maskiou_loss = loss_weight * maskiou_loss
    
    return maskiou_loss


def mask_iou_inference(pred_instances, pred_maskiou):
    labels = cat([i.pred_classes for i in pred_instances])
    num_masks = pred_maskiou.shape[0]
    index = torch.arange(num_masks, device=labels.device)
    num_boxes_per_image = [len(i) for i in pred_instances]
    maskious = pred_maskiou[index, labels].split(num_boxes_per_image, dim=0)
    for maskiou, box in zip(maskious, pred_instances):
        box.mask_scores = box.scores * maskiou


@ROI_MASKIOU_HEAD_REGISTRY.register()
class MaskIoUHead(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(MaskIoUHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASKIOU_HEAD.CONV_DIM
        num_conv          = cfg.MODEL.ROI_MASKIOU_HEAD.NUM_CONV
        input_channels    = input_shape.channels + 1
        resolution        = input_shape.width // 2
        # fmt: on

        self.conv_relus = []
        stride = 1
        for k in range(num_conv):
            if (k+1) == num_conv:
                stride = 2
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=stride,
                padding=1,
                activation=F.relu
            )
            self.add_module("maskiou_fcn{}".format(k+1), conv)
            self.conv_relus.append(conv)
        self.maskiou_fc1 = Linear(conv_dims*resolution**2, 1024)
        self.maskiou_fc2 = Linear(1024, 1024)
        self.maskiou = Linear(1024, num_classes)
        self.pooling = MaxPool2d(kernel_size=2, stride=2)


        for l in self.conv_relus:
            nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)
        for l in [self.maskiou_fc1, self.maskiou_fc2]:
            nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)


        nn.init.normal_(self.maskiou.weight, mean=0, std=0.01)
        nn.init.constant_(self.maskiou.bias, 0)

    def forward(self, x, mask):
        mask_pool = self.pooling(mask)
        x = torch.cat((x, mask_pool), 1)

        for layer in self.conv_relus:
            x = layer(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.maskiou_fc1(x))
        x = F.relu(self.maskiou_fc2(x))
        x = self.maskiou(x)
        return x


def build_maskiou_head(cfg, input_shape):
    """
    Build a mask iou head defined by `cfg.MODEL.ROI_MASKIOU_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASKIOU_HEAD.NAME
    return ROI_MASKIOU_HEAD_REGISTRY.get(name)(cfg, input_shape)