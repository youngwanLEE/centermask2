# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Sangrok Lee and Youngwan Lee (ETRI), 2020. All Rights Reserved.
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import List
import copy
import pycocotools.mask as mask_utils

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.structures.masks import PolygonMasks

ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def crop(polygons: List[List[np.ndarray]], boxes: torch.Tensor) -> "PolygonMasks":
    boxes = boxes.to(torch.device("cpu")).numpy()
    results = [
        _crop(polygon, box) for polygon, box in zip(polygons, boxes)
    ]

    return PolygonMasks(results)


def _crop(polygons: np.ndarray, box: np.ndarray) -> List[np.ndarray]:
    w, h = box[2] - box[0], box[3] - box[1]

    polygons = copy.deepcopy(polygons)
    for p in polygons:
        p[0::2] = p[0::2] - box[0]  # .clamp(min=0, max=w)
        p[1::2] = p[1::2] - box[1]  # .clamp(min=0, max=h)

    return polygons


def mask_rcnn_loss(pred_mask_logits, instances, maskiou_on):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    mask_ratios = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue

        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        if maskiou_on:
            cropped_mask = crop(instances_per_image.gt_masks.polygons, instances_per_image.proposal_boxes.tensor)
            cropped_mask = torch.tensor(
                [mask_utils.area(mask_utils.frPyObjects([p for p in obj], box[3]-box[1], box[2]-box[0])).sum().astype(float)
                for obj, box in zip(cropped_mask.polygons, instances_per_image.proposal_boxes.tensor)]
                )
                
            mask_ratios.append(
                (cropped_mask / instances_per_image.gt_masks.area())
                .to(device=pred_mask_logits.device).clamp(min=0., max=1.)
            )
        
        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    #gt_classes = cat(gt_classes, dim=0)

    if len(gt_masks) == 0:
        gt_classes = torch.LongTensor(gt_classes)
        if maskiou_on:
            selected_index = torch.arange(pred_mask_logits.shape[0], device=pred_mask_logits.device)
            if cls_agnostic_mask:
                selected_mask = pred_mask_logits[:, 0]
            else:
                # gt_classes = torch.LongTensor(gt_classes)
                selected_mask = pred_mask_logits[selected_index, gt_classes]
            mask_num, mask_h, mask_w = selected_mask.shape
            selected_mask = selected_mask.reshape(mask_num, 1, mask_h, mask_w)
            return pred_mask_logits.sum() * 0, selected_mask, gt_classes, None
        
        else:
            return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
        gt_classes = torch.zeros(total_num_masks, dtype=torch.int64)
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0) #ywlee
        pred_mask_logits = pred_mask_logits[indices, gt_classes] # (num_mask, Hmask, Wmask)

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)

    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="mean"
    )
    
    if maskiou_on:
        mask_ratios = cat(mask_ratios, dim=0)

        value_eps = 1e-10 * torch.ones(gt_masks.shape[0], device=gt_masks.device).double()
        mask_ratios = torch.max(mask_ratios, value_eps)

        pred_masks = pred_mask_logits > 0

        mask_targets_full_area = gt_masks.sum(dim=[1,2]) / mask_ratios

        mask_ovr_area = (pred_masks * gt_masks).sum(dim=[1,2]).float()
        mask_union_area = pred_masks.sum(dim=[1,2]) + mask_targets_full_area - mask_ovr_area
        value_1 = torch.ones(pred_masks.shape[0], device=gt_masks.device).double()
        value_0 = torch.zeros(pred_masks.shape[0], device=gt_masks.device)
        mask_union_area = torch.max(mask_union_area, value_1)
        mask_ovr_area = torch.max(mask_ovr_area, value_0)
        maskiou_targets = mask_ovr_area / mask_union_area
        mask_num, mask_h, mask_w = pred_mask_logits.shape
        selected_mask = pred_mask_logits.reshape(mask_num, 1, mask_h, mask_w)
        selected_mask = selected_mask.sigmoid()
        return mask_loss, selected_mask, gt_classes, maskiou_targets.detach()
    else:
        return mask_loss


def mask_rcnn_inference(pred_mask_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(MaskRCNNConvUpsampleHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on

        self.conv_norm_relus = []

        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = F.relu(self.deconv(x))
        return self.predictor(x)


def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)
