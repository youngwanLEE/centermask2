# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Youngwan Lee (ETRI) in 28/01/2020.
import math
import torch

from detectron2.structures import Instances


def add_ground_truth_to_proposals(targets, proposals):
    """
    Call `add_ground_truth_to_proposals_single_image` for all images.

    Args:
        targets(list[Instances]): list of N elements. Element i is a Boxes
            representing the gound-truth for image i.
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.

    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
    """
    assert targets is not None

    assert len(proposals) == len(targets)
    if len(proposals) == 0:
        return proposals

    return [
        add_ground_truth_to_proposals_single_image(tagets_i, proposals_i)
        for tagets_i, proposals_i in zip(targets, proposals)
    ]


def add_ground_truth_to_proposals_single_image(targets_i, proposals):
    """
    Augment `proposals` with ground-truth boxes from `gt_boxes`.

    Args:
        Same as `add_ground_truth_to_proposals`, but with targets and proposals
        per image.

    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    device = proposals.scores.device
    proposals.proposal_boxes = proposals.pred_boxes
    proposals.remove("pred_boxes")
    # Concatenating gt_boxes with proposals requires them to have the same fields
    # Assign all ground-truth boxes an objectness logit corresponding to P(object) \approx 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(targets_i), device=device)
    gt_proposal = Instances(proposals.image_size)
    gt_proposal.proposal_boxes = targets_i.gt_boxes
    # to have the same fields with proposals
    gt_proposal.scores = gt_logits
    gt_proposal.pred_classes = targets_i.gt_classes
    gt_proposal.locations = torch.ones((len(targets_i), 2), device=device)

    new_proposals = Instances.cat([proposals, gt_proposal])

    return new_proposals
