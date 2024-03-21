import torch
import torch.nn as nn


def focal_loss(input, target, alpha=0.25, gamma=2.):
    '''
    Args:
        input:  prediction, 'batch x c x h x w'
        target:  ground truth, 'batch x c x h x w'
        alpha: hyper param, default in 0.25
        gamma: hyper param, default in 2.0
    Reference: Focal Loss for Dense Object Detection, ICCV'17
    '''

    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    loss = 0

    pos_loss = torch.log(input) * torch.pow(1 - input, gamma) * pos_inds * alpha
    neg_loss = torch.log(1 - input) * torch.pow(input, gamma) * neg_inds * (1 - alpha)

    num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss.mean()


def focal_loss_cornernet(input, target, gamma=2.):
    '''
    Args:
        input:  prediction, 'batch x c x h x w'
        target:  ground truth, 'batch x c x h x w'
        gamma: hyper param, default in 2.0
    Reference: Cornernet: Detecting Objects as Paired Keypoints, ECCV'18
    '''
    eps = 1e-12
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, 4)

    loss = 0

    pos_loss = torch.log(input+eps) * torch.pow(1 - input, gamma) * pos_inds
    neg_loss = torch.log(1 - input+eps) * torch.pow(input, gamma) * neg_inds * neg_weights

    num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss.mean()


def focal_loss_cornernet_with_boundary(input, target, boundary, gamma=2.):
    '''
    Args:
        input:  prediction, 'batch x c x h x w'
        target:  ground truth, 'batch x c x h x w'
        gamma: hyper param, default in 2.0
    Reference: Cornernet: Detecting Objects as Paired Keypoints, ECCV'18
    '''
    total_num_pos = 0
    total_pos_loss = 0
    total_neg_loss = 0
    batch_size = boundary.shape[0]
    for bs in range(batch_size):
        x1, y1, x2, y2 = (boundary[bs]/4).int()
        bs_input = input[bs:bs+1, :, y1:y2+1, x1:x2+1]
        bs_target = target[bs:bs+1, :, y1:y2+1, x1:x2+1]
        eps = 1e-12
        pos_inds = bs_target.eq(1).float()
        neg_inds = bs_target.lt(1).float()

        neg_weights = torch.pow(1 - bs_target, 4)


        pos_loss = torch.log(bs_input+eps) * torch.pow(1 - bs_input, gamma) * pos_inds
        neg_loss = torch.log(1 - bs_input+eps) * torch.pow(bs_input, gamma) * neg_inds * neg_weights

        total_num_pos += pos_inds.float().sum()

        total_pos_loss += pos_loss.sum()
        total_neg_loss += neg_loss.sum()
    loss = 0
    if total_num_pos == 0:
        loss = loss - total_neg_loss
    else:
        loss = loss - (total_pos_loss + total_neg_loss) / total_num_pos

    return loss.mean()

