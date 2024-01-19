import numpy as np
import torch
import torch.nn as nn
from lib.datasets.utils import class2angle
from scipy.optimize import minimize_scalar
import numba as nb
import cv2 as cv

#TODO 可以修改interval的超参数
@nb.jit(parallel = True)
def func(x, interval=0.1, mu=0, b=1):
    right = 0.5 * (1+np.sign(x+interval-mu)*(1-np.exp(-np.abs(x+interval-mu)/b)))
    left = 0.5 * (1+np.sign(x-interval-mu)*(1-np.exp(-np.abs(x-interval-mu)/b)))
    return left - right

@nb.jit(parallel = True)
def fun(x, parameters):
    res = 0
    for i in range(len(parameters[0])):
        res += func(x, interval = parameters[2][i], mu = parameters[0][i], b = parameters[1][i])
    return res

@nb.jit(parallel = True)
def calculate_depth_and_conf(batch_size, K, laplace_b, laplace_miu, lower_bound, upper_bound, interval):
    merge_depth = np.zeros([batch_size, K, 1])
    merge_conf = np.zeros([batch_size, K, 1])
    for i in range(batch_size):
        for j in range(K):
            parameters = np.zeros((3, laplace_b.shape[2]))
            parameters[0] = laplace_miu[i][j]
            parameters[1] = laplace_b[i][j]
            parameters[2] = interval
            result = minimize_scalar(fun, bracket=(lower_bound, upper_bound), method = 'golden', args = (parameters, ))
            merge_depth[i][j] = result['x']
            merge_conf[i][j] = -result['fun']/49
    return merge_depth, merge_conf

def decode_detections(dets, info, calibs, cls_mean_size, threshold, problist=None):
    '''
    NOTE: THIS IS A NUMPY FUNCTION
    input: dets, numpy array, shape in [batch x max_dets x dim]
    input: img_info, dict, necessary information of input images
    input: calibs, corresponding calibs for the input batch
    output:
    '''

    results = {}
    for i in range(dets.shape[0]):  # batch
        preds = []
        for j in range(dets.shape[1]):  # max_dets
            cls_id = int(dets[i, j, 0])
            score = dets[i, j, 1]
            if score < threshold: continue

            # 2d bboxs decoding
            x = dets[i, j, 2] * info['bbox_downsample_ratio'][i][0]
            y = dets[i, j, 3] * info['bbox_downsample_ratio'][i][1]
            w = dets[i, j, 4] * info['bbox_downsample_ratio'][i][0]
            h = dets[i, j, 5] * info['bbox_downsample_ratio'][i][1]
            bbox = [x-w/2, y-h/2, x+w/2, y+h/2]

            depth = dets[i, j, -2]
            score *= dets[i, j, -1]

            # heading angle decoding
            alpha = get_heading_angle(dets[i, j, 6:30])
            ry = calibs[i].alpha2ry(alpha, x)

            # dimensions decoding
            dimensions = dets[i, j, 30:33]
            dimensions += cls_mean_size[int(cls_id)]
            if True in (dimensions<0.0): continue

            # positions decoding
            x3d = dets[i, j, 33] * info['bbox_downsample_ratio'][i][0]
            y3d = dets[i, j, 34] * info['bbox_downsample_ratio'][i][1]
            locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            locations[1] += dimensions[0] / 2

            preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])
        results[info['img_id'][i]] = preds
    return results


# def convert_color_map(img, mode=cv.INTER_LINEAR, size=49):
#     # temp = img / np.max(img) * 255
#     # temp = img / 50 * 255
#     temp = cv.resize(img.cpu().numpy(), size, interpolation=mode) * 250
#     temp = temp.astype(np.uint8)
#     im_color = cv.applyColorMap(temp, cv.COLORMAP_JET)
#     return im_color

#two stage style
def extract_dets_from_outputs(outputs, conf_mode='ada', K=50, lower_bound=None, upper_bound = None, interval=None):
    # get src outputs
    heatmap = outputs['heatmap']
    size_2d = outputs['size_2d']
    offset_2d = outputs['offset_2d']

    batch, channel, height, width = heatmap.size() # get shape
    heading = outputs['heading'].view(batch,K,-1)

    vis_depth = outputs['vis_depth'].view(batch,K,7,7)
    att_depth = outputs['att_depth'].view(batch,K,7,7)
    ins_depth = vis_depth + att_depth

    ins_depth_uncer = outputs['ins_depth_uncer'].view(batch,K,7,7)
    laplace_b  = (ins_depth_uncer.exp()/1.414).view(batch, K, -1).detach().cpu().numpy()
    laplace_miu = (ins_depth).view(batch, K, -1).detach().cpu().numpy()
    merge_depth, merge_conf = calculate_depth_and_conf(batch, K, laplace_b, laplace_miu, lower_bound, upper_bound, interval)
    merge_depth = torch.from_numpy(merge_depth).to(ins_depth.device)
    merge_conf = torch.from_numpy(merge_conf).to(ins_depth.device)
    
    # merge_depth = torch.zeros(batch, K, 1, device = ins_depth.device)
    # merge_conf = torch.zeros(batch, K, 1, device = ins_depth.device)
    # for i in range(ins_depth_uncer.shape[0]):
    #     for j in range(ins_depth_uncer.shape[1]):
    #         parameters = [[], []]
    #         parameters[0] = laplace_miu[i][j].reshape(-1)
    #         parameters[1] = laplace_b[i][j].reshape(-1)
    #         partial_func = partial(fun, parameters = parameters)
    #         result = minimize_scalar(partial_func, bounds=(0, 70), method='bounded')
    #         merge_depth[i][j] = result['x']
    #         merge_conf[i][j] = -result['fun']/49
    
    # merge_prob = (-(0.5 * ins_depth_uncer).exp()).exp()
    # merge_depth = (torch.sum((ins_depth*merge_prob).view(batch,K,-1), dim=-1) /
    #                torch.sum(merge_prob.view(batch,K,-1), dim=-1))
    # merge_depth = merge_depth.unsqueeze(2)

    # if conf_mode == 'ada':
    #     merge_conf = (torch.sum(merge_prob.view(batch,K,-1)**2, dim=-1) / \
    #                   torch.sum(merge_prob.view(batch,K,-1), dim=-1)).unsqueeze(2)
    # elif conf_mode == 'max':
    #     merge_conf = (merge_prob.view(batch, K, -1).max(-1))[0].unsqueeze(2)
    # else:
    #     raise NotImplementedError("%s confidence aggreation is not supported" % conf_mode)

    size_3d = outputs['size_3d'].view(batch,K,-1)
    offset_3d = outputs['offset_3d'].view(batch,K,-1)

    heatmap = torch.clamp(heatmap.sigmoid_(), min=1e-4, max=1 - 1e-4)
    # perform nms on heatmaps
    heatmap = _nms(heatmap)
    scores, inds, cls_ids, xs, ys = _topk(heatmap, K=K)

    offset_2d = _transpose_and_gather_feat(offset_2d, inds)
    offset_2d = offset_2d.view(batch, K, 2)
    xs2d = xs.view(batch, K, 1) + offset_2d[:, :, 0:1]
    ys2d = ys.view(batch, K, 1) + offset_2d[:, :, 1:2]

    xs3d = xs.view(batch, K, 1) + offset_3d[:, :, 0:1]
    ys3d = ys.view(batch, K, 1) + offset_3d[:, :, 1:2]

    cls_ids = cls_ids.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    # check shape
    xs2d = xs2d.view(batch, K, 1)
    ys2d = ys2d.view(batch, K, 1)
    xs3d = xs3d.view(batch, K, 1)
    ys3d = ys3d.view(batch, K, 1)

    size_2d = _transpose_and_gather_feat(size_2d, inds)
    size_2d = size_2d.view(batch, K, 2)

    detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, heading, size_3d, xs3d, ys3d, merge_depth, merge_conf], dim=2)

    return detections

def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    if torch.__version__ == '1.6.0':
        topk_ys = (topk_inds // width).int().float()
    else:
        topk_ys = (topk_inds / width).int().float()
    # topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    if torch.__version__ == '1.6.0':
        topk_cls_ids = (topk_ind // K).int()
    else:
        topk_cls_ids = (topk_ind / K).int()
    # topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim  = feat.size(2)  # get channel dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()   # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))   # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)     # B * len(ind) * C
    return feat


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)



if __name__ == '__main__':
    ## testing
    from lib.datasets.kitti import KITTI
    from torch.utils.data import DataLoader

    dataset = KITTI('../../data', 'train')
    dataloader = DataLoader(dataset=dataset, batch_size=2)
