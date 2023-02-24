from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2

import sys
import os.path as osp
cur_dir = osp.dirname(os.path.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, '..', 'data'))

from lib.opts import opts
from lib.models.model import create_model, load_model
from lib.utils.image import get_affine_transform, transform_preds
from lib.utils.gpfit import fitgaussian
from lib.utils.pnp.cuboid_pnp_shell import pnp_shell
from lib.utils.debugger import Debugger
from lib.utils.pnp.cuboid_objectron import Cuboid3d
from objectron.dataset.iou import IoU
from objectron.dataset.box import Box

import copy
import numpy as np
import torch
import torch.nn as nn
import json
from scipy.spatial.transform import Rotation as R

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

SHANGRU_DATASET = False

image_ext = ['jpg', 'jpeg', 'png', 'webp']

def soft_nms_nvidia(src_boxes, sigma=0.5, Nt=0.3, threshold=0.001, method=0):
    N = src_boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0

    for i in range(N):
        maxscore = src_boxes[i]['score']
        maxpos = i

        tx1 = src_boxes[i]['bbox'][0]
        ty1 = src_boxes[i]['bbox'][1]
        tx2 = src_boxes[i]['bbox'][2]
        ty2 = src_boxes[i]['bbox'][3]
        ts = src_boxes[i]['score']

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < src_boxes[pos]['score']:
                maxscore = src_boxes[pos]['score']
                maxpos = pos
            pos = pos + 1

        # add max box as a detection

        src_boxes[i]['bbox'] = src_boxes[maxpos]['bbox']
        src_boxes[i]['score'] = src_boxes[maxpos]['score']

        # swap ith box with position of max box
        src_boxes[maxpos]['bbox'] = [tx1, ty1, tx2, ty2]
        src_boxes[maxpos]['score'] = ts

        for key in src_boxes[0]:
            if key is not 'bbox' and key is not 'score':
                tmp = src_boxes[i][key]
                src_boxes[i][key] = src_boxes[maxpos][key]
                src_boxes[maxpos][key] = tmp

        tx1 = src_boxes[i]['bbox'][0]
        ty1 = src_boxes[i]['bbox'][1]
        tx2 = src_boxes[i]['bbox'][2]
        ty2 = src_boxes[i]['bbox'][3]
        ts = src_boxes[i]['score']

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:

            x1 = src_boxes[pos]['bbox'][0]
            y1 = src_boxes[pos]['bbox'][1]
            x2 = src_boxes[pos]['bbox'][2]
            y2 = src_boxes[pos]['bbox'][3]
            s = src_boxes[pos]['score']

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box

                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov) / sigma)
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    src_boxes[pos]['score'] = weight * src_boxes[pos]['score']

                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if src_boxes[pos]['score'] < threshold:

                        src_boxes[pos]['bbox'] = src_boxes[N - 1]['bbox']
                        src_boxes[pos]['score'] = src_boxes[N - 1]['score']

                        for key in src_boxes[0]:
                            if key is not 'bbox' and key is not 'score':
                                tmp = src_boxes[pos][key]
                                src_boxes[pos][key] = src_boxes[N - 1][key]
                                src_boxes[N - 1][key] = tmp

                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep

def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _gather_feat(feat, ind, mask=None):
    if len(ind.size()) > 2:

        num_symmetry = ind.size(1)
        dim = feat.size(2)
        ind = ind.unsqueeze(3).expand(ind.size(0), ind.size(1), ind.size(2),
                                      dim)  # batch x num_symmetry x max_object x (num_joint x 2)

        ind = ind.view(ind.size(0), -1, ind.size(3))  # batch x (num_symmetry x max_object) x (num_joint x 2)

        feat = feat.gather(1, ind)
        feat = feat.view(ind.size(0), num_symmetry, -1,
                         ind.size(2))  # batch x num_symmetry x max_object x (num_joint x 2)
        if mask is not None:
            mask = mask.unsqueeze(3).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
    else:
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)

        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
    return feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def _center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

class SimpDetector(object):
    def __init__(self, opt) -> None:
        opt.device = torch.device('cuda')
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv, opt)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True

        self.pre_images = None

    def merge_outputs(self, detections):

        # Todo: We use 0 here, since we only work on a single category, need to be updated
        # Group all the detection result from different scales on a single image (We only deal with one iamge input one time)
        results = []
        for det in detections[0]:
            if det['score'] > self.opt.vis_thresh:
                results.append(det)
        results = np.array(results)
        if self.opt.nms or len(self.opt.test_scales) > 1:
            keep = soft_nms_nvidia(results, Nt=0.5, method=2, threshold=self.opt.vis_thresh)
            results = results[keep]

        return results

    def pre_process(self, image, scale, input_meta={}):
        '''
              Prepare input image in different testing modes.
                Currently support: fix short size/ center crop to a fixed size/
                keep original resolution but pad to a multiplication of 32
        '''
        # Center Crop
        # height, width = image.shape[0:2]
        # resize_min = min(height, width)
        # image = _center_crop(image, (resize_min, resize_min))

        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        inp_height, inp_width = self.opt.input_h, self.opt.input_w
        c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])

        out_height = inp_height // self.opt.down_ratio
        out_width = inp_width // self.opt.down_ratio
        trans_output = get_affine_transform(c, s, 0, [out_width, out_height])

        resized_image = cv2.resize(image, (new_width, new_height))
        
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)

        images = torch.from_numpy(images)
        meta = {'c': c, 's': s, 'height': height, 'width': width,
                'out_height': out_height, 'out_width': out_width,
                'inp_height': inp_height, 'inp_width': inp_width,
                'trans_input': trans_input, 'trans_output': trans_output}

        if 'camera_matrix' in input_meta:
            meta['camera_matrix'] = input_meta['camera_matrix']
        
        return images, meta

    def object_pose_decode(self, heat, kps, wh=None, kps_displacement_std=None, obj_scale=None, reg=None, hm_hp=None,
                                hp_offset=None, opt=None, Inference=False):
        K = opt.K
        rep_mode = opt.rep_mode
        batch, cat, height, width = heat.size()
        num_joints = kps.shape[1] // 2

        # perform nms on heatmaps
        heat = _nms(heat)
        scores, inds, clses, ys, xs = _topk(heat, K=K)  # inds: index in a single heatmap
        kps = _transpose_and_gather_feat(kps, inds)  # 100*34
        kps = kps.view(batch, K, num_joints * 2)  # joint offset from the centroid loc
        kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)  # + centroid loc
        kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
        
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

        clses = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)

        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)

        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)

        if Inference == True:
            # Save a copy for future use
            hm_hp_copy = hm_hp.clone()                    

        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.view(batch, K, num_joints, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2

        mask_temp = torch.ones((batch, num_joints, K, 1)).to(kps.device)
        mask_temp = (mask_temp > 0).float().expand(batch, num_joints, K, 2)
        kps_displacement_mean = mask_temp * kps
        kps_displacement_mean = kps_displacement_mean.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_joints * 2)

        # Continue normal processing
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)  # b x J x K x K x 2
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
        
        hp_offset = _transpose_and_gather_feat(
                hp_offset, hm_inds.view(batch, -1))
        hp_offset = hp_offset.view(batch, num_joints, K, 2)
        hm_xs = hm_xs + hp_offset[:, :, :, 0]
        hm_ys = hm_ys + hp_offset[:, :, :, 1]

        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score  # -1 or hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys  # -10000 or hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs

        # Find the nearest keypoint in the corresponding heatmap for each displacement representation
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_joints, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)  # b x J x K x K
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
            batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)

        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
                (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
                (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_joints, K, 2)

        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
                batch, K, num_joints * 2)

        if Inference == True:
            # Have to satisfy all the requirements: within an enlarged 2D bbox/
            # hm_score high enough/center_score high enough/not far away from the corresponding representation
            scores_copy = scores.unsqueeze(1).expand(batch, num_joints, K, 2)

            mask_2 = (hm_kps[..., 0:1] > 0.8 * l) + (hm_kps[..., 0:1] < 1.2 * r) + \
                        (hm_kps[..., 1:2] > 0.8 * t) + (hm_kps[..., 1:2] < 1.2 * b) + \
                        (hm_score > thresh) + (min_dist < (torch.max(b - t, r - l) * 0.5)) + \
                        (scores_copy > thresh)

            mask_2 = (mask_2 == 7).float().expand(batch, num_joints, K, 2)
            hm_kps_filtered = mask_2 * hm_kps + (1 - mask_2) * -10000

            hm_xs_filtered = hm_kps_filtered[:, :, :, 0].detach().cpu().numpy()
            hm_ys_filtered = hm_kps_filtered[:, :, :, 1].detach().cpu().numpy()

            # Fit a 2D gaussian distribution on the heatmap
            # Save a copy for further processing
            kps_heatmap_mean = torch.ones([batch, K, num_joints * 2], dtype=torch.float32) * -10000
            kps_heatmap_std = torch.ones([batch, K, num_joints * 2], dtype=torch.float32) * -10000
            kps_heatmap_height = torch.ones([batch, K, num_joints], dtype=torch.float32) * -10000

            # Need optimization
            for idx_batch in range(batch):
                for idx_joint in range(num_joints):
                    data = hm_hp_copy[idx_batch][idx_joint].detach().cpu().numpy()
                    for idx_K in range(K):
                        if hm_xs_filtered[idx_batch][idx_joint][idx_K] == -10000 or \
                                hm_ys_filtered[idx_batch][idx_joint][idx_K] == -10000:
                            continue
                        else:

                            win = 11
                            ran = win // 2

                            # For the tracking task, both rep_mode 1 and 2 needs this step
                            if opt.tracking_task or opt.refined_Kalman or rep_mode == 2:

                                data_enlarged = np.zeros((data.shape[0] + 2 * ran, data.shape[1] + 2 * ran))
                                data_enlarged[ran:data.shape[0] + ran, ran:data.shape[1] + ran] = data
                                weights = data_enlarged[int(hm_ys_filtered[idx_batch][idx_joint][idx_K]):
                                                        int(hm_ys_filtered[idx_batch][idx_joint][
                                                                idx_K] + 2 * ran + 1),
                                            int(hm_xs_filtered[idx_batch][idx_joint][idx_K]):
                                            int(hm_xs_filtered[idx_batch][idx_joint][idx_K] + 2 * ran + 1)
                                            ]

                                params = fitgaussian(weights)

                                # mu will be slightly different from ran
                                height, mu_x, mu_y, std_x, std_y = params
                            elif rep_mode == 1:

                                # For fair comparison, do not use fitted gaussian for correction
                                mu_x = ran
                                mu_y = ran
                                height = data[int(hm_ys_filtered[idx_batch][idx_joint][idx_K]),
                                                int(hm_xs_filtered[idx_batch][idx_joint][idx_K])]
                                std_x = 1  # Just used as a mark
                                std_y = 1  # Just used as a mark

                            kps_heatmap_mean[idx_batch][idx_K][idx_joint * 2:idx_joint * 2 + 2] = \
                                torch.FloatTensor([hm_xs_filtered[idx_batch][idx_joint][idx_K] + mu_x - ran,
                                                    hm_ys_filtered[idx_batch][idx_joint][idx_K] + mu_y - ran])
                            kps_heatmap_std[idx_batch][idx_K][idx_joint * 2:idx_joint * 2 + 2] = \
                                torch.FloatTensor([std_x, std_y])
                            kps_heatmap_height[idx_batch][idx_K][idx_joint] = torch.from_numpy(np.array(height))

        kps_heatmap_mean = kps_heatmap_mean.to(kps_displacement_mean.device)
        kps_heatmap_std = kps_heatmap_std.to(kps_displacement_mean.device)
        kps_heatmap_height = kps_heatmap_height.to(kps_displacement_mean.device)


        if kps_displacement_std is not None:
            kps_displacement_std = _transpose_and_gather_feat(kps_displacement_std, inds)

            # Since we pred log(var) while saving std, we need to convert it first
            kps_displacement_std = torch.sqrt(torch.exp(kps_displacement_std))
            kps_displacement_std = kps_displacement_std * opt.balance_coefficient[opt.c]
            kps_displacement_std = kps_displacement_std.view(batch, K, num_joints * 2)  # joint offset from the centroid loc
        else:
            kps_displacement_std = torch.zeros([batch, K, num_joints * 2], dtype=torch.float32)
            kps_displacement_std = kps_displacement_std.to(scores.device)
        
        # object scale
        obj_scale = _transpose_and_gather_feat(obj_scale, inds)
        obj_scale = obj_scale.view(batch, K, 3)

        obj_scale_uncertainty = torch.zeros([batch, K, 3], dtype=torch.float32)
        obj_scale_uncertainty = obj_scale_uncertainty.to(scores.device)

        detections = {'bboxes': bboxes,
                      'scores': scores,
                      'kps': kps,
                      'clses': clses,
                      'obj_scale': obj_scale,
                      'obj_scale_uncertainty': obj_scale_uncertainty,
                      'kps_displacement_mean': kps_displacement_mean,
                      'kps_displacement_std': kps_displacement_std,
                      'kps_heatmap_mean': kps_heatmap_mean,
                      'kps_heatmap_std': kps_heatmap_std,
                      'kps_heatmap_height': kps_heatmap_height,
                      }
        return detections

    def run(self, images):

        with torch.no_grad():

            torch.cuda.synchronize()
            output = self.model(images)[-1]

            output['hm'] = output['hm'].sigmoid_()
            output['hm_hp'] = output['hm_hp'].sigmoid_()

            print(output.keys())
            wh = output['wh']
            reg = output['reg']
            hm_hp = output['hm_hp']
            hp_offset = output['hp_offset']
            obj_scale = output['scale']
            hps_uncertainty = None
            
            torch.cuda.synchronize()

            dets = self.object_pose_decode(
                output['hm'], output['hps'], wh=wh, kps_displacement_std=hps_uncertainty, obj_scale=obj_scale,
                reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, opt=self.opt, Inference=True)
            
            for k in dets:
                dets[k] = dets[k].detach().cpu().numpy()

        return output, dets    

    def object_pose_post_process(self, dets, c, s, h, w, opt, Inference=False):

        # A scale factor
        coefficient = 0.32

        # Scale bbox & pts and Regroup
        if not ('scores' in dets):
            return [[{}]]
        
        ret = []
        for i in range(dets['scores'].shape[0]):

            preds = []

            for j in range(len(dets['scores'][i])):
                item = {}
                item['score'] = float(dets['scores'][i][j])
                item['cls'] = int(dets['clses'][i][j])
                item['obj_scale'] = dets['obj_scale'][i][j]
                item['obj_scale_uncertainty'] = dets['obj_scale_uncertainty'][i][j]

                kps_displacement_std = dets['kps_displacement_std'][i, j] * (s[i] / max(w, h)) * coefficient
                item['kps_displacement_std'] = kps_displacement_std.reshape(-1, 16).flatten()

                # from w,h to c[i], s[i]
                bbox = transform_preds(dets['bboxes'][i, j].reshape(-1, 2), c[i], s[i], (w, h))
                item['bbox'] = bbox.reshape(-1, 4).flatten()

                item['ct'] = [(item['bbox'][0] + item['bbox'][2]) / 2, (item['bbox'][1] + item['bbox'][3]) / 2]

                kps = transform_preds(dets['kps'][i, j].reshape(-1, 2), c[i], s[i], (w, h))
                item['kps'] = kps.reshape(-1, 16).flatten()

                # To save some time, only perform this step when it is inference time
                if Inference == True:
                    kps_displacement_mean = transform_preds(dets['kps_displacement_mean'][i, j].reshape(-1, 2), c[i], s[i],
                                                            (w, h))
                    item['kps_displacement_mean'] = kps_displacement_mean.reshape(-1, 16).flatten()

                    kps_heatmap_mean = transform_preds(dets['kps_heatmap_mean'][i, j].reshape(-1, 2), c[i], s[i], (w, h))
                    item['kps_heatmap_mean'] = kps_heatmap_mean.reshape(-1, 16).flatten()

                    kps_heatmap_std = dets['kps_heatmap_std'][i, j] * (s[i] / max(w, h)) * coefficient
                    item['kps_heatmap_std'] = kps_heatmap_std.reshape(-1, 16).flatten()

                    item['kps_heatmap_height'] = dets['kps_heatmap_height'][i, j]

                preds.append(item)

            ret.append(preds)
        return ret

    def mapping_syn2obj(self, inp):
            syn_obj_mapping = {1:8, 2:5, 3:7, 4:6, 5:4, 6:1, 7:3, 8:2}
            temp = inp.copy()
            for idx, key in enumerate(syn_obj_mapping):
                temp[int(key)-1] = inp[int(syn_obj_mapping[key]-1)]
            return temp 

    def vis_keypoints(self, debugger, images, dets, output, scale=1, pre_hms=None, pre_hm_hp=None):
        # It will not affect the original dets value as we deepcopy it here
        dets['bboxes'] *= self.opt.down_ratio
        dets['kps'] *= self.opt.down_ratio
        dets['kps_displacement_mean'] *= self.opt.down_ratio
        dets['kps_displacement_std'] *= self.opt.down_ratio
        dets['kps_heatmap_mean'] *= self.opt.down_ratio

        # Save heatmap
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
                               img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        debugger.add_img(img, img_id='out_kps_processed_pred')
        heat = output['hm']
        K = 100
        heat = _nms(heat)
        scores, inds, clses, ys, xs = _topk(heat, K=K)  # inds: index in a single heatmap
        for i in range(K):

            if scores[0][i] > self.opt.vis_thresh:
                debugger.add_coco_hp(dets['kps_displacement_mean'][0, i], img_id='out_kps_processed_pred',
                                     pred_flag='pnp')

    def process(self, image_or_path_or_tensor, meta_inp):
        
        self.image_or_path_or_tensor = image_or_path_or_tensor

        self.image = cv2.imread(image_or_path_or_tensor)

        images, meta = self.pre_process(self.image, 1.0, meta_inp)

        images = images.to(self.opt.device)

        torch.cuda.synchronize()

        # run the network
        output, dets = self.run(images)
        torch.cuda.synchronize()
        
        # debug the keypoints
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
                            theme=self.opt.debugger_theme)
        self.vis_keypoints(debugger, images, copy.deepcopy(dets), output, 1.0, None, None)

        # post processing
        detections = []
        dets = self.object_pose_post_process(dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt, Inference=True)
        
        detections.append(dets)
        torch.cuda.synchronize()
        
        # Mainly apply NMS
        results = self.merge_outputs(detections[0])
        torch.cuda.synchronize()
        
        boxes = []
        if self.opt.use_pnp == True:
            for bbox in results:
                # 16 representation
                points_1 = np.array(bbox['kps_displacement_mean']).reshape(-1, 2)
                points_1 = [(x[0], x[1]) for x in points_1]
                points_2 = np.array(bbox['kps_heatmap_mean']).reshape(-1, 2)
                points_2 = [(x[0], x[1]) for x in points_2]
                points = np.hstack((points_1, points_2)).reshape(-1, 2)
                points_filtered = points

                ret = pnp_shell(self.opt, meta, bbox, points_filtered, bbox['obj_scale'], OPENCV_RETURN=self.opt.show_axes)
                if ret is not None:
                    boxes.append(ret)


        self.save_results(debugger, results)

        # Save results
        if os.path.isdir(self.opt.demo):
            # We set the saving folder's name = demo_save folder + source folder
            target_dir_path = os.path.join(self.opt.demo_save,
                                           f'{os.path.basename(self.opt.demo)}')
        else:
            # We set the saving folder's name = demo_save folder + source name
            target_dir_path = os.path.join(self.opt.demo_save,
                                           f'{os.path.splitext(os.path.basename(self.opt.demo))[0]}')
        if not os.path.exists(self.opt.demo_save):
            os.mkdir(self.opt.demo_save)
        debugger.save_all_imgs_demo(self.image_or_path_or_tensor, path=target_dir_path)
        
        return results


    def save_results(self, debugger, results):
        
        debugger.add_img(self.image, img_id='out_img_pred')

        for bbox in results:
            if bbox['score'] > self.opt.vis_thresh:
                if self.opt.reg_bbox:
                    # draw bounding box
                    debugger.add_coco_bbox(bbox['bbox'], 0, bbox['score'], img_id='out_img_pred')
                    # draw projected cuboid
                    debugger.add_coco_hp(bbox['projected_cuboid'], img_id='out_img_pred', pred_flag='pnp')
                    # print 6d pose
                    debugger.add_axes(bbox['kps_3d_cam'], self.opt.cam_intrinsic, img_id='out_img_pred')

                    obj_location = bbox['location']
                    obj_quaternion = bbox['quaternion_xyzw']
                    obj_scale = bbox['obj_scale']
                    

        # # read ground truth
        if SHANGRU_DATASET:
            path_json = self.image_or_path_or_tensor.replace('.jpg', ".json")
        else:
            path_json = self.image_or_path_or_tensor.replace('.png', ".json")

        max_objs = 10
        with open(path_json) as f:
            anns = json.load(f)

        num_objs = min(len(anns['objects']), max_objs)
        
        for k in range(num_objs):
            ann = anns['objects'][k]
            
            if SHANGRU_DATASET:
                kpts_3d = np.array(ann['keypoints_3d'][1:])
                kpts_3d_mapped = kpts_3d
            else:
                kpts_3d = np.array(ann['local_cuboid'][:8])
                kpts_3d_mapped = self.mapping_syn2obj(kpts_3d)
            
            x_scale = np.linalg.norm(kpts_3d_mapped[0] - kpts_3d_mapped[4])
            y_scale = np.linalg.norm(kpts_3d_mapped[0] - kpts_3d_mapped[2])
            z_scale = np.linalg.norm(kpts_3d_mapped[0] - kpts_3d_mapped[1])
            gt_scale = np.array([x_scale, y_scale, z_scale])
            gt_scale = np.abs(gt_scale) / gt_scale[1]

            kpts_2d = np.array(ann['projected_cuboid'][:8])

            kpts_2d_mapped = self.mapping_syn2obj(kpts_2d)
            debugger.add_coco_hp(kpts_2d_mapped, img_id='out_img_pred', pred_flag='pnp')
            
            
            # gt_location = ann['location']
            # gt_quaternion = ann['quaternion_xyzw']

        # self.cal_3dIoU(obj_scale, gt_scale, obj_location, obj_quaternion, gt_location, gt_quaternion)
        
        
    

    def cal_3dIoU(self, obj_scale, gt_scale, obj_location, obj_quat, gt_location, gt_quat):
        
        # transform to local space
        # relative dimension
        obj_cuboid3d = Cuboid3d(1 * np.array(obj_scale) / obj_scale[1])
        ori = R.from_quat(obj_quat).as_matrix()
        pose_pred = np.identity(4)
        pose_pred[:3, :3] = ori
        pose_pred[:3, 3] = obj_location
        point_3d_obj = obj_cuboid3d.get_vertices()

        point_3d_cam = pose_pred.T @ np.hstack(
            (np.array(point_3d_obj), np.ones((np.array(point_3d_obj).shape[0], 1)))).T
        point_3d_cam = point_3d_cam[:3, :].T  # 8 * 3
        point_3d_cam = np.insert(point_3d_cam, 0, np.mean(point_3d_cam, axis=0), axis=0)
        print(point_3d_cam)

        # relative dimension
        gt_cuboid3d = Cuboid3d(1 * np.array(gt_scale) / gt_scale[1])
        ori = R.from_quat(gt_quat).as_matrix()
        pose_pred = np.identity(4)
        pose_pred[:3, :3] = ori
        pose_pred[:3, 3] = gt_location
        point_3d_gt = gt_cuboid3d.get_vertices()

        point_3d_cam_gt = pose_pred.T @ np.hstack(
            (np.array(point_3d_gt), np.ones((np.array(point_3d_gt).shape[0], 1)))).T
        point_3d_cam_gt = point_3d_cam_gt[:3, :].T  # 8 * 3
        point_3d_cam_gt = np.insert(point_3d_cam_gt, 0, np.mean(point_3d_cam_gt, axis=0), axis=0)
        print(point_3d_cam_gt)

        # 3D iou
        box1 = Box(point_3d_cam)
        box2 = Box(point_3d_cam_gt)
        cal_3dIoU = IoU(box1, box2)
        out = cal_3dIoU.iou()
        print(out)

        # self.visualize_pointcloud_meshcat(point_3d_cam, point_3d_cam_gt)
        # self.visualize_scale_meshcat(obj_scale, gt_scale, obj_location, obj_quat, gt_location, gt_quat)
    
    def visualize_pointcloud_meshcat(self, point_3d_cam, point_3d_cam_gt):
        
        vis = meshcat.Visualizer(zmq_url = 'tcp://127.0.0.1:7000')
        vis.delete()
        
        obj_point = np.zeros((3, 100000))
        for k in range(point_3d_cam.shape[1]):
            obj_min = point_3d_cam[:,k].min()
            obj_max = point_3d_cam[:,k].max()
            obj_num = np.random.uniform(obj_min, obj_max, (1,100000))
            obj_point[k, :] = obj_num

        obj_color = np.ones_like(obj_point)
        obj_color[1,:] = 0.5
        vis['obj'].set_object(g.Points(
            g.PointsGeometry(obj_point, color=obj_color),
            g.PointsMaterial()
        ))

        gt_point = np.zeros((3, 100000))
        for k in range(point_3d_cam_gt.shape[1]):
            obj_min = point_3d_cam_gt[:,k].min()
            obj_max = point_3d_cam_gt[:,k].max()
            obj_num = np.random.uniform(obj_min, obj_max, (1,100000))
            gt_point[k, :] = obj_num

        obj_color = np.ones_like(gt_point)
        obj_color[2,:] = 0.5
        vis['gt'].set_object(g.Points(
            g.PointsGeometry(gt_point, color=obj_color),
            g.PointsMaterial()
        ))

    def visualize_scale_meshcat(self, obj_scale, gt_scale, obj_location, obj_quat, gt_location, gt_quat):
        
        # Visualization
        vis = meshcat.Visualizer(zmq_url = 'tcp://127.0.0.1:7000')
        vis.delete()
        object_scale = obj_scale
        new_obj = np.zeros((object_scale.shape))
        for i in range(len(object_scale)):
            new_obj[i] = object_scale[i]

        print('predicted', new_obj)
        vis['obj'].set_object(g.Box(new_obj),g.MeshBasicMaterial(color=[0,1.0,1.0], transparency=False, opacity=1)) #[1.09030998, 1.00013745, 0.67467052]
        vis['obj'].set_transform(tf.translation_matrix(obj_location)) #[-0.02249266131597753, -0.34124135508847536, 7.879638094486412]
        vis['obj'].set_transform(tf.quaternion_matrix(obj_quat)) #[-0.33831656, 0.38733576, -0.84074639, 0.16928799]

        vis["/Background"].set_property("top_color", [1, 0, 0])

        print('ground truth', gt_scale)
        vis['gt'].set_object(g.Box(gt_scale)) # [1.17995785, 1., 0.65468566]
        vis['gt'].set_transform(tf.translation_matrix(gt_location)) # [-9.834e-07, 1.728e-06, -12.770]
        vis['gt'].set_transform(tf.quaternion_matrix(gt_quat)) # [0.171, 0.832, 0.386, 0.360]
        
                


if __name__ == '__main__':
    opt = opts().parser.parse_args()
    # opt.demo = '/home/jianhey/CenterPose/custom_data/RealData/KV119/'
    opt.demo = '/home/jianhey/CenterPose/custom_data/test/000003.png'
    opt.demo_save = '../demo'
    opt.arch = 'dlav1_34'
    # opt.load_model = '/home/jianhey/CenterPose/exp/object_pose/objectron_retail_dlav1_34_2023-02-20-23-01/retail_best.pth'
    opt.load_model = '/home/jianhey/CenterPose/exp/object_pose/objectron_retail_dlav1_34_2023-02-07-09-17/retail_best.pth'
    
    # Default setting
    opt.nms = True
    opt.obj_scale = True
    opt.use_pnp = True
    opt = opts().parse(opt)
    opt = opts().init(opt)

    meta = {}
    # meta['camera_matrix'] = np.array(
    #         [[482.84283447265625, 0, 200], [0, 482.84283447265625, 200], [0, 0, 1]])
    meta['camera_matrix'] = np.array(
                [[1303.67529296875, 0, 960], [0, 1303.67529296875, 540], [0, 0, 1]])
    opt.cam_intrinsic = meta['camera_matrix']

    # input can be foler or image
    if os.path.isdir(opt.demo):
        image_names = []
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(opt.demo, file_name))
    else:
        image_names = [opt.demo]

    detector = SimpDetector(opt)
    
    for idx, image_name in enumerate(image_names):
        ret = detector.process(image_name, meta_inp=meta)

