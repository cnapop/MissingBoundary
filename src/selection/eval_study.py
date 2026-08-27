#!/usr/bin/env python3
"""
eval_study.py — Selection Study 评估
====================================
fork /tmp/test_DRAEM_pipe_fryum.py: 固定 obj_list=['pipe_fryum'],
GT mask 先 (gt>0.01) 二值化 (VisA GT 软值), 输出 METRIC:{json} 单行供 driver 解析。
用法: PYTHONPATH=/data/chenjiawen/DRAEM python eval_study.py \
    --gpu_id 0 --base_model_name DRAEM_test_0.0001_200_bs8 \
    --data_path outputs/visa_datasets --checkpoint_path outputs/selection/S0_random/checkpoints/fix2
"""
import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os, json


def test(obj_names, mvtec_path, checkpoint_path, base_model_name, gpu_id):
    obj_ap_pixel_list, obj_auroc_pixel_list = [], []
    obj_ap_image_list, obj_auroc_image_list = [], []
    for obj_name in obj_names:
        img_dim = 256
        run_name = base_model_name + "_" + obj_name + '_'
        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name + ".pckl"),
                                         map_location=f'cuda:{gpu_id}'))
        model.cuda()
        model.eval()
        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name + "_seg.pckl"),
                                             map_location=f'cuda:{gpu_id}'))
        model_seg.cuda()
        model_seg.eval()

        dataset = MVTecDRAEMTestDataset(os.path.join(mvtec_path, obj_name, "test"),
                                        resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0
        anomaly_score_gt, anomaly_score_prediction = [], []

        for sample_batched in dataloader:
            gray_batch = sample_batched["image"].cuda()
            is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"].detach().numpy()[0].transpose((1, 2, 0))
            gray_rec = model(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)
            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)
            out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()
            out_mask_averaged = torch.nn.functional.avg_pool2d(
                out_mask_sm[:, 1:, :, :], 21, stride=1, padding=21 // 2).cpu().detach().numpy()
            anomaly_score_prediction.append(np.max(out_mask_averaged))
            flat_true = true_mask.flatten()
            flat_out = out_mask_cv.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true
            mask_cnt += 1

        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

        total_gt_pixel_scores = (total_gt_pixel_scores > 0.01).astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)

        obj_auroc_image_list.append(auroc)
        obj_ap_image_list.append(ap)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_ap_pixel_list.append(ap_pixel)
        print(json.dumps({
            'obj': obj_name,
            'image_auc': round(float(auroc), 4),
            'image_ap': round(float(ap), 4),
            'pixel_auc': round(float(auroc_pixel), 4),
            'pixel_ap': round(float(ap_pixel), 4),
        }))

    metric = {
        'obj': obj_names,
        'image_auc': float(np.mean(obj_auroc_image_list)),
        'image_ap': float(np.mean(obj_ap_image_list)),
        'pixel_auc': float(np.mean(obj_auroc_pixel_list)),
        'pixel_ap': float(np.mean(obj_ap_pixel_list)),
    }
    print('METRIC:' + json.dumps(metric))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--base_model_name', action='store', type=str, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    args = parser.parse_args()
    obj_list = ["pipe_fryum"]
    with torch.cuda.device(args.gpu_id):
        test(obj_list, args.data_path, args.checkpoint_path, args.base_model_name, args.gpu_id)
