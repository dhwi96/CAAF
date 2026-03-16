#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
import numpy as np
from PIL import Image
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3


# -----------------------------
# RE10K .txt 파서
# -----------------------------
def parse_re10k_txt(txt_path, images_dir):
    with open(txt_path, 'r') as f:
        lines = [ln.strip() for ln in f.readlines()]
    if len(lines) < 2:
        return []

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    img_paths = sorted(
        os.path.join(images_dir, fn) for fn in os.listdir(images_dir)
        if os.path.splitext(fn.lower())[1] in valid_exts
    )

    frame_lines = lines[1:]
    n = min(len(frame_lines), len(img_paths))
    frame_lines = frame_lines[:n]
    img_paths   = img_paths[:n]

    recs = []
    for ln, imgp in zip(frame_lines, img_paths):
        cols = ln.split()
        if len(cols) != 19:
            continue
        ts = int(cols[0])
        fx, fy, cx, cy = map(float, cols[1:5])

        with Image.open(imgp) as im:
            W, H = im.size
        K = np.array([
            [W * fx, 0.0,   W * cx],
            [0.0,   H * fy, H * cy],
            [0.0,    0.0,   1.0   ]
        ], dtype=np.float64)

        pose_vals = list(map(float, cols[7:19]))
        P34 = np.array(pose_vals, dtype=np.float64).reshape(3, 4)
        P44 = np.eye(4, dtype=np.float64)
        P44[:3, :4] = P34

        recs.append({
            "timestamp": ts,
            "img_path": imgp,
            "K": K,
            "P_w2c_4x4": P44
        })
    return recs


# -----------------------------
# 각도/메트릭 유틸
# -----------------------------
def rotation_angle_deg(R_gt, R_pr):
    dR = R_gt.transpose(-1, -2) @ R_pr
    tr = torch.diagonal(dR, dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1.0) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.rad2deg(torch.acos(cos))

def translation_angle_deg(t_gt, t_pr, ambiguity=True):
    dot = (t_gt * t_pr).sum(-1)
    nrm = (t_gt.norm(dim=-1) * t_pr.norm(dim=-1)).clamp_min(1e-8)
    cos = (dot / nrm).clamp(-1 + 1e-7, 1 - 1e-7)
    ang = torch.rad2deg(torch.acos(cos))
    if ambiguity:
        ang = torch.minimum(ang, (180.0 - ang).abs())
    return ang

def build_all_pairs_relative_errors(pred_w2c, gt_w2c):
    N = pred_w2c.shape[0]
    idx = torch.combinations(torch.arange(N, device=pred_w2c.device), r=2)
    i1, i2 = idx[:, 0], idx[:, 1]

    inv_gt = closed_form_inverse_se3(gt_w2c[i2])
    inv_pr = closed_form_inverse_se3(pred_w2c[i2])

    Rel_gt = gt_w2c[i1].bmm(inv_gt)
    Rel_pr = pred_w2c[i1].bmm(inv_pr)

    Rgt, tgt = Rel_gt[:, :3, :3], Rel_gt[:, :3, 3]
    Rpr, tpr = Rel_pr[:, :3, :3], Rel_pr[:, :3, 3]

    r_deg = rotation_angle_deg(Rgt, Rpr)
    t_deg = translation_angle_deg(tgt, tpr, ambiguity=True)
    return r_deg, t_deg

def accuracy_at_threshold(x_deg, th):
    return (x_deg < th).float().mean().item()

def auc_at_threshold(r_deg, t_deg, th_max):
    mx = torch.maximum(r_deg, t_deg).clamp_max(th_max - 1e-6)
    hist = torch.histc(mx, bins=th_max, min=0, max=th_max)
    hist = hist / mx.numel()
    cdf = torch.cumsum(hist, dim=0)
    return cdf.mean().item()


# -----------------------------
# 한 씬 평가
# -----------------------------
def eval_scene_one(model, scene_id, images_root, meta_dir, num_frames, device, amp_dtype):
    images_dir = os.path.join(images_root, scene_id)
    txt_path   = os.path.join(meta_dir, f"{scene_id}.txt")

    if not os.path.isdir(images_dir) or not os.path.isfile(txt_path):
        return None

    recs = parse_re10k_txt(txt_path, images_dir)
    if len(recs) < 2:
        return None

    # ★ 랜덤으로 num_frames개 샘플링 (프레임 수가 부족하면 있는 것 전부 사용)
    if num_frames is not None and len(recs) > num_frames:
        recs = random.sample(recs, num_frames)

    gt_w2c = torch.from_numpy(np.stack([r["P_w2c_4x4"] for r in recs], 0)).to(device)

    img_paths = [r["img_path"] for r in recs]
    images = load_and_preprocess_images(img_paths).to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            out = model(images)
        with torch.amp.autocast('cuda', dtype=torch.float64):
            pred_extri, _ = pose_encoding_to_extri_intri(out["pose_enc"], images.shape[-2:])
            pred_w2c = pred_extri[0].to(torch.float64)

    with torch.amp.autocast('cuda', dtype=torch.float64):
        r_deg, t_deg = build_all_pairs_relative_errors(pred_w2c, gt_w2c)

    metrics = {
        "RRA@15": accuracy_at_threshold(r_deg, 15.0),
        "RTA@15": accuracy_at_threshold(t_deg, 15.0),
        "AUC@15": auc_at_threshold(r_deg, t_deg, 15),
        "RRA@30": accuracy_at_threshold(r_deg, 30.0),
        "RTA@30": accuracy_at_threshold(t_deg, 30.0),
        "AUC@30": auc_at_threshold(r_deg, t_deg, 30),
    }
    return metrics


# -----------------------------
# 메인 (배치)
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", type=str, required=True,
                    help=r'예: D:\3DGS\Datasets\RealEstate10k\re10k_test_images')
    ap.add_argument("--meta_dir", default="D:/CVPR_2026/RealEstate10k/txt/test", type=str, required=False,
                    help=r'RE10K .txt 들이 있는 폴더 (scene_id.txt)')
    ap.add_argument("--scene_list", default="./re10K_test_1800.txt", type=str, required=False,
                    help=r'한 줄에 하나씩 scene_id가 적힌 텍스트 파일')
    ap.add_argument("--model_path", default="D:/CVPR_2026/VGGT_eval/Camera_pose/model_tracker_fixed_e20.pt", type=str,
                    help=r'VGGT 체크포인트 경로')
    ap.add_argument("--num_frames", type=int, default=10,
                    help="씬당 랜덤 샘플링할 프레임 수 (default: 10)")
    ap.add_argument("--seed", type=int, default=777, help="랜덤 시드")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = (torch.bfloat16 if (torch.cuda.is_available()
                                    and torch.cuda.get_device_capability()[0] >= 8)
                 else torch.float16)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = VGGT().to(device).eval()
    if args.model_path:
        print(f"[LOAD] {args.model_path}")
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state)

    with open(args.scene_list, "r", encoding="utf-8") as f:
        scene_ids = [ln.strip() for ln in f if ln.strip()]
    print(f"[INFO] #scenes = {len(scene_ids)}, num_frames per scene = {args.num_frames}, seed = {args.seed}")

    per_scene = {}
    for sid in scene_ids:
        m = eval_scene_one(model, sid, args.images_root, args.meta_dir,
                           args.num_frames, device, amp_dtype)
        if m is None:
            continue
        per_scene[sid] = m
        print(f"{sid:>s}  "
              f"RRA@15 {m['RRA@15']:.4f}  RTA@15 {m['RTA@15']:.4f}  AUC@15 {m['AUC@15']:.4f}  |  "
              f"RRA@30 {m['RRA@30']:.4f}  RTA@30 {m['RTA@30']:.4f}  AUC@30 {m['AUC@30']:.4f}")

    if not per_scene:
        print("[WARN] No valid scenes evaluated.")
        return

    keys = ["RRA@15", "RTA@15", "AUC@15", "RRA@30", "RTA@30", "AUC@30"]
    mean_metrics = {k: float(np.mean([per_scene[s][k] for s in per_scene])) for k in keys}

    print("=" * 80)
    print("Per-scene means (macro average):")
    print("  " + "  ".join(f"{k} {mean_metrics[k]:.4f}" for k in keys))
    print("=" * 80)

if __name__ == "__main__":
    main()
