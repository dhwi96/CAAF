#!/usr/bin/env python3
"""
BlendedMVS Chamfer-only evaluation using VGGT (camera-free; predicted cameras)

- Computes ONLY 3D metrics per scene:
  * Accuracy    : mean min distance(pred -> GT)
  * Completeness: mean min distance(GT -> pred)
  * Chamfer     : 0.5 * (Accuracy + Completeness)

Dataset layout (per scene_id):
  {bms_root}/{scene_id}/blended_images/{########}.jpg (또는 {########}_masked.jpg 등 변형 포함 가능)
  {bms_root}/{scene_id}/rendered_depth_maps/{########}.pfm   <-- (리드닷 없음)
  {bms_root}/{scene_id}/cams/{########}_cam.txt              <-- (선택) GT 카메라 (MVSNet 포맷)

Custom image root (옵션; denoise/압축본 사용 시):
  --images_root 를 넘기면 blended_images 대신 다음 경로를 사용:
    {images_root}/{scene_id}/{images_subdir}/{########}.jpg (기본 images_subdir='.')
  파일명 키(########)는 depth의 파일명과 동일하게 매칭함.
  *_masked.jpg 등 변형이 섞여도 동일 베이스키(########)로 그룹핑 후 원본 '########.jpg'를 우선 사용.

Notes
- GT 3D는 GT 카메라(K,w2c)로, Pred 3D는 VGGT pose_enc(예측 카메라)로 언프로젝션.
- 모든 cv2.resize는 safe_resize()를 통해서만 수행하여 dsize=0 오류 방지.
- pred depth/conf가 (S,H,W) 또는 (1,S,H,W,1) 등 다양한 형태로 나올 수 있어,
  프레임별 2D 맵을 _extract_frame_2d()로 안전 추출.
- 정합은 Umeyama(유사변환, scale 포함). (VGGT/MASt3R/DTU 관행)
- 옵션: --per_frame_align 로 프레임 개별 정합 후 평균 Chamfer 산출.
- 옵션: --unit_autofix (CO3D 패치처럼 mm<->m 휴리스틱 교정)
- 옵션: --dist_cap_m (Chamfer 계산 시 거리 상한; 예: 0.5m)

Example:
  python ./test_blendedmvs_depth.py ^
    --bms_root D:/CVPR_2026/BlendedMVS ^
    --images_root D:/CVPR_2026/ProcessedImages ^
    --images_subdir proc_images ^
    --model_path ./model_tracker_fixed_e20.pt ^
    --num_frames 10 ^
    --unit_autofix --dist_cap_m 0.5 ^
    --depth_conf_thresh 0 ^
    --depth_clip_percentile 99.5 ^
    --per_frame_align ^
    --dump_preds_shapes --verbose
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 스크립트 최상단에 추가

import argparse
import os
import os.path as osp
import glob
import random
import csv
import numpy as np
import cv2
import torch
from scipy.spatial import cKDTree
from collections import defaultdict
from PIL import Image  # H,W 복구용

# VGGT
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# --------------------- Args ---------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--bms_root', default="./BlendedMVS")
    p.add_argument('--scene_ids', nargs='*', default=None, help='If omitted, evaluate all scenes under bms_root')
    p.add_argument('--model_path', default='D:/CVPR_2026/VGGT_eval/Camera_pose/model_tracker_fixed_e20.pt')
    p.add_argument('--num_frames', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--depth_conf_thresh', type=float, default=0, help='mask by confidence if provided')
    p.add_argument('--resize_interp', choices=['nearest','linear'], default='nearest')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--dump_preds_shapes', action='store_true')
    p.add_argument('--save_csv', type=str, default=None)
    p.add_argument('--max_scenes', type=int, default=None)
    p.add_argument('--max_points', type=int, default=100000, help='Subsample size per cloud before alignment')
    p.add_argument('--depth_clip_percentile', type=float, default=99.5,
                   help='e.g., 99.9 to suppress far-out depth outliers for both GT/Pred before unprojection')
    p.add_argument('--per_frame_align', action='store_true', default=True,
                   help='Align & compute Chamfer per-frame, then average')
    p.add_argument('--flip_gt_extrinsic_autotest', action='store_true',
                   help='Try flipped GT extrinsic on first frame; keep if reduces Chamfer')

    # ★ 새 옵션: unit auto-fix & distance cap
    p.add_argument('--unit_autofix', action='store_true', default=True,
                   help='Auto-fix mm<->m by comparing pred/gt median magnitudes (like CO3D patch)')
    p.add_argument('--dist_cap_m', type=float, default=0.5,
                   help='Optional max distance cap (meters) in Chamfer (e.g., 0.5)')

    # ★ 새 옵션: 별도의 이미지 루트 사용(denoise/압축본)
    p.add_argument('--images_root', type=str, default=None,
                   help='If set, images are taken from {images_root}/{scene_id}/{images_subdir}')
    p.add_argument('--images_subdir', type=str, default='.',
                   help='Subdirectory under images_root/scene_id to read images from (default=".")')
    return p.parse_args()

# --------------------- Safe resize ---------------------
def safe_resize(arr, W, H, inter=cv2.INTER_NEAREST, name='array', verbose=False):
    """Resize only when needed; assert W,H>0; print helpful diagnostics on error."""
    if arr is None:
        return None
    if arr.size == 0:
        raise ValueError(f"[safe_resize] source {name} is empty (shape={arr.shape})")
    if arr.ndim < 2:
        raise ValueError(f"[safe_resize] source {name} must be at least 2D, got shape={arr.shape}")
    h0, w0 = arr.shape[:2]
    if (h0, w0) == (H, W):
        return arr
    if H <= 0 or W <= 0:
        raise ValueError(f"[safe_resize] invalid target size for {name}: (W={W}, H={H})")
    try:
        out = cv2.resize(arr, (int(W), int(H)), interpolation=inter)
        return out
    except Exception as e:
        if verbose:
            print(f"[safe_resize][ERR] name={name}, src_shape={arr.shape}, tgt=(W={W},H={H}), inter={inter}, dtype={getattr(arr,'dtype','unknown')}")
        raise

# --------------------- I/O ---------------------
def load_pfm(path):
    with open(path, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header not in ('Pf', 'PF'):
            raise ValueError('Not a PFM file: ' + header)
        dims = f.readline().decode('utf-8').strip()
        while dims.startswith('#'):
            dims = f.readline().decode('utf-8').strip()
        w, h = map(int, dims.split())
        scale = float(f.readline().decode('utf-8').strip())
        data = np.fromfile(f, '<f' if scale < 0 else '>f')
        if header == 'PF':
            data = data.reshape((h, w, 3))[..., 0]
        else:
            data = data.reshape((h, w))
        data = data.astype(np.float32)
        data[~np.isfinite(data)] = 0.0
        return data

# --------------- BlendedMVS (MVSNet) cam parser ---------------
def load_bmvs_cam_txt(cam_path):
    """
    MVSNet/BlendedMVS cam.txt parser
    Format:
      extrinsic
      4x4
      intrinsic
      3x3
      depth_min depth_interval   (ignored here)
    Returns: (K[3x3], w2c[3x4])
    Note:
      cam.txt extrinsic is commonly camera-to-world (c2w).
      We convert to world-to-camera (w2c) (3x4).
    """
    with open(cam_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    assert lines[0].lower().startswith('extrinsic'), f'bad cam file: {cam_path}'
    E = np.array([[float(x) for x in lines[i].split()] for i in range(1,5)], dtype=np.float64)  # 4x4
    idx = 5
    assert lines[idx].lower().startswith('intrinsic'), f'bad cam file: {cam_path}'
    K = np.array([[float(x) for x in lines[idx+1].split()],
                  [float(x) for x in lines[idx+2].split()],
                  [float(x) for x in lines[idx+3].split()]], dtype=np.float64)
    # c2w -> w2c
    R_c2w = E[:3,:3]; t_c2w = E[:3,3]
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    w2c = np.hstack([R_w2c, t_w2c.reshape(3,1)])  # 3x4
    return K, w2c

# ----------------- 3D helpers -------------------
def unproject(depth, K, extri_w2c, mask=None, verbose=False, scene_id=''):
    H, W = depth.shape
    # mask 리사이즈는 "필요할 때만" 수행
    if mask is not None:
        mask_u8 = (mask.astype(np.uint8) if mask.dtype != np.uint8 else mask)
        if mask_u8.shape != (H, W):
            inter = cv2.INTER_NEAREST
            mask_u8 = safe_resize(mask_u8, W, H, inter=inter, name='mask', verbose=verbose)
        valid = (depth > 0) & np.isfinite(depth) & (mask_u8 > 0)
    else:
        valid = (depth > 0) & np.isfinite(depth)
    if valid.sum() == 0:
        return np.zeros((0, 3), dtype=np.float64)

    ys, xs = np.indices((H, W))
    xs = xs.astype(np.float64); ys = ys.astype(np.float64)
    z = depth.astype(np.float64)

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    Z = z
    pts_cam = np.stack([X, Y, Z], -1).reshape(-1, 3)[valid.reshape(-1)]
    if extri_w2c is None:
        return pts_cam
    R = extri_w2c[:3, :3]; t = extri_w2c[:3, 3]
    pts_w = (R.T @ (pts_cam.T - t[:, None])).T
    return pts_w

def umeyama_align(src, dst, with_scale=True):
    src = np.asarray(src, dtype=np.float64); dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(0); mu_d = dst.mean(0)
    src_c = src - mu_s; dst_c = dst - mu_d
    cov = (dst_c.T @ src_c) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2,2] = -1
    R = U @ S @ Vt
    s = 1.0
    if with_scale:
        var = (src_c**2).sum() / src.shape[0]
        s = np.trace(np.diag(D) @ S) / var
    t = mu_d - s * R @ mu_s
    return R, t, s

def chamfer_metrics(pred_pts, gt_pts):
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.inf, np.inf, np.inf
    tree_gt = cKDTree(gt_pts); d1, _ = tree_gt.query(pred_pts, k=1)
    tree_pd = cKDTree(pred_pts); d2, _ = tree_pd.query(gt_pts, k=1)
    acc  = float(d1.mean())
    comp = float(d2.mean())
    cham = 0.5 * (acc + comp)
    return acc, comp, cham

def chamfer_with_cap(P, G, cap=None):
    if len(P) == 0 or len(G) == 0:
        return np.inf, np.inf, np.inf
    tree_g = cKDTree(G); d_pg, _ = tree_g.query(P, k=1)
    tree_p = cKDTree(P); d_gp, _ = tree_p.query(G, k=1)
    if cap is not None and np.isfinite(cap):
        d_pg = np.minimum(d_pg, cap)
        d_gp = np.minimum(d_gp, cap)
    acc  = float(d_pg.mean())
    comp = float(d_gp.mean())
    cham = 0.5 * (acc + comp)
    return acc, comp, cham

def get_frame_mat(arr, idx):
    if arr is None: return None
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    a = np.asarray(arr)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 3 and a.shape[0] > idx:
        return a[idx]
    if a.ndim == 2 and a.size in (9, 12):
        return a.reshape(3, 3) if a.size == 9 else a.reshape(3, 4)
    return None

# ----------------- Frame extractor -----------------
def _extract_frame_2d(tensor, frame_idx, expect_shape, name='tensor', verbose=False):
    if tensor is None:
        return None
    a = tensor.detach().cpu().numpy()

    if a.ndim == 5:
        B, S, H, W, C = a.shape
        if C == 1:
            a = a[..., 0]
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]

    if a.ndim == 4:
        if expect_shape is not None and a.shape[-2:] == expect_shape:
            if a.shape[0] == 1 and a.shape[1] > frame_idx:
                return np.asarray(a[0, frame_idx], dtype=np.float32)
        if a.shape[-1] == 1:
            S = a.shape[0]
            if S > frame_idx:
                return np.asarray(a[frame_idx, ..., 0], dtype=np.float32)

    if a.ndim == 3:
        if expect_shape is not None and a.shape[1:] == expect_shape and a.shape[0] > frame_idx:
            return np.asarray(a[frame_idx], dtype=np.float32)
        if expect_shape is not None and a.shape[1:] == expect_shape and a.shape[0] == 1:
            return np.asarray(a[0], dtype=np.float32)
        if a.shape[-1] == 3 and (a.shape[0], a.shape[1]) == expect_shape:
            return np.asarray(a[..., 2], dtype=np.float32)

    if a.ndim == 2:
        return np.asarray(a, dtype=np.float32)

    a2 = np.squeeze(a)
    if a2.ndim == 3 and expect_shape is not None and a2.shape[1:] == expect_shape:
        if a2.shape[0] > frame_idx:
            return np.asarray(a2[frame_idx], dtype=np.float32)
    if verbose:
        print(f"[WARN] {_extract_frame_2d.__name__}: unexpected shape for {name} -> {a.shape}, squeezed={a2.shape}")
    return None

def extract_depth_from_preds(preds, frame_idx, expect_shape, verbose=False):
    d = None
    if isinstance(preds, dict):
        for k in ['depth','depth_map','pred_depth','dmap']:
            if k in preds:
                d = _extract_frame_2d(preds[k], frame_idx, expect_shape, name=f"{k}", verbose=verbose)
                if d is not None:
                    break
    if d is None and isinstance(preds, dict) and 'world_points' in preds:
        wp = preds['world_points'].detach().cpu().numpy()
        if wp.ndim == 5 and wp.shape[0] == 1:
            wp = wp[0]
        if wp.ndim == 4 and wp.shape[-1] == 3 and wp.shape[0] > frame_idx:
            d = wp[frame_idx, ..., 2].astype(np.float32)
        elif wp.ndim == 3 and wp.shape[0] == 3:
            d = wp[2].astype(np.float32)
        else:
            arr = np.squeeze(wp)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                d = arr[..., 2].astype(np.float32)
    if d is None:
        return None
    H, W = expect_shape
    if d.shape != (H, W):
        inter = cv2.INTER_NEAREST
        d = safe_resize(d, W, H, inter=inter, name='pred_depth', verbose=True).astype(np.float32)
    return d

def extract_conf_from_preds(preds, frame_idx, expect_shape, verbose=False):
    c = None
    if isinstance(preds, dict):
        for k in ['depth_conf','depth_confidence','dconf','world_points_conf']:
            if k in preds:
                c = _extract_frame_2d(preds[k], frame_idx, expect_shape, name=f"{k}", verbose=verbose)
                if c is not None:
                    break
    if c is None:
        return None
    H, W = expect_shape
    if c.shape != (H, W):
        inter = cv2.INTER_NEAREST
        c = safe_resize(c, W, H, inter=inter, name='pred_conf', verbose=True).astype(np.float32)
    return c

# --------------- Scene discovery ---------------
def discover_scenes(bms_root):
    scenes = []
    if not osp.isdir(bms_root):
        return scenes
    for name in sorted(os.listdir(bms_root)):
        sp = osp.join(bms_root, name)
        if not osp.isdir(sp): continue
        if osp.isdir(osp.join(sp, 'rendered_depth_maps')):
            # 이미지 디렉토리는 images_root 사용 시에도 필수는 아님
            scenes.append(name)
    return scenes

# 이미지 경로 수집 유틸 (blended_images 또는 images_root/scene/subdir)
def collect_images_for_scene(args, scene_id, depth_keys):
    if args.images_root:
        base = osp.join(args.images_root, scene_id, args.images_subdir)
    else:
        base = osp.join(args.bms_root, scene_id, 'blended_images')

    img_patterns = ['*.jpg','*.jpeg','*.png','*.JPG','*.JPEG','*.PNG']
    all_imgs = []
    if osp.isdir(base):
        for pat in img_patterns:
            all_imgs += glob.glob(osp.join(base, pat))
    all_imgs = sorted(set(all_imgs))

    def _img_base_key(p):
        base = osp.splitext(osp.basename(p))[0]
        base = base.lstrip('.')
        base = base.split('_')[0]
        return base

    groups = defaultdict(list)
    for p in all_imgs:
        k = _img_base_key(p)
        if k in depth_keys:
            groups[k].append(p)

    img_map = {}
    for k, plist in groups.items():
        preferred = [p for p in plist if osp.basename(p).lower() == f"{k}.jpg"]
        img_map[k] = preferred[0] if preferred else sorted(plist)[0]
    return img_map, base, all_imgs

# --------------- Per-scene eval ----------------
def evaluate_scene(args, model, scene_id):
    scene_dir = osp.join(args.bms_root, scene_id)
    dep_dir = osp.join(scene_dir, 'rendered_depth_maps')

    # 깊이: "########.pfm" (리드닷 없음)
    dep_paths = []
    for pat in ['*.pfm','*.PFM']:
        dep_paths += glob.glob(osp.join(dep_dir, pat))
    dep_paths = sorted(set(dep_paths))

    def _depth_key(p):
        return osp.splitext(osp.basename(p))[0]

    dep_map = { _depth_key(p): p for p in dep_paths }
    dep_keys = set(dep_map.keys())

    # 이미지 맵 구성(별도 이미지 루트 우선)
    img_map, img_base_dir, img_all = collect_images_for_scene(args, scene_id, dep_keys)
    common_keys = sorted(set(img_map.keys()) & set(dep_map.keys()))

    # GT cams 디렉토리 탐색 및 로딩
    cams_dir = None
    for c in [osp.join(scene_dir, 'cams'), osp.join(scene_dir, 'Cams'), osp.join(scene_dir, 'cameras')]:
        if osp.isdir(c):
            cams_dir = c
            break
    gtK_map, gtw2c_map = {}, {}
    if cams_dir is not None:
        for k in dep_map.keys():
            cam_txt = osp.join(cams_dir, f"{k}_cam.txt")
            if osp.exists(cam_txt):
                try:
                    Kgt, W2Cgt = load_bmvs_cam_txt(cam_txt)
                    gtK_map[k] = Kgt
                    gtw2c_map[k] = W2Cgt
                except Exception as e:
                    if args.verbose:
                        print(f"[{scene_id}] [WARN] parse cam failed: {osp.basename(cam_txt)} -> {e}")

    # ---------- 진단 ----------
    if args.verbose:
        print(f"[{scene_id}] DISCOVER")
        print(f"  images base: {img_base_dir}  (exists={osp.isdir(img_base_dir)})")
        print(f"  images total: {len(img_all)}")
        print(f"  depths total: {len(dep_paths)}  (expect '########.pfm')")
        print(f"  image keys:   {len(img_map)}")
        print(f"  depth keys:   {len(dep_map)}")
        print(f"  common keys:  {len(common_keys)}")
        if len(img_all) > 0:
            print(f"  sample image: {osp.basename(img_all[0])}")
        if len(dep_paths) > 0:
            print(f"  sample depth: {osp.basename(dep_paths[0])}")
        if cams_dir:
            has_cam = sum(1 for k in common_keys if k in gtK_map)
            print(f"  GT cams dir: {osp.basename(cams_dir)}  (found {has_cam}/{len(common_keys)} frames)")
        if len(common_keys) == 0:
            img_only = sorted(set(img_map.keys()) - set(dep_map.keys()))
            dep_only = sorted(set(dep_map.keys()) - set(img_map.keys()))
            print(f"  [DIAG] only-in-images (first 5): {img_only[:5]}")
            print(f"  [DIAG] only-in-depths (first 5): {dep_only[:5]}")

    if len(common_keys) == 0:
        return None

    # ---------- 프레임 샘플링 ----------
    chosen = common_keys if len(common_keys) <= args.num_frames else sorted(random.sample(common_keys, args.num_frames))
    ims = [img_map[k] for k in chosen]
    dps = [dep_map[k] for k in chosen]

    # ---------- 모델 입력 ----------
    device = args.device
    if device.startswith('cuda'):
        cc = torch.cuda.get_device_capability(device=None)[0]
        amp_dtype = torch.bfloat16 if cc >= 8 else torch.float16
    else:
        amp_dtype = torch.float32

    imgs_tensor = load_and_preprocess_images(ims).to(device)

    # [★ 수정] 차원 확인 후 배치 차원(B) 추가
    if imgs_tensor.ndim == 4:
        # (S, C, H, W) -> (1, S, C, H, W) 로 강제 변환
        imgs_tensor = imgs_tensor.unsqueeze(0)

    # [★ 수정] 이제 안전하게 5개로 풀 수 있음
    B, S, C, H, W = imgs_tensor.shape

    MAX_DIM = 512  # 640이나 512로 줄여보세요
    B, S, C, H, W = imgs_tensor.shape
    if max(H, W) > MAX_DIM:
        import torch.nn.functional as F
        scale = MAX_DIM / max(H, W)
        new_H, new_W = int(H * scale), int(W * scale)
        # 32의 배수로 맞춤 (ViT 모델 등에서 중요할 수 있음)
        new_H = (new_H // 14) * 14
        new_W = (new_W // 14) * 14
        
        # 0이 되는 것 방지
        if new_H == 0: new_H = 14
        if new_W == 0: new_W = 14
        
        # (B*S, C, H, W)로 병합 후 리사이즈
        imgs_tensor = imgs_tensor.view(B*S, C, H, W)
        imgs_tensor = F.interpolate(imgs_tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)
        imgs_tensor = imgs_tensor.view(B, S, C, new_H, new_W)
        
        if args.verbose:
            print(f"[{scene_id}] Resized input to ({new_H}, {new_W}) to save VRAM")

    H = int(imgs_tensor.shape[-2]) if imgs_tensor.ndim >= 3 else 0
    W = int(imgs_tensor.shape[-1]) if imgs_tensor.ndim >= 3 else 0
    if (H <= 0 or W <= 0):
        try:
            with Image.open(ims[0]) as im0:
                W0, H0 = im0.size
                H, W = int(H0), int(W0)
                if args.verbose:
                    print(f"[{scene_id}] Recovered H,W from image file: ({H},{W})")
        except Exception as e:
            if args.verbose:
                print(f"[{scene_id}] [ERR] Cannot recover H,W from {ims[0]}: {e}")
            return None

    if args.verbose:
        print(f"[{scene_id}] frames={len(ims)} size=({H},{W})")
        print(f"[{scene_id}] first image path: {osp.basename(ims[0])}")
        print(f"[{scene_id}] first depth path: {osp.basename(dps[0])}")

    # ---------- GT depth 로드(+리사이즈) ----------
    gts, valm, gt_hw_native = [], [], []
    for dp in dps:
        d = load_pfm(dp)
        h0, w0 = d.shape[:2]
        gt_hw_native.append((h0, w0))
        if d.shape != (H, W):
            inter = cv2.INTER_NEAREST if args.resize_interp == 'nearest' else cv2.INTER_LINEAR
            d = safe_resize(d, W, H, inter=inter, name='gt_depth', verbose=args.verbose).astype(np.float32)
        gts.append(d)
        valm.append((d > 0) & np.isfinite(d))
    gts = np.stack(gts, 0); valm = np.stack(valm, 0)

    # (옵션) 깊이 아웃라이어 클리핑 (GT)
    if args.depth_clip_percentile is not None:
        q = float(args.depth_clip_percentile)
        for i in range(len(gts)):
            g = gts[i]
            valid = g[g > 0]
            if valid.size > 0:
                clip = float(np.percentile(valid, q))
                gts[i] = np.minimum(gts[i], clip)

    # ---------- Inference ----------
    with torch.no_grad():
        if device.startswith('cuda'):
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                preds = model(imgs_tensor)
        else:
            preds = model(imgs_tensor)

    if args.dump_preds_shapes:
        print(f'[{scene_id}] PRED KEYS & SHAPES:')
        if isinstance(preds, dict):
            for k, v in preds.items():
                print(f"  {k}: shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)}")
        else:
            print('  preds type:', type(preds))

    # ---------- Predicted depth/Confidence ----------
    pdepth, pconf = [], []
    for i in range(len(ims)):
        d = extract_depth_from_preds(preds, i, (H, W), verbose=args.verbose)
        if d is None:
            if args.verbose: print(f"[{scene_id}] [WARN] no depth for frame {i}")
            d = np.zeros((H, W), dtype=np.float32)
        c = extract_conf_from_preds(preds, i, (H, W), verbose=args.verbose)
        pdepth.append(d); pconf.append(c)
    pdepth = np.stack(pdepth, 0)

    # (옵션) 깊이 아웃라이어 클리핑 (Pred)
    if args.depth_clip_percentile is not None:
        q = float(args.depth_clip_percentile)
        for i in range(len(pdepth)):
            p = pdepth[i]
            valid = p[p > 0]
            if valid.size > 0:
                clip = float(np.percentile(valid, q))
                pdepth[i] = np.minimum(pdepth[i], clip)

    # ---------- 마스크 ----------
    mask = valm.copy()
    if any(c is not None for c in pconf):
        conf_all = np.stack([c if c is not None else np.ones((H, W), np.float32) for c in pconf], 0)
        mask = mask & (conf_all >= args.depth_conf_thresh)

    # ---------- Predicted cameras ----------
    extri_all, intri_all = None, None
    if isinstance(preds, dict) and 'pose_enc' in preds:
        try:
            extri_all, intri_all = pose_encoding_to_extri_intri(preds['pose_enc'], (H, W))
            if isinstance(extri_all, torch.Tensor):
                extri_all = extri_all.detach().cpu().numpy()
            if isinstance(intri_all, torch.Tensor):
                intri_all = intri_all.detach().cpu().numpy()
        except Exception as e:
            if args.verbose: print(f"[{scene_id}] pose_encoding_to_extri_intri failed: {e}")
            extri_all, intri_all = None, None
    if extri_all is None or intri_all is None:
        if args.verbose: print(f"[{scene_id}] no predicted cameras -> skip scene")
        return None

    # ---------- Unproject & Chamfer ----------
    def maybe_flip_w2c(w2c):
        R = w2c[:3,:3]; t = w2c[:3,3]
        c2w = np.eye(4); c2w[:3,:3] = R.T; c2w[:3,3] = -R.T @ t
        R2 = c2w[:3,:3].T; t2 = -R2 @ c2w[:3,3]
        return np.hstack([R2, t2.reshape(3,1)])

    pred_pts_all, gt_pts_all = [], []
    per_frame_scores = []

    for i, k in enumerate(chosen):
        w2c_pred = get_frame_mat(extri_all, i)
        K_pred   = get_frame_mat(intri_all, i)

        # GT 카메라 (cam.txt 있으면 사용; 없으면 pred로 fallback)
        if k in gtK_map and k in gtw2c_map:
            K_gt, w2c_gt = gtK_map[k], gtw2c_map[k]
        else:
            K_gt, w2c_gt = K_pred, w2c_pred
            if args.verbose:
                print(f"[{scene_id}] [WARN] no GT cam for frame {k} -> using predicted cam (fallback)")

        if any(x is None for x in [w2c_pred, K_pred, w2c_gt, K_gt]):
            if args.verbose: print(f"[{scene_id}] missing pose (pred/gt) for frame {k}")
            continue

        if isinstance(w2c_pred, torch.Tensor): w2c_pred = w2c_pred.detach().cpu().numpy()
        if isinstance(K_pred, torch.Tensor):   K_pred   = K_pred.detach().cpu().numpy()
        if isinstance(w2c_gt, torch.Tensor):   w2c_gt   = w2c_gt.detach().cpu().numpy()
        if isinstance(K_gt, torch.Tensor):     K_gt     = K_gt.detach().cpu().numpy()

        # GT K 스케일링 (원본 GT 해상도 -> (H,W))
        h0, w0 = gt_hw_native[i]
        sx = W / float(w0)
        sy = H / float(h0)
        K_gt_scaled = K_gt.copy().astype(np.float64)
        K_gt_scaled[0,0] *= sx; K_gt_scaled[1,1] *= sy
        K_gt_scaled[0,2] *= sx; K_gt_scaled[1,2] *= sy
        if args.verbose and i == 0:
            print(f"[{scene_id}] scale gtK: sx={sx:.4f} sy={sy:.4f}  (gt_native=({h0},{w0}) -> ({H},{W}))")

        m = mask[i]

        # (옵션) GT extrinsic 플립 자동시험(첫 프레임만)
        if args.flip_gt_extrinsic_autotest and i == 0:
            try:
                pr3d_a = unproject(pdepth[i], K_pred,     w2c_pred, mask=m)
                gt3d_a = unproject(gts[i],    K_gt_scaled, w2c_gt,   mask=m)
                w2c_gt_flip = maybe_flip_w2c(w2c_gt)
                gt3d_b = unproject(gts[i], K_gt_scaled, w2c_gt_flip, mask=m)

                def chamfer_after_umeyama(P, G):
                    if P.shape[0]==0 or G.shape[0]==0: return np.inf
                    R,t,s = umeyama_align(P, G, with_scale=True)
                    Pa = (s*(R@P.T).T)+t
                    a,c,ch = chamfer_metrics(Pa, G)
                    return ch
                ch_a = chamfer_after_umeyama(pr3d_a, gt3d_a)
                ch_b = chamfer_after_umeyama(pr3d_a, gt3d_b)
                if ch_b < ch_a * 0.9:
                    if args.verbose:
                        print(f"[{scene_id}] [INFO] using flipped GT extrinsic (reduced chamfer {ch_a:.4f}->{ch_b:.4f})")
                    w2c_gt = w2c_gt_flip
            except Exception as e:
                if args.verbose:
                    print(f"[{scene_id}] [WARN] flip autotest failed: {e}")

        try:
            gt3d = unproject(gts[i],    K_gt_scaled, w2c_gt,   mask=m, verbose=args.verbose, scene_id=scene_id)
            pr3d = unproject(pdepth[i], K_pred,      w2c_pred, mask=m, verbose=args.verbose, scene_id=scene_id)
        except Exception as e:
            if args.verbose:
                print(f"[{scene_id}] unproject failed @frame {k}: {e}")
            continue

        if args.per_frame_align:
            if pr3d.shape[0] > 0 and gt3d.shape[0] > 0:
                R,t,s = umeyama_align(pr3d, gt3d, with_scale=True)
                pr3d_a = (s * (R @ pr3d.T).T) + t
                a,c,ch = chamfer_with_cap(pr3d_a, gt3d, cap=args.dist_cap_m)
                per_frame_scores.append((a,c,ch))

        if gt3d.shape[0] > 0: gt_pts_all.append(gt3d.astype(np.float64))
        if pr3d.shape[0] > 0: pred_pts_all.append(pr3d.astype(np.float64))

    # 결과 산출
    if args.per_frame_align:
        if not per_frame_scores:
            return None
        acc = float(np.mean([x[0] for x in per_frame_scores]))
        comp = float(np.mean([x[1] for x in per_frame_scores]))
        cham = float(np.mean([x[2] for x in per_frame_scores]))
        print(f"\nSCENE ({scene_id}) [per-frame]: acc={acc:.6f} comp={comp:.6f} chamfer={cham:.6f} "
              f"(frames={len(per_frame_scores)})")
        return {
            'scene_id': scene_id,
            'Chamfer_acc': acc,
            'Chamfer_comp': comp,
            'Chamfer': cham,
            'n_pred3d': -1, 'n_gt3d': -1,
            'cam_mode': 'per-frame align'
        }

    if not (pred_pts_all and gt_pts_all):
        if args.verbose:
            npreds = sum(len(x) for x in pred_pts_all)
            ngts   = sum(len(x) for x in gt_pts_all)
            print(f"[{scene_id}] not enough 3D points -> skip (pred_pts={npreds}, gt_pts={ngts})")
        return None

    pred = np.concatenate(pred_pts_all, 0)
    gt   = np.concatenate(gt_pts_all, 0)

    if pred.shape[0] > args.max_points:
        pred = pred[np.random.choice(pred.shape[0], args.max_points, replace=False)]
    if gt.shape[0] > args.max_points:
        gt   = gt[np.random.choice(gt.shape[0],   args.max_points, replace=False)]

    # ---------- [PATCH] Unit auto-fix (mm<->m) ----------
    if args.unit_autofix:
        pred_med = float(np.median(np.abs(pred)))
        gt_med   = float(np.median(np.abs(gt)))
        if args.verbose:
            print(f"[{scene_id}] [UNIT CHECK] |pred|_med={pred_med:.4f}, |gt|_med={gt_med:.4f}")
        if gt_med > 100.0 and pred_med < 10.0 and (gt_med / (pred_med + 1e-9) > 50.0):
            if args.verbose: print(f"[{scene_id}] [UNIT FIX] GT mm->m (/1000)")
            gt = gt / 1000.0
            gt_med = float(np.median(np.abs(gt)))
        if pred_med > 100.0 and gt_med < 10.0 and (pred_med / (gt_med + 1e-9) > 50.0):
            if args.verbose: print(f"[{scene_id}] [UNIT FIX] PRED mm->m (/1000)")
            pred = pred / 1000.0

    # ---------- Alignment (with scale) ----------
    try:
        R, t, s = umeyama_align(pred, gt, with_scale=True)
        pred_aligned = (s * (R @ pred.T).T) + t
    except Exception:
        pred_aligned = pred

    # ---------- Chamfer with optional cap ----------
    acc, comp, cham = chamfer_with_cap(pred_aligned, gt, cap=args.dist_cap_m)

    print(f"\nSCENE ({scene_id}): acc={acc:.6f} comp={comp:.6f} chamfer={cham:.6f} "
          f"(n_pred3d={pred.shape[0]} n_gt3d={gt.shape[0]})")

    return {
        'scene_id': scene_id,
        'Chamfer_acc': acc,
        'Chamfer_comp': comp,
        'Chamfer': cham,
        'n_pred3d': int(pred.shape[0]),
        'n_gt3d': int(gt.shape[0]),
        'cam_mode': 'pred(pred) vs gt(gt)'
    }

# --------------------- Main ---------------------
def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # discover scenes
    scene_ids = discover_scenes(args.bms_root) if not args.scene_ids else args.scene_ids
    if args.max_scenes: scene_ids = scene_ids[:args.max_scenes]
    if not scene_ids: raise RuntimeError('No valid scenes under bms_root')

    # load model
    device = args.device
    if args.model_path.startswith('facebook/') and not osp.exists(args.model_path):
        model = VGGT.from_pretrained(args.model_path).to(device)
    else:
        model = VGGT()
        state = torch.load(args.model_path, map_location='cpu')
        try: model.load_state_dict(state)
        except Exception: model.load_state_dict(state, strict=False)
        model.to(device)
    model.eval()

    # csv
    writer = None; fh = None
    if args.save_csv:
        exists = osp.exists(args.save_csv)
        fh = open(args.save_csv, 'a', newline='')
        fieldnames = ['scene_id','Chamfer_acc','Chamfer_comp','Chamfer','n_pred3d','n_gt3d','cam_mode']
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists: writer.writeheader()

    # per-scene eval
    rows = []
    try:
        for sid in scene_ids:
            try:
                r = evaluate_scene(args, model, sid)
                if r is None:
                    if args.verbose: print(f"[{sid}] skipped (no pairs/poses/points)")
                    continue
                rows.append(r)
                if writer: writer.writerow(r)
            except Exception as e:
                print(f"[ERROR] Failed scene {sid}: {e}")
    finally:
        if fh is not None: fh.close()

    if not rows:
        print("\nNo Chamfer results produced.")
        return

    # global averages
    def smean(vals):
        arr = np.array([v for v in vals if v is not None and np.isfinite(v)])
        return float(arr.mean()) if arr.size else float('nan')

    gacc  = smean([r['Chamfer_acc'] for r in rows])
    gcomp = smean([r['Chamfer_comp'] for r in rows])
    gch   = smean([r['Chamfer'] for r in rows])

    print("\nGLOBAL (Chamfer-only) SUMMARY over {} scenes:".format(len(rows)))
    print("  acc={:.6f} comp={:.6f} chamfer={:.6f}".format(gacc, gcomp, gch))

if __name__ == '__main__':
    main()
