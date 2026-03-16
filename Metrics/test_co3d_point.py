#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CO3Dv2 point-map eval with VGGT

기능 요약
- tqdm 진행바 (--no_tqdm로 끌 수 있음)
- I/O 프리패치 (--prefetch_io)
- world_points만 사용할지 강제 (--require_world_points) → depth unproject 경로 생략 가능
- 경로 해석 간소화 (--trust_filepath) → glob 최소화
- GT 탐색 디버그 (--debug_gt)
- 리사이즈 (--resize_short): 짧은 변 기준, intrinsics 자동 보정
- 패치 배수 패딩 (--pad_to_patch, --patch_size): 입력 H,W를 patch_size 배수로 오른쪽/아래만 0-패딩
"""

import argparse
import gzip
import json
import os
import os.path as osp
import random
import glob as _glob
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

# ---- 프로젝트 의존 ----
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# torchvision (리사이즈용)
try:
    import torchvision.transforms.functional as TF
except Exception:
    TF = None


# --------------------- CLI ---------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--co3d_dir', required=True, help='이미지 루트 (jgz filepath는 이 경로 기준)')
    p.add_argument('--co3d_anno_dir', default='.', help='jgz 폴더')
    p.add_argument('--model_path', default='D:/CVPR_2026/VGGT_eval/Camera_pose/model_tracker_fixed_e20.pt')
    p.add_argument('--category', required=True, help='jgz prefix (예: merged→ merged_test.jgz)')
    p.add_argument('--sequence', default=None, help='특정 시퀀스만 평가')
    p.add_argument('--num_frames', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--min_num_images', type=int, default=1, help='이 수 미만이면 시퀀스 스킵')
    p.add_argument('--depth_conf_thresh', type=float, default=0.2)
    p.add_argument('--point_conf_thresh', type=float, default=0.2)
    p.add_argument('--use_pointmap_branch', action='store_true')
    p.add_argument('--max_points', type=int, default=500000)
    p.add_argument('--gt_ply', type=str, default=None)
    p.add_argument('--gt_dir', type=str, default=None)
    p.add_argument('--gt_roots', type=str, default=None,
                   help='여러 GT 루트는 세미콜론/콤마로 구분: "root1;root2,root3"')
    p.add_argument('--dump_preds_shapes', action='store_true')
    p.add_argument('--debug_gt', action='store_true')

    # 속도/편의
    p.add_argument('--prefetch_io', action='store_true', help='이미지 파일을 미리 조금 읽어서 디스크 워밍업')
    p.add_argument('--require_world_points', action='store_true',
                   help='world_points가 없는 경우 해당 시퀀스/프레임 스킵 (depth unproject 생략)')
    p.add_argument('--trust_filepath', action='store_true',
                   help='jgz의 filepath를 그대로 신뢰(존재 확인만 하고 glob 탐색 안 함)')

    # tqdm
    p.add_argument('--no_tqdm', action='store_true', help='진행바 끄기')

    # 리사이즈
    p.add_argument('--resize_short', type=int, default=None,
                   help='짧은 변을 이 값으로 리사이즈(종횡비 유지). intrinsics 자동 보정')

    # 패치 배수 패딩
    p.add_argument('--pad_to_patch', action='store_true',
                   help='입력 이미지 H,W를 패치 크기 배수로 오른쪽/아래 패딩')
    p.add_argument('--patch_size', type=int, default=14,
                   help='VGGT patch size (기본 14)')

    return p.parse_args()


# ----------------- 유틸 함수들 -----------------
def _split_multi_paths(s):
    if s is None:
        return []
    return [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]


def resolve_path_try_exts(base_dir, rel_path, try_exts=None):
    """확장자 애매할 때 여러 확장자/이름으로 탐색."""
    if try_exts is None:
        try_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    p = osp.normpath(osp.join(base_dir, rel_path))
    if osp.exists(p):
        return p
    base_noext = osp.splitext(p)[0]
    for ext in try_exts:
        cand = base_noext + ext
        if osp.exists(cand):
            return cand
    d = osp.dirname(p)
    basename = osp.splitext(osp.basename(p))[0].lower()
    if osp.exists(d):
        for f in os.listdir(d):
            if f.lower().startswith(basename):
                cand = osp.join(d, f)
                if osp.exists(cand):
                    return cand
    return None


def resolve_path_fast(base_dir, rel_path):
    """glob 없이 경로 그대로 존재하는지만 확인."""
    p = osp.normpath(osp.join(base_dir, rel_path))
    return p if osp.exists(p) else None


def load_annotation(anno_file):
    with gzip.open(anno_file, 'r') as f:
        return json.loads(f.read())


# ------------------- GT 찾기 -------------------
def find_gt_ply_path(seq_name, args):
    # args.gt_dir가 H:\CVPR_final_dataset\Datasets\CO3Dv2\GT 라고 가정
    # seq_name이 곧 scene_id인 경우
    if args.gt_dir and args.category:
        cand = osp.join(args.gt_dir, args.category, seq_name, "pointcloud.ply")
        if osp.exists(cand):
            if args.debug_gt:
                print(f"[DEBUG] GT Found: {cand}")
            return cand
    
    # 예외 상황을 대비해 기존 로직을 fallback으로 유지
    if args.debug_gt:
        print(f"[DEBUG] Direct path failed, searching in {args.gt_dir} for {seq_name}")

    def _search_in_root(root):
        # 여러 패턴 순차 탐색
        patterns = [
            osp.join(root, "**", seq_name, "pointcloud.ply"),
            osp.join(root, "**", seq_name, "*.ply"),
            osp.join(root, "**", f"{seq_name}.ply"),
            osp.join(root, "**", seq_name, "pointclouds.ply"),
            osp.join(root, "**", f"{seq_name}*", "pointcloud.ply"),
            osp.join(root, "**", f"{seq_name}*", "*.ply"),
            osp.join(root, "**", f"*{seq_name}*pointcloud*.ply"),
        ]
        for pat in patterns:
            hits = _glob.glob(pat, recursive=True)
            if args.debug_gt:
                print(f"[DEBUG]  glob {pat} -> {len(hits)}")
                if len(hits) > 0:
                    print("        sample:", hits[0])
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                print(f"[WARN] multiple GT matches for {seq_name} with pattern {pat}, picking the first")
                return hits[0]
        return None

    # gt_dir부터
    if args.gt_dir:
        cand1 = osp.join(args.gt_dir, f"{seq_name}.ply")
        if osp.exists(cand1):
            if args.debug_gt:
                print(f"[DEBUG] use exact {cand1}")
            return cand1
        found = _search_in_root(args.gt_dir)
        if found:
            return found
        try:
            files = [f for f in os.listdir(args.gt_dir) if f.lower().endswith('.ply')]
            if len(files) == 1:
                return osp.join(args.gt_dir, files[0])
        except Exception:
            pass

    # gt_roots 여러 개
    for root in _split_multi_paths(args.gt_roots):
        found = _search_in_root(root)
        if found:
            return found

    return None



# ----------------- 정렬/메트릭 -----------------
def umeyama_align(src, dst, with_scaling=True):
    """
    src, dst: (N,3)
    크기 다르면 최근접 이웃으로 대응 맞춰서 Umeyama.
    """
    src = np.asarray(src).astype(np.float64)
    dst = np.asarray(dst).astype(np.float64)
    assert src.ndim == 2 and dst.ndim == 2 and src.shape[1] == 3 and dst.shape[1] == 3
    if src.shape[0] == 0 or dst.shape[0] == 0:
        raise ValueError("Empty source or destination in umeyama_align")

    if src.shape[0] != dst.shape[0]:
        tree = cKDTree(dst)
        _, idx = tree.query(src, k=1)
        dst_paired = dst[idx]
        src_paired = src
    else:
        src_paired = src
        dst_paired = dst

    N = src_paired.shape[0]
    mu_src = src_paired.mean(axis=0)
    mu_dst = dst_paired.mean(axis=0)
    src_c = src_paired - mu_src
    dst_c = dst_paired - mu_dst
    cov = (dst_c.T @ src_c) / N
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scaling:
        var_src = (src_c ** 2).sum() / N
        s = np.trace(np.diag(D) @ S) / var_src
    else:
        s = 1.0
    t = mu_dst - s * R @ mu_src
    return R, t, s


def load_ply_xyz(ply_path):
    # PLY 로더 (plyfile 우선, 실패 시 수동 파서)
    try:
        from plyfile import PlyData
        plydata = PlyData.read(ply_path)
        if 'vertex' in plydata.elements:
            v = plydata['vertex'].data
            if all(k in v.dtype.names for k in ('x', 'y', 'z')):
                pts = np.vstack([v['x'], v['y'], v['z']]).T
                return pts.astype(np.float64)
            else:
                names = v.dtype.names
                if len(names) >= 3:
                    pts = np.vstack([v[names[0]], v[names[1]], v[names[2]]]).T
                    return pts.astype(np.float64)
        raise RuntimeError("plyfile read but no vertex/x,y,z found")
    except Exception:
        pass

    # 수동 파서
    with open(ply_path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF while reading PLY header")
            try:
                sline = line.decode('ascii').strip()
            except Exception:
                sline = line.decode('latin1').strip()
            header_lines.append(sline)
            if sline.lower() == 'end_header':
                break

        format_line = None
        vertex_count = None
        in_vertex = False
        vertex_props = []
        for ln in header_lines:
            parts = ln.split()
            if len(parts) == 0:
                continue
            if parts[0] == 'format':
                format_line = parts[1]
            elif parts[0] == 'element' and parts[1] == 'vertex':
                vertex_count = int(parts[2])
                in_vertex = True
            elif parts[0] == 'element':
                in_vertex = False
            elif parts[0] == 'property' and in_vertex:
                if len(parts) == 3:
                    ptype, pname = parts[1], parts[2]
                    vertex_props.append((ptype, pname))
        if vertex_count is None or len(vertex_props) == 0:
            raise RuntimeError("No vertex element/properties found in PLY header")
        prop_names = [pn for _, pn in vertex_props]
        try:
            ix = prop_names.index('x')
            iy = prop_names.index('y')
            iz = prop_names.index('z')
        except ValueError:
            ix, iy, iz = 0, 1, 2
        type_map = {
            'char': ('b', 1), 'uchar': ('B', 1), 'int8': ('b', 1), 'uint8': ('B', 1),
            'short': ('h', 2), 'ushort': ('H', 2), 'int16': ('h', 2), 'uint16': ('H', 2),
            'int': ('i', 4), 'int32': ('i', 4), 'uint': ('I', 4), 'uint32': ('I', 4),
            'float': ('f', 4), 'float32': ('f', 4), 'double': ('d', 8), 'float64': ('d', 8)
        }
        fmt_chars = []
        byte_len = 0
        for ptype, _ in vertex_props:
            ch, sz = type_map[ptype.lower()]
            fmt_chars.append(ch)
            byte_len += sz
        if format_line is None:
            raise RuntimeError("PLY format not specified")
        if 'binary_little_endian' in format_line:
            endian = '<'
        elif 'binary_big_endian' in format_line:
            endian = '>'
        elif 'ascii' in format_line:
            endian = 'ascii'
        else:
            raise RuntimeError("Unknown PLY format: " + format_line)

        if endian == 'ascii':
            pts = []
            for _ in range(vertex_count):
                line = f.readline().decode('ascii').strip()
                parts = line.split()
                if len(parts) < 3:
                    continue
                x = float(parts[ix])
                y = float(parts[iy])
                z = float(parts[iz])
                pts.append([x, y, z])
            return np.array(pts, dtype=np.float64)

        data = f.read(vertex_count * byte_len)
        if len(data) < vertex_count * byte_len:
            raise RuntimeError("PLY binary data truncated")
        import struct as _struct
        unpack = _struct.unpack_from
        pts = []
        offset = 0
        struct_fmt = endian + ''.join(fmt_chars)
        for _ in range(vertex_count):
            vals = unpack(struct_fmt, data, offset)
            offset += byte_len
            x = float(vals[ix])
            y = float(vals[iy])
            z = float(vals[iz])
            pts.append([x, y, z])
        return np.array(pts, dtype=np.float64)


def chamfer_metrics(pred_pts, gt_pts):
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.inf, np.inf, np.inf
    tree_gt = cKDTree(gt_pts)
    d_pred_to_gt, _ = tree_gt.query(pred_pts, k=1)
    tree_pred = cKDTree(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts, k=1)
    acc = float(d_pred_to_gt.mean())
    comp = float(d_gt_to_pred.mean())
    chamfer = 0.5 * (acc + comp)
    return acc, comp, chamfer


# ----------------- 텐서 헬퍼 -----------------
def tensor_frame_to_map(tensor, frame_idx, batch_size):
    if tensor is None:
        return None
    arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    if arr.ndim == 5:
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            out = arr[0, frame_idx]
            if out.shape[-1] == 1:
                out = np.squeeze(out, axis=-1)
            return out
        if arr.shape[0] == 1:
            a2 = arr.squeeze(0)
            if a2.shape[0] > frame_idx:
                out = a2[frame_idx]
                if out.shape[-1] == 1:
                    out = np.squeeze(out, axis=-1)
                return out
            return np.squeeze(a2)
    if arr.ndim == 4:
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            out = arr[0, frame_idx]
            if out.ndim == 3 and out.shape[-1] == 1:
                out = np.squeeze(out, axis=-1)
            return out
        if arr.shape[0] > 1 and arr.shape[0] == frame_idx + 1:
            return arr[frame_idx]
        return np.squeeze(arr)
    if arr.ndim == 3:
        if arr.shape[0] == frame_idx + 1 or arr.shape[0] == frame_idx:
            return arr[frame_idx]
        if arr.shape[0] == 1:
            return np.squeeze(arr, axis=0)
        return arr
    if arr.ndim == 2:
        return arr
    return np.squeeze(arr)


def get_frame_from_tensor(tensor, idx, batch_n, as_torch=True, device=None, dtype=None):
    if tensor is None:
        return None
    try:
        if isinstance(tensor, torch.Tensor):
            t = tensor
            nd = t.dim()
            if nd >= 3 and t.shape[0] == 1 and t.shape[1] > 1:
                sel = t[0, idx]
            elif nd >= 3 and t.shape[0] == 1 and t.shape[1] == batch_n:
                sel = t[0, idx]
            elif nd >= 3 and t.shape[0] == batch_n:
                sel = t[idx]
            elif nd >= 2 and t.shape[0] == 1:
                sel = t.squeeze(0)
                if sel.dim() >= 1 and sel.shape[0] > 1 and sel.shape[0] == batch_n:
                    sel = sel[idx]
            else:
                sel = t
            if as_torch and device is not None:
                sel = sel.to(device=device, dtype=dtype) if dtype is not None else sel.to(device=device)
            return sel
        else:
            arr = np.asarray(tensor)
            m = tensor_frame_to_map(arr, idx, batch_n)
            if m is None:
                return None
            if as_torch:
                t = torch.from_numpy(m)
                if device is not None:
                    t = t.to(device=device)
                if dtype is not None:
                    t = t.to(dtype=dtype)
                return t
            return m
    except Exception:
        return None


# ----------------- I/O 프리패치 -----------------
def prefetch_files(paths, chunk=65536):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=min(8, max(1, os.cpu_count() or 4))) as ex:
        futs = [ex.submit(_touch_read, p, chunk) for p in paths]
        for _ in as_completed(futs):
            pass


def _touch_read(path, chunk):
    try:
        with open(path, 'rb') as f:
            _ = f.read(chunk)
    except Exception:
        pass


# ----------------- 리사이즈/패딩 -----------------
def _resize_batch_keep_aspect(imgs_tensor, target_short):
    """
    imgs_tensor: (B,C,H,W)
    target_short: int (짧은 변)
    반환: imgs_resized, scales(list of (sx, sy)), new_size(H_new, W_new)
    """
    if target_short is None:
        H, W = imgs_tensor.shape[-2:]
        return imgs_tensor, [(1.0, 1.0)] * imgs_tensor.shape[0], (H, W)

    if TF is None:
        raise RuntimeError("torchvision이 필요합니다. (transform.functional)")

    B, C, H0, W0 = imgs_tensor.shape
    short0 = min(H0, W0)
    if short0 == 0:
        raise ValueError("Invalid image size")

    scale = float(target_short) / float(short0)
    Hn, Wn = int(round(H0 * scale)), int(round(W0 * scale))

    imgs_out = []
    scales = []
    for i in range(B):
        t = TF.resize(imgs_tensor[i], [Hn, Wn], antialias=True)
        imgs_out.append(t)
        scales.append((Wn / W0, Hn / H0))  # (sx, sy)
    return torch.stack(imgs_out, dim=0), scales, (Hn, Wn)


def _pad_to_patch_multiple(imgs_tensor, patch_size: int):
    """
    imgs_tensor: (B, C, H, W)
    오른쪽/아래로만 0-패딩해서 H,W를 patch_size의 배수로 맞춤.
    반환: imgs_padded, (pad_right, pad_bottom)
    """
    B, C, H, W = imgs_tensor.shape
    need_H = (patch_size - (H % patch_size)) % patch_size
    need_W = (patch_size - (W % patch_size)) % patch_size
    if need_H == 0 and need_W == 0:
        return imgs_tensor, (0, 0)
    # pad 순서: (left, right, top, bottom)
    imgs_pad = F.pad(imgs_tensor, (0, need_W, 0, need_H), mode="constant", value=0.0)
    return imgs_pad, (need_W, need_H)


# ----------------------- 메인 -----------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device.startswith('cuda'):
        try:
            cc = torch.cuda.get_device_capability()[0]
        except Exception:
            cc = 8
        dtype = torch.bfloat16 if cc >= 8 else torch.float16
    else:
        dtype = torch.float32

    print("Loading model:", args.model_path)
    if args.model_path.startswith("facebook/") or (('/' in args.model_path or '\\' in args.model_path) and not osp.exists(args.model_path)):
        model = VGGT.from_pretrained(args.model_path).to(device)
    else:
        model = VGGT()
        state = torch.load(args.model_path, map_location='cpu')
        try:
            model.load_state_dict(state)
        except Exception:
            model.load_state_dict(state, strict=False)
        model.to(device)
    model.eval()

    anno_file = osp.join(args.co3d_anno_dir, f"{args.category}_test.jgz")
    if not osp.exists(anno_file):
        raise FileNotFoundError(f"Annotation file not found: {anno_file}")
    annotation = load_annotation(anno_file)

    seq_names = sorted(list(annotation.keys()))
    if args.sequence is not None:
        if args.sequence not in annotation:
            raise ValueError(f"Sequence {args.sequence} not found")
        seq_names = [args.sequence]

    if args.gt_ply is None and args.gt_dir is None and args.gt_roots is None:
        print("[WARN] No GT provided (gt_ply/gt_dir/gt_roots). Nothing to evaluate.")
        return

    results = {}
    seq_iter = tqdm(seq_names, desc="Sequences", disable=args.no_tqdm)

    for seq_name in seq_iter:
        seq_iter.set_postfix_str(seq_name)
        print("Processing", seq_name)

        seq_data = annotation[seq_name]
        if len(seq_data) < args.min_num_images:
            print(f" skip (only {len(seq_data)} images < min {args.min_num_images})")
            continue

        # 프레임 샘플링
        if len(seq_data) <= args.num_frames:
            chosen = list(range(len(seq_data)))
        else:
            chosen = sorted(np.random.choice(len(seq_data), args.num_frames, replace=False).tolist())
        print(" chosen indices:", chosen)

        # GT 탐색
        gt_path = find_gt_ply_path(seq_name, args)
        if gt_path is None:
            print(f"  [WARN] GT not found for sequence {seq_name} -> skip")
            continue
        gt_pts = load_ply_xyz(gt_path)

        # 이미지 경로 해석
        resolver = resolve_path_fast if args.trust_filepath else resolve_path_try_exts
        image_paths, entries = [], []
        
        for idx in chosen:
            entry = seq_data[idx]
            
            # 1. 파일 이름만 추출 (예: 'frame001.jpg')
            fname = osp.basename(entry['filepath'])
            
            # 2. CMQE 경로와 결합: D:\CVPR_2026\CMQE_v3\CO3Dv2\{scene_id}\{fname}
            img_path = osp.join(args.co3d_dir, seq_name, fname)
            
            if osp.exists(img_path):
                image_paths.append(img_path)
                entries.append(entry)
            else:
                # 확장자를 바꿔서 재시도 (.png <-> .jpg <-> .jpeg 등)
                base_noext = osp.splitext(img_path)[0]
                alt_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
                found_alt = None
                for ext in alt_exts:
                    cand = base_noext + ext
                    if osp.exists(cand):
                        found_alt = cand
                        break

                if found_alt is None:
                    # CMQE 과정에서 하위 폴더(images/)가 생성됐을 가능성 체크
                    base_noext_sub = osp.splitext(
                        osp.join(args.co3d_dir, seq_name, "images", fname)
                    )[0]
                    for ext in alt_exts:
                        cand = base_noext_sub + ext
                        if osp.exists(cand):
                            found_alt = cand
                            break

                if found_alt is not None:
                    image_paths.append(found_alt)
                    entries.append(entry)
                else:
                    print(f"  [WARN] Image not found: {img_path}")

        if len(image_paths) == 0:
            print(f"  [SKIP] {seq_name}: No valid images found in {args.co3d_dir}")
            continue

        # 프리패치
        if args.prefetch_io:
            prefetch_files(image_paths)

        # 이미지 로드
        imgs_tensor = load_and_preprocess_images(list(image_paths)).to(device)
        H0, W0 = imgs_tensor.shape[-2:]

        # 리사이즈 (짧은 변 기준)
        resize_scales = [(1.0, 1.0)] * imgs_tensor.shape[0]
        if args.resize_short is not None:
            imgs_tensor, resize_scales, (Hn, Wn) = _resize_batch_keep_aspect(imgs_tensor, args.resize_short)
        else:
            Hn, Wn = H0, W0

        # 패치 배수 패딩 (오른쪽/아래만 → cx,cy 변하지 않으므로 intrinsics 추가 보정 불필요)
        if args.pad_to_patch:
            imgs_tensor, (pad_w, pad_h) = _pad_to_patch_multiple(imgs_tensor, args.patch_size)
            Hn, Wn = imgs_tensor.shape[-2:]
        else:
            pad_w, pad_h = 0, 0

        # 추론
        with torch.no_grad():
            if device.startswith('cuda'):
                with torch.amp.autocast(device_type='cuda', dtype=dtype):
                    preds = model(imgs_tensor)
            else:
                preds = model(imgs_tensor)

        if args.dump_preds_shapes:
            print("PRED KEYS & SHAPES:")
            if isinstance(preds, dict):
                for k, v in preds.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: torch{tuple(v.shape)} {v.dtype}")
                    else:
                        try:
                            a = np.asarray(v)
                            print(f"  {k}: np{tuple(a.shape)} {a.dtype}")
                        except Exception:
                            print(f"  {k}: type {type(v)}")

        # world_points 강제 모드
        world_points_tensor = preds.get('world_points') if isinstance(preds, dict) and 'world_points' in preds else None
        world_points_conf = preds.get('world_points_conf') if isinstance(preds, dict) and 'world_points_conf' in preds else None
        if args.require_world_points and (world_points_tensor is None):
            print("  [SKIP] no world_points in preds and --require_world_points is set")
            continue

        # 보조 텐서
        depth_tensor = preds.get('depth') if isinstance(preds, dict) and 'depth' in preds else None
        depth_conf_tensor = preds.get('depth_conf') if isinstance(preds, dict) and 'depth_conf' in preds else None
        point_map_tensor = preds.get('point_map') if isinstance(preds, dict) and 'point_map' in preds else None
        point_conf_tensor = preds.get('point_conf') if isinstance(preds, dict) and 'point_conf' in preds else None

        # pose_enc → extri, intri (여기서는 패딩/리사이즈 후 최종 해상도 사용)
        extrinsic_all = None
        intrinsic_all = None
        if isinstance(preds, dict) and 'pose_enc' in preds:
            try:
                extrinsic_all, intrinsic_all = pose_encoding_to_extri_intri(preds['pose_enc'], (Hn, Wn))
            except Exception as e:
                print("  [WARN] pose_encoding_to_extri_intri failed:", e)
        if extrinsic_all is None and isinstance(preds, dict):
            for k in ['extrinsic', 'extri', 'cam_extri', 'camera']:
                if k in preds:
                    extrinsic_all = preds[k]
                    break

        pred_world_list = []

        frame_iter = tqdm(enumerate(entries), total=len(entries),
                          desc=f"Frames({seq_name})", leave=False, disable=args.no_tqdm)
        for i, entry in frame_iter:
            frame_idx = int(chosen[i])
            frame_iter.set_postfix_str(f"idx={frame_idx}")

            # ----- world_points branch (모델이 직접 world_points를 줄 경우) -----
            used_world_branch = False
            if world_points_tensor is not None:
                try:
                    wp = tensor_frame_to_map(world_points_tensor, i, imgs_tensor.shape[0])
                    wc = tensor_frame_to_map(world_points_conf, i, imgs_tensor.shape[0]) if world_points_conf is not None else None
                    if wp is not None:
                        wp = np.asarray(wp)
                        if wp.ndim == 3 and wp.shape[0] == 3:
                            pts = wp.reshape(3, -1).T
                        elif wp.ndim == 3 and wp.shape[2] == 3:
                            pts = wp.reshape(-1, 3)
                        else:
                            pts = np.reshape(wp, (-1, 3))
                        if wc is not None:
                            wc = np.asarray(wc)
                            if wc.ndim == 3 and wc.shape[-1] == 1:
                                wc = wc[:, :, 0]
                            flat_conf = wc.reshape(-1)
                            if flat_conf.shape[0] != pts.shape[0]:
                                minlen = min(flat_conf.shape[0], pts.shape[0])
                                pts = pts[:minlen]
                                flat_conf = flat_conf[:minlen]
                            pts = pts[flat_conf >= args.point_conf_thresh]
                        
                        if pts.shape[0] > 0:
                            pred_world_list.append(pts.astype(np.float64))
                            used_world_branch = True
                except Exception as e:
                    print(f"   [WARN] world_points branch failed for frame {frame_idx}: {e}")

            if args.require_world_points:
                if not used_world_branch:
                    print(f"   [SKIP] frame {frame_idx}: no world_points under --require_world_points")
                continue 

            # ----- fallback: depth + pose → 직접 좌표 계산 (world_points가 없을 경우) -----
            if not used_world_branch:
                try:
                    # 1. Extrinsic & Intrinsic 준비
                    extr = get_frame_from_tensor(extrinsic_all, i, imgs_tensor.shape[0], as_torch=True, device=device)
                    if extr is not None and extr.shape == (4, 4): 
                        extr = extr[:3, :4]
                    
                    intr = get_frame_from_tensor(intrinsic_all, i, imgs_tensor.shape[0], as_torch=True, device=device)
                    
                    if extr is None or intr is None:
                        print(f"   [WARN] No pose (extr/intr) for frame {frame_idx}")
                        continue

                    # 2. Depth 추출 및 2D 평면화 [H, W]
                    d_raw = depth_tensor[0, i] if (depth_tensor is not None and depth_tensor.ndim >= 4) else (depth_tensor[i] if depth_tensor is not None else None)
                    if d_raw is None:
                        continue
                        
                    d_2d = d_raw.clone()
                    for _ in range(d_2d.ndim - 2):
                        if d_2d.shape[0] == 1: d_2d = d_2d[0]
                        elif d_2d.shape[-1] == 1: d_2d = d_2d[..., 0]

                    H, W = d_2d.shape
                    
                    # 3. 픽셀 그리드 생성
                    y, x = torch.meshgrid(
                        torch.arange(H, device=device, dtype=d_2d.dtype),
                        torch.arange(W, device=device, dtype=d_2d.dtype),
                        indexing='ij'
                    )
                    
                    # 4. 카메라 좌표계로 변환
                    fx, fy = intr[0, 0], intr[1, 1]
                    cx, cy = intr[0, 2], intr[1, 2]
                    z = d_2d
                    x_c = (x - cx) * z / fx
                    y_c = (y - cy) * z / fy
                    cam_pts_map = torch.stack([x_c, y_c, z], dim=-1)
                    
                    # 5. 유효성 마스크 적용
                    cd_raw = depth_conf_tensor[0, i] if (depth_conf_tensor is not None and depth_conf_tensor.ndim >= 4) else (depth_conf_tensor[i] if depth_conf_tensor is not None else None)
                    if cd_raw is not None:
                        cd_2d = cd_raw.clone()
                        for _ in range(cd_2d.ndim - 2):
                            if cd_2d.shape[0] == 1: cd_2d = cd_2d[0]
                            elif cd_2d.shape[-1] == 1: cd_2d = cd_2d[..., 0]
                        mask = (cd_2d >= args.depth_conf_thresh) & (z > 0)
                    else:
                        mask = (z > 0)

                    curr_cam_pts = cam_pts_map[mask]
                    
                    if curr_cam_pts.shape[0] > 0:
                        # 6. 카메라 -> 월드 변환 (R_inv * (X_c - t))
                        # CO3D/VGGT의 extrinsic (extr)은 World-to-Camera (W2C)입니다.
                        # W2C: X_c = R*X_w + t  =>  X_w = R^T * (X_c - t)
                        R = extr[:3, :3]
                        t = extr[:3, 3]
                        
                        # (N, 3) - (3,) -> (N, 3). 그 후 R^T (transpose) 곱함
                        curr_world_pts = torch.matmul(curr_cam_pts - t, R) 
                        pred_world_list.append(curr_world_pts.detach().cpu().numpy().astype(np.float64))
                except Exception as e:
                    print(f"   [WARN] Manual unproject failed for frame {frame_idx}: {e}")
                    continue

        # --- 중요: 이 아래에 있던 Rm, t, Rinv, pts_world 관련 중복 코드는 삭제되었습니다 ---

        if len(pred_world_list) == 0:
            print("  no predicted points for sequence -> skip")
            continue

        pred_world_all = np.concatenate(pred_world_list, axis=0)
        if pred_world_all.shape[0] > args.max_points:
            idxs = np.random.choice(pred_world_all.shape[0], args.max_points, replace=False)
            pred_world_all = pred_world_all[idxs]

        # Umeyama + Chamfer
        R_u, t_u, s_u = umeyama_align(pred_world_all, gt_pts)
        pred_aligned = (s_u * (R_u @ pred_world_all.T).T) + t_u
        acc, comp, chamfer = chamfer_metrics(pred_aligned, gt_pts)
        results[seq_name] = {
            'accuracy': acc,
            'completeness': comp,
            'chamfer': chamfer,
            'n_pred': pred_world_all.shape[0],
            'n_gt': gt_pts.shape[0],
        }
        seq_iter.set_postfix(acc=f"{acc:.3f}", comp=f"{comp:.3f}", ch=f"{chamfer:.3f}")

    # 전체 요약
    if len(results) > 0:
        accs = np.array([v['accuracy'] for v in results.values() if np.isfinite(v['accuracy'])])
        comps = np.array([v['completeness'] for v in results.values() if np.isfinite(v['completeness'])])
        chs = np.array([v['chamfer'] for v in results.values() if np.isfinite(v['chamfer'])])
        print("\nSUMMARY:")
        if len(accs):
            print(f" mean accuracy: {np.mean(accs):.4f}")
        if len(comps):
            print(f" mean completeness: {np.mean(comps):.4f}")
        if len(chs):
            print(f" mean chamfer: {np.mean(chs):.4f}")
    else:
        print("No results computed.")


if __name__ == '__main__':
    main()