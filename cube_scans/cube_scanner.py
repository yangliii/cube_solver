#!/usr/bin/env python3
"""魔方颜色扫描仪：用摄像头逐面识别魔方 3x3 格颜色并保存。

用法:
    python cube_scanner.py                # 摄像头 0，结果保存到脚本所在目录
    python cube_scanner.py --out my_dir   # 自定义保存目录
    python cube_scanner.py --debug        # 额外打印每格 HSV 原始值

操作:
    1. 把魔方一个面朝向摄像头，不用对准任何框：程序会自动找到魔方、
       自动摆正对齐。让整个面完整出现在画面里、离得不太远即可；
    2. 画面里会自动框出识别到的魔方面并叠加每格识别结果；绿框保持稳定约半秒
       就会自动采集（不用按任何键），也可以随时按 空格 立即采集；
    3. 采集成功后画面会提示转动魔方扫描下一个面；如果同一面又拍了一次，
       程序会识别出来并提示，用新结果更新那一面，不会重复占用面数；
    4. 采满 6 面自动保存；已采满后重拍任一面会更新它；
    5. r = 撤销最近采集的面, q / Esc = 退出（已采集的面退出时也会保存）。
    6. 若网页导入时上次结果没通过校验（或没采满 6 面），再次启动本程序时
       会提示按 Y 载入上次的数据继续修正，并指出哪几个面有问题：只重拍
       这几个面即可（重拍会自动更新对应面）；按 N 忽略、从头扫描。

颜色代号: W=白 Y=黄 R=红 O=橙 B=蓝 G=绿 ?=无法确定

保存内容: cube_<时间戳>.json（颜色矩阵 + 原始采样值）
          cube_<时间戳>_faceN.png（每面透视校正后的对齐标注图）
          默认都存到脚本所在目录（cube_scans/），可用 --out 覆盖
采满 6 面保存后会自动清理目录里上一次扫描遗留的 cube_*.json 与
cube_*_face*.png，只保留本次结果；未采满 6 面就退出时不动旧文件。
"""

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

WARP_SIDE = 480           # 透视校正后魔方面的正方形边长（像素）
SAMPLES_PER_CAPTURE = 5   # 每次采集取多帧中值，降低手抖/噪点影响
AUTO_CAPTURE_FRAMES = 15      # 绿框连续稳定这么多帧后自动采集（约半秒）
AUTO_CAPTURE_MAX_SHIFT = 28   # 稳定判定：角点相对这轮稳定起始位置的允许偏移（像素）
AUTO_CAPTURE_CLEAR_FRAMES = 4  # 采集后绿框要先消失这么多帧，才允许再次自动采集
STICKER_MIN_AREA = 800    # 单个贴纸最小像素面积（约 28px 边长，再小识别不可靠）

COLOR_NAMES = {"W": "白", "Y": "黄", "R": "红", "O": "橙", "B": "蓝", "G": "绿", "?": "?"}


def classify_color(bgr) -> str:
    """把一格的采样颜色分类为 W/Y/R/O/B/G；拿不准时返回 '?'。

    规则基于 OpenCV HSV（H 0-179）：低饱和度亮色为白，其余按色相分段。
    红/橙在本相机+光照下色相几乎重叠（都落在 0-8），改用 Lab 色相角区分，
    它对反光/曝光变化比 RGB 比例（如 G/R）稳定；黄/绿分界取两者实测
    色相（约 39 与 71）中间值。
    """
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    if v < 45:                  # 太暗，无法判断
        return "?"
    if s < 50:                  # 低饱和度
        return "W" if v > 110 else "?"
    if h <= 25:                 # 红/橙色相区，用 Lab 色相角细分
        lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0]
        hue_ab = math.degrees(math.atan2(int(lab[2]) - 128, int(lab[1]) - 128))
        return "R" if hue_ab < 38 else "O"
    if h >= 165:                # 色相环另一端的红
        return "R"
    if h <= 55:
        return "Y"
    if h <= 95:
        return "G"
    if h <= 140:
        return "B"
    return "?"


# ---------------- 自动找魔方面 ----------------
# 思路：贴纸之间有黑色塑料分隔线，每张贴纸是一个独立、近似正方形的色块。
# 找出候选贴纸块，再从里面挑出 9 个中心构成 3x3 网格的组合，用单应变换
# 拟合网格 → 得到魔方面的四边形角点；贴纸粘连/线看不清时退化为「整面当
# 一大块色块」的兜底检测。

# 理想 3x3 网格：中心贴纸在 (1.5,1.5)，相邻贴纸中心距离为 1
_GRID_IDEAL = np.array([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5],
                        [0.5, 1.5], [1.5, 1.5], [2.5, 1.5],
                        [0.5, 2.5], [1.5, 2.5], [2.5, 2.5]], np.float32)
# 环绕中心的 8 个理想位置，按围绕中心的方向角（y 向下、从 0° 起）排序
_RING_IDEAL = np.array([[2.5, 1.5], [2.5, 2.5], [1.5, 2.5], [0.5, 2.5],
                        [0.5, 1.5], [0.5, 0.5], [1.5, 0.5], [2.5, 0.5]], np.float32)
# 贴纸整体的外框（理想坐标 0..3），即魔方面的可见范围
_GRID_CORNERS = np.array([[0, 0], [3, 0], [3, 3], [0, 3]], np.float32).reshape(-1, 1, 2)


def sticker_mask(frame):
    """贴纸像素掩码（uint8 0/1）：高饱和度（彩色贴纸）或高亮度低饱和（白色贴纸）。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    return (((s > 60) & (v > 50)) | ((v > 190) & (s < 60))).astype(np.uint8)


def sticker_candidates(frame):
    """找候选贴纸块：完全在画面内、大小与方正度像贴纸的独立色块。

    返回 [(x, y, area), ...]（各块中心坐标与面积），每张贴纸最多一个。
    """
    h, w = frame.shape[:2]
    n, labels, stats, cents = cv2.connectedComponentsWithStats(sticker_mask(frame), 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < STICKER_MIN_AREA:
            continue
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue                              # 贴到画面边缘：多半是背景，或贴纸被裁掉
        if max(bw, bh) > 0.45 * min(h, w):
            continue
        comp = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        if rw < 12 or rh < 12 or not 0.45 <= rw / rh <= 2.2:
            continue
        if cv2.contourArea(cnt) < 0.6 * rw * rh:
            continue                              # 不够方正
        out.append((float(cents[i][0]), float(cents[i][1]), float(area)))
    return _prune_candidates(out)


def _prune_candidates(cands):
    """同一张贴纸可能被上面的图案（logo/文字/划线）分成几块：合并近重合的块，只留大的。

    再清掉两类不可能是单张贴纸的块：明显偏大的（多张贴纸被糊在一起）
    和明显偏小的碎块（保证删完还剩足够多的块）。"""
    kept = []
    for c in sorted(cands, key=lambda t: -t[2]):
        x, y, a = c
        if any(np.hypot(x - k[0], y - k[1]) < 0.5 * np.sqrt(min(a, k[2])) for k in kept):
            continue                              # 与已保留的大块基本重合 = 同一张贴纸的碎块
        kept.append(c)
    if len(kept) >= 8:
        med = float(np.median([c[2] for c in kept]))
        kept = [c for c in kept if c[2] <= 2.4 * med]   # 明显偏大 = 多张贴纸连成一片
    if len(kept) > 9:
        big = [c for c in kept if c[2] >= 0.25 * kept[0][2]]
        if len(big) >= 9:
            kept = big
    return kept


def _fit_grid_points(src_pts, dst_pts):
    """给定理想网格点 ↔ 观测点的对应关系，拟合单应，返回 (平均误差, H) 或 None。"""
    src = np.asarray(src_pts, np.float32)
    dst = np.asarray(dst_pts, np.float32)
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        return None
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - dst, axis=1))), H


def _keep_better(best, cand):
    if cand is not None and (best is None or cand[0] < best[0]):
        return cand
    return best


def _quad_from_H(H):
    return cv2.perspectiveTransform(_GRID_CORNERS, H).reshape(4, 2)


def _fit_sticker_grid(pts9):
    """9 个贴纸中心点 → 拟合 3x3 网格，返回贴纸整体外框四边形（4x2）或 None。

    正常拟合 9 点；若误差过大再试"去掉一个离群点"的 8 点拟合
    （贴纸被画面裁掉一角、或被手指挡掉一部分时，该块中心会明显偏移）。
    """
    pts = np.asarray(pts9, np.float32)
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1)
    ci = int(np.argmin(d))                        # 离质心最近的点 = 中心贴纸
    ring = np.delete(pts, ci, axis=0)
    ring = ring[np.argsort(np.arctan2(ring[:, 1] - c[1], ring[:, 0] - c[0]))]
    pitch = float(np.median(np.linalg.norm(ring - c, axis=1)))   # 相邻贴纸中心距（约 1~1.4 格）
    if pitch <= 0:
        return None
    center = np.vstack([[[1.5, 1.5]], _RING_IDEAL])
    strict = None
    for k in range(8):                # 观测环的起始方向未知，逐个旋转试配
        strict = _keep_better(strict, _fit_grid_points(center, np.vstack([pts[ci], np.roll(ring, -k, axis=0)])))
    if strict is not None and strict[0] <= 0.10 * pitch:
        return _quad_from_H(strict[1])
    # 严格拟合不过关时，允许去掉一个离群点
    # （整张贴纸被画面裁掉一角、或被手指挡掉一部分时，该块中心会明显偏移）
    best = None
    for k in range(8):
        rk = np.roll(ring, -k, axis=0)
        best = _keep_better(best, _fit_grid_points(_RING_IDEAL, rk))            # 去掉中心贴纸
        for j in range(8):            # 逐个去掉一个环位贴纸
            best = _keep_better(best, _fit_grid_points(
                np.vstack([[[1.5, 1.5]], np.delete(_RING_IDEAL, j, axis=0)]),
                np.vstack([pts[ci], np.delete(rk, j, axis=0)])))
    if best is None or best[0] > 0.10 * pitch:
        return None
    return _quad_from_H(best[1])


def _fit_sticker_grid8(pts8):
    """8 个贴纸中心点（有一个贴纸被手指挡住等原因缺失）→ 拟合网格。

    两种情况：缺中心贴纸（8 点全在环上）；或中心贴纸在、环上缺一个
    （环形观测里最大的角度缺口处即为缺失的环位）。返回外框四边形或 None。
    """
    pts = np.asarray(pts8, np.float32)
    c = pts.mean(axis=0)
    ring = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    pitch = float(np.median(np.linalg.norm(ring - c, axis=1)))
    if pitch <= 0:
        return None
    best = None
    for k in range(8):                # 情况 1：8 点全是环位，缺中心贴纸
        best = _keep_better(best, _fit_grid_points(_RING_IDEAL, np.roll(ring, -k, axis=0)))
    for ci in range(8):               # 情况 2：每个点轮流当中心，环上缺一个
        others = np.delete(pts, ci, axis=0)
        ang = np.arctan2(others[:, 1] - pts[ci][1], others[:, 0] - pts[ci][0])
        so = np.argsort(ang)
        gaps = np.diff(np.r_[ang[so], ang[so][0] + 2 * np.pi])
        seq = np.roll(others[so], -(int(np.argmax(gaps)) + 1), axis=0)   # 从缺口后开始排
        for k in range(8):            # 缺口后第一个观测对应的环位序号未知
            idx = [(k + j) % 8 for j in range(7)]
            src = np.vstack([[[1.5, 1.5]], _RING_IDEAL[idx]])
            best = _keep_better(best, _fit_grid_points(src, np.vstack([[pts[ci]], seq])))
    if best is None or best[0] > 0.10 * pitch:
        return None
    return _quad_from_H(best[1])


def _order_quad(quad):
    """角点排序为 左上→右上→右下→左下。"""
    pts = np.asarray(quad, np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def _grid_lines_ok(img):
    """校正后的图里应能看到 3x3 贴纸之间的黑色分隔线（V 明显低于整体）。

    用来确认兜底检测框住的确实是魔方面而不是别的色块。"""
    side = img.shape[0]
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    med = float(np.median(v))
    if med <= 0:
        return False
    band = max(2, side // 24)
    ratios = []
    for pos in (side // 3, 2 * side // 3):
        ratios.append(float(np.median(v[:, pos - band:pos + band + 1])))
        ratios.append(float(np.median(v[pos - band:pos + band + 1, :])))
    return min(ratios) < 0.62 * med


def _silhouette_quad(frame):
    """兜底检测：把整个魔方面当作一大块色块，找它的外接四边形。

    这类拟合没有贴纸网格那么强的约束，最后用分隔线检查确认框住的是魔方面。"""
    h, w = frame.shape[:2]
    mask = sticker_mask(frame) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)   # 跨过黑色分隔线把整面连起来
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    def accept(quad):
        quad = _order_quad(quad)
        return quad if _grid_lines_ok(warp_face(frame, quad)) else None

    cnt = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)
    if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
        return None                               # 贴边：多半是背景而不是魔方
    area = cv2.contourArea(cnt)
    if area < 0.02 * h * w:
        return None
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            quad = ap.reshape(4, 2).astype(np.float32)
            sides = [float(np.linalg.norm(quad[i] - quad[(i + 1) % 4])) for i in range(4)]
            if max(sides) / max(min(sides), 1e-6) <= 2.2:
                got = accept(quad)
                if got is not None:
                    return got
    rect = cv2.minAreaRect(cnt)
    (_, _), (rw, rh), _ = rect
    if rw > 0 and rh > 0 and max(rw, rh) / min(rw, rh) <= 1.8 and area >= 0.72 * rw * rh:
        got = accept(cv2.boxPoints(rect))
        if got is not None:
            return got
    return None


def detect_face_quad(frame):
    """在画面中自动找魔方的一个面。返回排好序的 4 个角点（4x2 float32）或 None。"""
    cands = sticker_candidates(frame)
    if len(cands) >= 9:
        top = sorted(cands, key=lambda t: -t[2])[:24]   # 大块优先，控制计算量
        for i, (cx, cy, _) in enumerate(top):
            d = sorted((float(np.hypot(x - cx, y - cy)), x, y)
                       for j, (x, y, _) in enumerate(top) if j != i)
            if len(d) < 8:
                continue
            d8 = [t[0] for t in d[:8]]
            if d8[0] < 1 or max(d8) > 3.2 * d8[0]:
                continue          # 中心到 8 个近邻的距离太不整齐，不像同一面的 3x3
            quad = _fit_sticker_grid([(cx, cy)] + [(t[1], t[2]) for t in d[:8]])
            if quad is not None:
                return _order_quad(quad)
    elif len(cands) == 8:         # 有一个贴纸被手指挡住等：8 个也能定位
        quad = _fit_sticker_grid8([(x, y) for x, y, _ in cands])
        if quad is not None:
            return _order_quad(quad)
    return _silhouette_quad(frame)


def warp_face(frame, quad, side=WARP_SIDE):
    """把 quad 圈定的魔方面透视校正成 side×side 的正方形（正对视角）。"""
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], np.float32)
    m = cv2.getPerspectiveTransform(_order_quad(quad), dst)
    return cv2.warpPerspective(frame, m, (side, side))


def cell_regions(x, y, side):
    """返回 9 个格子中心采样区域的 (y0, y1, x0, x1)（行优先）。"""
    cell = side // 3
    margin = cell // 4          # 只采样每格中间约 50% 的区域
    regions = []
    for r in range(3):
        for c in range(3):
            regions.append((y + r * cell + margin,
                            y + (r + 1) * cell - margin,
                            x + c * cell + margin,
                            x + (c + 1) * cell - margin))
    return regions


def sample_region(img, x, y, side):
    """对图像里 (x, y) 起、边长 side 的正方形区域按 3x3 采样。

    返回 (3x3 颜色代号, 3x3 中位 BGR, 3x3 HSV)，全部为可 JSON 序列化的嵌套列表。
    """
    letters, medians, hsvs = [], [], []
    for (y0, y1, x0, x1) in cell_regions(x, y, side):
        patch = img[y0:y1, x0:x1].reshape(-1, 3)
        med = np.median(patch, axis=0)   # 中值抗单点反光
        bgr = tuple(int(v) for v in med)
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
        letters.append(classify_color(bgr))
        medians.append(bgr)
        hsvs.append(tuple(int(v) for v in hsv))
    return (np.array(letters).reshape(3, 3).tolist(),
            np.array(medians).reshape(3, 3, 3).tolist(),
            np.array(hsvs).reshape(3, 3, 3).tolist())


def _draw_labels(img, centers, radius, glyphs, colors):
    """在 9 个格子中心（3x3）画识别色点与字母。"""
    cell = radius * 6
    for r in range(3):
        for c in range(3):
            bgr = tuple(int(v) for v in colors[r][c])
            cx, cy = int(centers[r][c][0]), int(centers[r][c][1])
            cv2.circle(img, (cx, cy), radius, bgr, -1)
            cv2.circle(img, (cx, cy), radius, (0, 0, 0), 1)
            lum = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
            fg = (0, 0, 0) if lum > 128 else (255, 255, 255)
            org = (cx + cell // 14, cy - cell // 12)
            cv2.putText(img, glyphs[r][c], org, cv2.FONT_HERSHEY_SIMPLEX,
                        cell / 250.0, (255, 255, 255) if fg == (0, 0, 0) else (0, 0, 0),
                        4, cv2.LINE_AA)
            cv2.putText(img, glyphs[r][c], org, cv2.FONT_HERSHEY_SIMPLEX,
                        cell / 250.0, fg, 2, cv2.LINE_AA)
    return img


def draw_overlay(frame, quad, letters=None, colors=None):
    """实时画面：框出检测到的魔方面（绿框），并把每格识别结果标在对应位置上。"""
    out = frame.copy()
    if quad is None:
        return out
    quad = _order_quad(quad)
    cv2.polylines(out, [np.int32(quad)], True, (0, 255, 0), 3)
    if letters is None:
        return out
    side = float(np.mean([np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]))
    radius = max(5, int(side / 3 * 0.17))
    dst = np.array([[0, 0], [WARP_SIDE - 1, 0],
                    [WARP_SIDE - 1, WARP_SIDE - 1], [0, WARP_SIDE - 1]], np.float32)
    inv = cv2.getPerspectiveTransform(dst, quad)   # 校正图 → 原始画面
    cell = WARP_SIDE / 3
    pts = np.array([[[(c + .5) * cell, (r + .5) * cell]]
                    for r in range(3) for c in range(3)], np.float32)
    centers = cv2.perspectiveTransform(pts, inv).reshape(3, 3, 2)
    return _draw_labels(out, centers, radius, letters, colors)


def annotate_face(img, letters, colors):
    """保存用：在透视校正后的正面图上画 3x3 网格与识别结果。"""
    out = img.copy()
    side = out.shape[0]
    cell = side // 3
    for i in (1, 2):
        cv2.line(out, (i * cell, 0), (i * cell, side), (150, 150, 150), 2)
        cv2.line(out, (0, i * cell), (side, i * cell), (150, 150, 150), 2)
    centers = np.array([[[(c + .5) * cell, (r + .5) * cell]]
                        for r in range(3) for c in range(3)], np.float32).reshape(3, 3, 2)
    return _draw_labels(out, centers, max(5, cell // 6), letters, colors)


# ---------------- 界面提示 ----------------
# 窗口里的中文提示需要带中文字库的渲染，优先用 Pillow + 系统中的中文字体；
# 没有 Pillow / 找不到中文字体时自动退回 OpenCV 内置字体的英文提示。

_CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
)
_FONT_PATH = next((p for p in _CJK_FONT_CANDIDATES if _HAS_PIL and Path(p).is_file()), None)
_FONT_CACHE = {}
_TEXT_CACHE = {}


def _load_font(px):
    if not _HAS_PIL or _FONT_PATH is None:
        return None
    f = _FONT_CACHE.get(px)
    if f is None:
        try:
            f = ImageFont.truetype(_FONT_PATH, px)
        except Exception:
            return None
        _FONT_CACHE[px] = f
    return f


_CJK = _load_font(30) is not None


def _T(zh, en):
    """界面提示文案：能画中文就用中文，否则退回英文。"""
    return zh if _CJK else en


def _text_overlay(text, px, color, stroke=2):
    """把一行文字渲染成带透明底的小图（BGR+A，缓存复用），color 为 BGR。"""
    key = (text, px, tuple(color), stroke)
    got = _TEXT_CACHE.get(key)
    if got is not None:
        return got
    font = _load_font(px)
    if font is not None:
        rgb = tuple(int(v) for v in color[::-1])
        bbox = font.getbbox(text, stroke_width=stroke)
        m = stroke + 3
        w, h = bbox[2] - bbox[0] + 2 * m, bbox[3] - bbox[1] + 2 * m
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((m - bbox[0], m - bbox[1]), text, font=font,
                                fill=rgb + (255,), stroke_width=stroke,
                                stroke_fill=(0, 0, 0, 255))
        arr = np.array(im)[:, :, [2, 1, 0, 3]]          # RGBA → BGRA
    else:
        scale = px / 33.0
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        arr = np.zeros((th + base + 8, tw + 8, 4), np.uint8)
        cv2.putText(arr, text, (4, th + 4), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0, 255), 5, cv2.LINE_AA)
        cv2.putText(arr, text, (4, th + 4), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    tuple(color) + (255,), 2, cv2.LINE_AA)
    _TEXT_CACHE[key] = arr
    return arr


def _paste_bgra(frame, arr, x, y):
    h, w = arr.shape[:2]
    fh, fw = frame.shape[:2]
    x = int(np.clip(x, 0, max(0, fw - w)))
    y = int(np.clip(y, 0, max(0, fh - h)))
    if w <= 0 or h <= 0 or x >= fw or y >= fh:
        return
    roi = frame[y:y + h, x:x + w]
    a = arr[:, :, 3:4].astype(np.float32) / 255.0
    roi[:] = (roi * (1 - a) + arr[:, :, :3] * a).astype(np.uint8)


def _put(frame, text, org, px, color, stroke=2):
    _paste_bgra(frame, _text_overlay(text, px, color, stroke), org[0], org[1])


def _banner(frame, text, color, px=34, y=None):
    """画面中部横幅提示（采集成功/重复面等瞬时消息），返回横幅底部的 y（便于往下续排一行）。"""
    arr = _text_overlay(text, px, color, 3)
    h, w = arr.shape[:2]
    fh, fw = frame.shape[:2]
    x0 = max(0, (fw - w) // 2)
    y0 = int(fh * 0.38) if y is None else int(y)
    pad = 12
    bx0, by0 = max(0, x0 - pad), max(0, y0 - pad)
    bx1, by1 = min(fw - 1, x0 + w + pad), min(fh - 1, y0 + h + pad)
    roi = frame[by0:by1, bx0:bx1]
    roi[:] = (roi * 0.25).astype(np.uint8)
    cv2.rectangle(frame, (bx0, by0), (bx1, by1), color, 2)
    _paste_bgra(frame, arr, x0, y0)
    return y0 + h + pad


def _draw_center_letter(img, x, y, side, letter, bgr):
    """在进度格里画中心块的字母（选与底色对比的颜色）。"""
    lum = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
    fg = (0, 0, 0) if lum > 128 else (255, 255, 255)
    scale = side / 66.0
    (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    org = (x + (side - tw) // 2, y + (side + th) // 2)
    cv2.putText(img, letter, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 2, cv2.LINE_AA)


def draw_progress(frame, faces):
    """右下角 6 个进度格：已采集的面用其中心块颜色 + 字母填充，下一面高亮。"""
    n, sq, gap = 6, 44, 9
    fh, fw = frame.shape[:2]
    x0 = fw - 14 - (n * sq + (n - 1) * gap)
    y0 = fh - sq - 14
    nxt = len(faces) + 1
    for i in range(n):
        x = x0 + i * (sq + gap)
        cv2.rectangle(frame, (x, y0), (x + sq, y0 + sq), (0, 0, 0), -1)
        if i < len(faces):
            bgr = tuple(int(v) for v in faces[i]["bgr"][1][1])
            cv2.rectangle(frame, (x + 3, y0 + 3), (x + sq - 3, y0 + sq - 3), bgr, -1)
            _draw_center_letter(frame, x, y0, sq, faces[i]["colors"][1][1], bgr)
            cv2.rectangle(frame, (x, y0), (x + sq, y0 + sq), (255, 255, 255), 1)
        else:
            cv2.rectangle(frame, (x + 3, y0 + 3), (x + sq - 3, y0 + sq - 3), (55, 55, 55), -1)
            _draw_center_letter(frame, x, y0, sq, str(i + 1), (55, 55, 55))
            border = (0, 210, 255) if i + 1 == nxt else (90, 90, 90)
            cv2.rectangle(frame, (x, y0), (x + sq, y0 + sq), border, 3 if i + 1 == nxt else 1)


def draw_scan_hud(frame, faces, detected, auto_streak=0, auto_armed=True):
    """左上角两行状态提示 + 右上角采集进度。auto_* 为自动采集的稳定计数状态。"""
    done = len(faces)
    if done < 6:
        _put(frame, _T(f"已采集 {done}/6，本次采集第 {done + 1} 面      空格=采集  R=撤销上一面  Q=退出",
                       f"Captured {done}/6, next: face {done + 1}      SPACE=capture  R=undo  Q=quit"),
             (14, 10), 30, (255, 255, 255))
    else:
        _put(frame, _T("已采集 6/6：重拍任一面会自动更新它      空格=采集  R=撤销上一面  Q=退出保存",
                       "Captured 6/6: re-shoot any face to update      SPACE=capture  R=undo  Q=quit"),
             (14, 10), 30, (255, 255, 255))
    if detected:
        if auto_armed:
            msg = _T(f"已识别到魔方面：保持稳定 {auto_streak}/{AUTO_CAPTURE_FRAMES} 帧后自动采集（也可按空格）",
                     f"Face detected: hold steady {auto_streak}/{AUTO_CAPTURE_FRAMES} frames to auto-capture (or SPACE)")
        else:
            msg = _T("已采集：请转动魔方换下一面对准镜头（稳定后会自动采集，也可按空格）",
                     "Captured: show the next side; auto-captures when steady (or SPACE)")
        _put(frame, msg, (14, 52), 26, (110, 255, 110))
    else:
        _put(frame, _T("未检测到魔方：不用对准取景框，把魔方整个面朝向镜头、完整放进画面里",
                       "No cube yet: face the whole side to the camera inside the frame"),
             (14, 52), 26, (80, 200, 255))
    draw_progress(frame, faces)


def find_duplicate(faces, letters):
    """本次识别是否与已采集的某一面重复，返回它的下标或 None。

    现实魔方六个面的中心块颜色互不相同，所以中心色相同即可断定是同一面；
    中心色没认准时（?）退化为整面颜色完全一致才判重。"""
    center = letters[1][1]
    if center != "?":
        for i, f in enumerate(faces):
            if f["colors"][1][1] == center:
                return i
    for i, f in enumerate(faces):
        if f["colors"] == letters:
            return i
    return None


# ---------------- 启动时载入上次未通过校验的扫描结果 ----------------
# 网页导入用的校验逻辑在 scan_to_facelet.py（未通过时网页会给出「状态未通过校验」
# 之类的提示）。若上次结果没过校验，用户往往只想重拍有问题的那一面：启动扫描程序
# 时用同一套逻辑检查目录里最新的 cube_*.json，没过就询问是否载入它继续修正。

_FACE_CN = {"U": "上", "L": "左", "F": "前", "R": "右", "B": "后", "D": "下"}
_scan_to_repr = None       # scan_to_facelet.scan_to_repr，首次用到时再导入


def _call_scan_to_repr(path):
    global _scan_to_repr
    if _scan_to_repr is None:
        scan_dir = str(Path(__file__).resolve().parent)
        if scan_dir not in sys.path:
            sys.path.insert(0, scan_dir)
        from scan_to_facelet import scan_to_repr
        _scan_to_repr = scan_to_repr
    return _scan_to_repr(path)


_CENTER_OF_FACE = {"U": "W", "D": "Y", "F": "G", "B": "B", "R": "R", "L": "O"}
_BLOCK_CORNER_RE = re.compile(r"角位 ([URFDLB]{3})")
_BLOCK_EDGE_RE = re.compile(r"棱位 ([URFDLB]{2})")


def _face_tag(faces, center_letter):
    """按中心色指认载入的面：返回「第 N 面（中心X）」或 None。"""
    for f in faces:
        if f["colors"][1][1] == center_letter:
            return f"第 {f['index']} 面（中心{COLOR_NAMES.get(center_letter, center_letter)}）"
    return None


def _face_names_of_block(code):
    """块名（如 URF / UF）→ 涉及其中的面名（如上右前）。"""
    return "".join(_FACE_CN.get(ch, ch) for ch in code)


def _translate_block_reason(reason):
    """把校验原因里的块名翻成面名（角位 URF → 角位 上右前），方便用户定位到面。"""
    reason = _BLOCK_CORNER_RE.sub(lambda m: "角位 " + _face_names_of_block(m.group(1)), reason)
    return _BLOCK_EDGE_RE.sub(lambda m: "棱位 " + _face_names_of_block(m.group(1)), reason)


def _face_ids(idxs):
    """[1, 2] → 「第 1、2 面」。"""
    return "第 " + "、".join(str(i) for i in idxs) + " 面"


def _center_issue(faces):
    """中心块异常（没认出颜色 / 多面中心同色）→ (短原因, 建议重拍) 或 None。"""
    unread = [f["index"] for f in faces if f["colors"][1][1] == "?"]
    same = {}
    for f in faces:
        c = f["colors"][1][1]
        if c != "?":
            same.setdefault(c, []).append(f["index"])
    parts, advice = [], []
    if unread:
        parts.append(_face_ids(unread) + "的中心块没认出颜色")
        advice.append("重拍：" + _face_ids(unread))
    for c, idxs in same.items():
        if len(idxs) < 2:
            continue
        who = _face_ids(idxs)
        parts.append(f"{who}的中心块都识别为{COLOR_NAMES.get(c, c)}")
        advice.append(f"重拍：{who}中拍错的那一面")
    if not parts:
        return None
    return "；".join(parts), "；".join(advice)


def _diagnose(res, faces):
    """由校验结果和载入的面给出 (简短原因, 建议重拍哪些面)；通过校验时返回 (None, "")。"""
    if not res.get("ok"):
        got = _center_issue(faces)
        if got:
            return got
        return str(res.get("error") or "数据无法读取"), ""
    if res.get("valid"):
        return None, ""
    grid = res.get("faces") or {}
    miss = [cn for fc, cn in _FACE_CN.items()
            if grid.get(fc) is not None and all(ch == "" for ch in grid[fc])]
    if miss:
        return "扫描不完整，缺 " + "/".join(miss) + " 面", "载入后把缺的面补拍上即可"
    unk, unk_total = [], 0
    for fc in _FACE_CN:
        n = sum(1 for ch in (grid.get(fc) or []) if ch == "")
        if n:
            unk_total += n
            tag = _face_tag(faces, _CENTER_OF_FACE[fc]) or (_FACE_CN[fc] + "面")
            unk.append(f"{tag}{n} 格")
    if unk:
        return f"有 {unk_total} 格没认出颜色", "建议重拍：" + "、".join(unk)
    bad = {c: n for c, n in (res.get("counts") or {}).items()
           if c in "WYGRBO" and n != 9}
    if bad:
        return ("颜色数量不符（" + "、".join(f"{c}={n}" for c, n in bad.items()) + "）",
                "无法定位到具体面，建议对照网页颜色表逐面核对")
    for w in res.get("warnings") or []:
        if w.startswith("状态未通过合法性校验："):
            reason = w.replace("状态未通过合法性校验：", "").replace("；请对照魔方修正后再求解", "")
            return _translate_block_reason(reason), ""
    return "状态未通过校验", ""


def _reclassify(colors, bgr):
    """用当前阈值对存档采样值重新识别：新结果没变差才采用（与 scan_to_facelet 同规则）。"""
    try:
        again = [[classify_color(cell) for cell in row] for row in bgr]
    except Exception:  # noqa: BLE001
        return colors
    n_old = sum(row.count("?") for row in colors)
    n_new = sum(row.count("?") for row in again)
    return again if n_new <= n_old else colors


def load_previous_faces(path):
    """从扫描 JSON 重建面列表。帧图不在 JSON 里，frame 置 None（保存时不再为它出图）。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    faces = []
    for f in data.get("faces") or []:
        try:
            colors = [[str(ch) for ch in row] for row in f["colors"]]
            if len(colors) != 3 or any(len(r) != 3 for r in colors):
                continue
        except Exception:  # noqa: BLE001
            continue
        bgr = f.get("median_bgr")
        colors = _reclassify(colors, bgr)
        if not (isinstance(bgr, list) and len(bgr) == 3):
            bgr = [[[120, 120, 120]] * 3 for _ in range(3)]   # 老文件没有采样值时给个灰块占位
        faces.append({"index": int(f.get("index", len(faces) + 1)),
                      "colors": colors, "bgr": bgr, "hsv": f.get("hsv"),
                      "frame": None})
    return faces


def previous_scan_problem(out_dir):
    """检查目录里最新的 cube_*.json 是否未通过校验。

    通过校验、没有扫描文件、或数据坏到无法载入时返回 None；
    否则返回 (路径, 简短原因, 建议重拍哪些面, 可载入的面列表)。"""
    paths = sorted(Path(out_dir).glob("cube_*.json"))
    if not paths:
        return None
    path = paths[-1]
    try:
        res = _call_scan_to_repr(str(path))
        faces = load_previous_faces(path)
        brief, advice = _diagnose(res, faces)
        if brief is None:
            return None
    except Exception:  # noqa: BLE001  文件坏了或校验模块不可用：不打扰用户
        return None
    if not faces:
        return None
    return path, brief, advice, faces


def ask_load_previous(cap, window, out_dir):
    """启动时检查上一次扫描结果：未通过校验时，在画面里询问是否载入以继续修正。

    返回 (faces, note, advice)：faces=None 表示用户选择退出；空列表表示从头扫描；
    advice 为「要重拍哪些面」的提示，载入后由主界面继续显示。"""
    problem = previous_scan_problem(out_dir)
    if problem is None:
        return [], "", ""
    path, brief, advice, faces = problem
    print("检测到上一次扫描未通过校验：" + brief)
    if advice:
        print("  " + advice)
    print(f"  来源: {path.name}（已采集 {len(faces)} 面）")
    print("  按 Y = 载入它继续修正（重拍有问题的面即可）；按 N = 忽略，重新扫描；按 Q = 退出")
    shown = brief if len(brief) <= 29 else brief[:28] + "…"
    hint = _T(f"上次扫描未通过校验：{shown}", "Previous scan data failed validation")
    keys = _T("按 Y 载入并修正 / 按 N 重新扫描 / 按 Q 退出",
              "Y = load it and fix   N = start fresh   Q = quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取画面失败，摄像头可能被断开。", file=sys.stderr)
            return [], "", ""
        vis = frame.copy()
        y = _banner(vis, hint, (0, 165, 255), px=30)
        if advice and _CJK:
            arr2 = _text_overlay(advice, 26, (120, 230, 255), 2)
            _paste_bgra(vis, arr2, (vis.shape[1] - arr2.shape[1]) // 2, y + 12)
            y += 12 + arr2.shape[0]
        arr = _text_overlay(keys, 26, (255, 255, 255), 2)
        _paste_bgra(vis, arr, (vis.shape[1] - arr.shape[1]) // 2, y + 12)
        cv2.imshow(window, vis)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("y"), ord("Y")):
            print(f"已载入上次扫描的 {len(faces)} 个面：重拍有问题的面会自动更新它。")
            return faces, _T(f"已载入上次扫描的 {len(faces)} 个面：重拍有问题的面，完成按 Q 保存",
                             f"Loaded {len(faces)} faces from last scan: re-shoot bad ones, Q to save"), advice
        if key in (ord("n"), ord("N")):
            print("已忽略上次结果，从头扫描。")
            return [], "", ""
        if key in (ord("q"), 27):
            print("已退出，未载入上次结果。")
            return None, "", ""


def capture_face(cap):
    """连续取 SAMPLES_PER_CAPTURE 帧取中值合成一帧，自动找魔方面、校正并采样。

    返回 (对齐图, 3x3 代号, 3x3 BGR, 3x3 HSV)；检测不到魔方面时返回 (None, 原因)。
    """
    frames = []
    for _ in range(SAMPLES_PER_CAPTURE):
        ok, f = cap.read()
        if ok:
            frames.append(f)
    if not frames:
        return None, "没有读到有效画面"
    frame = np.median(np.stack(frames), axis=0).astype(np.uint8)
    quad = detect_face_quad(frame)
    if quad is None:
        return None, "没有检测到魔方：请让魔方整个面朝向摄像头、完整出现在画面里"
    aligned = warp_face(frame, quad)
    letters, medians, hsvs = sample_region(aligned, 0, 0, WARP_SIDE)
    return aligned, letters, medians, hsvs


def print_face(letters):
    for row in letters:
        print("  " + " ".join(row))


def cleanup_previous_scans(out_dir, keep_names):
    """删除目录里上一次扫描遗留的 cube_*.json 与 cube_*_face*.png。
    keep_names 为本次刚保存的文件名集合，其余同名格式的旧文件全部删除，返回删除数。"""
    removed = 0
    for pattern in ("cube_*.json", "cube_*_face*.png"):
        for p in Path(out_dir).glob(pattern):
            if p.is_file() and p.name not in keep_names:
                p.unlink()
                removed += 1
    return removed


def save_results(faces, out_dir, stamp, complete):
    """保存 JSON 与每面标注图（载入的面没有 frame，跳过出图）；complete 表示是否采满 6 面。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = f"cube_{stamp}"

    data = {
        "captured_at": stamp,
        "complete": complete,
        "face_count": len(faces),
        "color_legend": {k: v for k, v in COLOR_NAMES.items() if k != "?"},
        "faces": [],
        "color_counts": {k: 0 for k in "WYROBG"},
    }
    img_names = []
    for f in faces:
        for row in f["colors"]:
            for g in row:
                if g in data["color_counts"]:
                    data["color_counts"][g] += 1
        img_name = None
        if f.get("frame") is not None:
            vis = annotate_face(f["frame"], f["colors"], f["bgr"])
            img_path = out / f"{base}_face{f['index']}.png"
            cv2.imwrite(str(img_path), vis)
            img_name = img_path.name
            img_names.append(img_name)
        data["faces"].append({
            "index": f["index"],
            "colors": f["colors"],
            "median_bgr": f["bgr"],
            "hsv": f["hsv"],
            "image": img_name,
        })

    json_path = out / f"{base}.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    note = (f"（含 {len(img_names)} 张标注图）" if len(img_names) == len(faces)
            else f"（{len(img_names)}/{len(faces)} 张标注图，载入的面没有原始画面）")
    print(f"已保存: {json_path}{note}")

    if complete:
        keep = {json_path.name} | set(img_names)
        removed = cleanup_previous_scans(out, keep)
        if removed:
            print(f"已清理上一次扫描遗留的 {removed} 个文件（目录里只保留本次结果）")

        bad = {k: n for k, n in data["color_counts"].items() if n != 9}
        if bad:
            print("注意: 以下颜色出现次数不是 9，可能存在识别错误:", bad)
            print("      可按 r 重拍可疑的面（重拍最后一面），退出时会重新保存。")
        else:
            print("颜色数量检查通过（六种颜色各 9 格）。")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="魔方颜色扫描仪：摄像头逐面识别，结果存 JSON 和标注图")
    ap.add_argument("--camera", type=int, default=0, help="摄像头编号（默认 0）")
    ap.add_argument("--out", default=None, help="结果保存目录（默认脚本所在目录）")
    ap.add_argument("--debug", action="store_true", help="打印每格 HSV 原始值")
    args = ap.parse_args()
    if args.out is None:
        args.out = str(Path(__file__).resolve().parent)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {args.camera}。", file=sys.stderr)
        return 1
    # 用 MJPG 请求 720p（与相机实测兼容；不支持时自动回退）
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window = "Cube Color Scanner"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)

    print(__doc__)
    print("提示: 把魔方一个面朝向摄像头即可，程序会自动找到并对齐魔方面；")
    print("      画面里出现绿色方框并保持稳定约半秒，就会自动采集（也可按空格立即采集）。")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    faces = []
    flash_text, flash_until, flash_color = "", 0.0, (80, 200, 80)

    def flash(text, color=(80, 200, 80), seconds=3.2):
        """在画面中部显示一条临时提示。"""
        nonlocal flash_text, flash_until, flash_color
        flash_text, flash_until, flash_color = text, time.time() + seconds, color

    recheck_hint = ""          # 载入上次结果后要重拍哪些面的提示（主界面持续显示）
    auto_streak = 0            # 绿框连续稳定的帧数（自动采集用）
    auto_anchor = None         # 本轮稳定的参考框（计数起点帧的角点）
    auto_armed = True          # 是否允许自动采集：每次采集后先移开画面才重新武装
    auto_clear = 0             # 采集后绿框消失的连续帧数（达到阈值后重新武装）
    try:
        loaded, note, advice = ask_load_previous(cap, window, args.out)
        if loaded is None:
            return 0
        faces = loaded
        if note:
            flash(note, (0, 165, 255), 6.0)
        if advice and _CJK:
            recheck_hint = advice

        while True:
            ok, frame = cap.read()
            if not ok:
                print("读取画面失败，摄像头可能被断开。", file=sys.stderr)
                break

            quad = detect_face_quad(frame)
            letters = colors = None
            if quad is not None:
                aligned = warp_face(frame, quad)
                letters, medians, _ = sample_region(aligned, 0, 0, WARP_SIDE)
                colors = medians

            # 自动采集：绿框连续稳定若干帧即视为拍好，不等用户按空格
            if quad is not None:
                if auto_armed:
                    if (auto_anchor is not None
                            and float(np.abs(quad - auto_anchor).max()) <= AUTO_CAPTURE_MAX_SHIFT):
                        auto_streak += 1
                    else:
                        auto_anchor, auto_streak = quad.copy(), 1
            else:
                auto_anchor, auto_streak = None, 0
                if not auto_armed:
                    auto_clear += 1
                    if auto_clear >= AUTO_CAPTURE_CLEAR_FRAMES:
                        auto_armed, auto_clear = True, 0
            auto_fire = (auto_armed and quad is not None
                         and auto_streak >= AUTO_CAPTURE_FRAMES)

            vis = draw_overlay(frame, quad, letters, colors)
            draw_scan_hud(vis, faces, quad is not None, auto_streak, auto_armed)
            if recheck_hint:
                _put(vis, recheck_hint, (14, 90), 24, (120, 230, 255))
            if time.time() < flash_until:
                _banner(vis, flash_text, flash_color)
            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == 32 or auto_fire:  # 空格立即采集，或绿框稳定后自动采集
                if auto_fire:
                    auto_armed, auto_clear = False, 0
                    if key != 32:
                        print(f"绿框已稳定 {AUTO_CAPTURE_FRAMES} 帧，自动采集。")
                result = capture_face(cap)
                auto_streak, auto_anchor = 0, None
                if result[0] is None:
                    print("采集失败：" + result[1])
                    flash(_T("采集失败：" + result[1], "Capture failed: " + result[1]),
                          (60, 60, 230))
                    continue
                frame_c, letters, med_bgr, hsvs = result
                auto_armed, auto_clear = False, 0   # 采完先移开画面，避免稳定后马上重拍同一面
                recheck_hint = ""          # 用户已重拍，旧提示不再显示
                center_cn = COLOR_NAMES.get(letters[1][1], "?")
                print_face(letters)
                if args.debug:
                    print("  HSV 原始值:", hsvs)
                warn = "；有格子为 ?，建议重拍这面" if any("?" in row for row in letters) else ""
                dup = find_duplicate(faces, letters)
                if dup is not None:
                    old = faces[dup]
                    faces[dup] = {"index": old["index"], "colors": letters,
                                  "bgr": med_bgr, "hsv": hsvs, "frame": frame_c}
                    print(f"这一面与第 {old['index']} 面重复（中心{center_cn}），已用本次结果更新它。")
                    if warn:
                        print("注意: 有格子无法确定颜色（?），建议调整光线/角度后重拍这面。")
                    flash(_T(f"这一面已采集过（第 {old['index']} 面，中心{center_cn}）：已更新为本次结果" + warn,
                             f"Already captured (#{old['index']}, center {letters[1][1]}): updated" + warn),
                          (0, 165, 255))
                elif len(faces) >= 6:
                    print("六个面均已采集，且这一面不在其中；如某面拍错了，先按 R 撤销最近采集的面再重拍。")
                    flash(_T("六个面都已采集，这不是其中任一面：如某面拍错，按 R 撤销后再重拍",
                             "All 6 captured and this is not one of them: press R to undo"),
                          (60, 60, 230))
                else:
                    idx = len(faces) + 1
                    faces.append({"index": idx, "colors": letters,
                                  "bgr": med_bgr, "hsv": hsvs, "frame": frame_c})
                    print(f"已采集第 {idx}/6 面（中心{center_cn}）。")
                    if warn:
                        print("注意: 有格子无法确定颜色（?），建议调整光线/角度后重拍这面。")
                    if idx < 6:
                        flash(_T(f"第 {idx} 面已采集（中心{center_cn}），请转动魔方扫描下一个面" + warn,
                                 f"Face {idx} captured (center {letters[1][1]}), turn to the next side" + warn))
                        print("请转动魔方，把下一个面朝向摄像头，稳定后会自动采集（顺序任意，建议 上/下/前/后/左/右）。")
                    else:
                        save_results(faces, args.out, stamp, complete=True)
                        flash(_T("六面采集完成，已保存！按 Q 退出，或重拍任一面更新" + warn,
                                 "All 6 faces captured and saved! Q to quit, or re-shoot any face" + warn))
                        print("六个面采集完成。可重拍任一面更新它，按 q 退出（退出时会重新保存）。")

            elif key == ord("r"):
                if faces:
                    f = faces.pop()
                    auto_armed, auto_clear = True, 0   # 撤销后允许继续自动重拍
                    auto_streak, auto_anchor = 0, None
                    print(f"已撤销第 {f['index']} 面，请把要重拍的面朝向摄像头，稳定后会自动采集（也可按空格）。")
                    flash(_T(f"已撤销第 {f['index']} 面：重新朝向摄像头，稳定后会自动重拍（也可按空格）",
                             f"Face {f['index']} removed: show it again; auto-captures when steady (or SPACE)"),
                          (0, 165, 255))
                else:
                    print("还没有采集任何面。")
    except KeyboardInterrupt:
        print("\n已中断。")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not faces:
        print("未采集任何面，未保存。")
        return 0
    save_results(faces, args.out, stamp, complete=len(faces) >= 6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
